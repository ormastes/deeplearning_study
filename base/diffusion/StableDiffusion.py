import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import random
import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Import necessary components from your modules
from base.gpt.MultiHeadAttention import MultiHeadAttention
from base.embedding.SequencePositionalEmbedding import SinusoidalPositionalEmbedding



# Configuration
class StableDiffusionConfig:
    def __init__(self):
        self.time_steps = None
        self.batch_size = 64
        self.epochs = 10
        self.lr = 1e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.in_channel = 1  # MNIST is grayscale
        self.out_channel = self.in_channel
        self.context_len = 1024  # just a large value
        self.num_enc_layers = 3
        self.num_dec_layers = 3
        self.num_layers = self.num_enc_layers + self.num_dec_layers

        self.time_steps = 1000
        self.channels = [64, 128, 256, 512, 512, 384]
        self.attention_enable = [False, True, False, False, False, True]
        self.upscale = [False, False, False, True, True, True]
        self.num_unet_group = 32
        self.conv_kernel_size = 3
        self.conv_stride = 1
        self.conv_padding = 1

        self.drop_rate = 0.1
        self.num_heads = 8
        self.qkv_bias = False
        self.seq_first = False
        self.reverse_position_embedding = False
        self.linformer_factor = 1.0
        self.attention_groups = 1
        self.attention = MultiHeadAttention
        self.attention_window = 0
        self.attention_dilation = 1  # TODO
        self.alibi = None


# GroupNorm layer
class GroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super(GroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_channels))
            self.bias = nn.Parameter(torch.Tensor(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x):
        assert x.shape[1] == self.num_channels, "Number of channels should be the same as the number of channels in the layer"
        x = x.view(-1, self.num_groups, self.num_channels // self.num_groups, *x.shape[2:])
        grouped_dim =[i+3 for i in range(len(x.shape)-3)]
        mean = x.mean(dim=grouped_dim, keepdim=True)
        var = x.var(dim=grouped_dim, unbiased=False, keepdim=True)
        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(-1, self.num_channels, *x.shape[3:])
        if self.affine:
            x = x * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return x


# Residual Block
class ResBlock(nn.Module):
    def __init__(self, channel, config):
        super(ResBlock, self).__init__()
        self.channel = channel
        self.config = config

        self.norm1 = GroupNorm(config.num_unet_group, channel)
        self.conv1 = nn.Conv2d(channel, channel, config.conv_kernel_size, config.conv_stride, config.conv_padding)
        self.relu1 = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(config.drop_rate)
        self.norm2 = GroupNorm(config.num_unet_group, channel)
        self.conv2 = nn.Conv2d(channel, channel, config.conv_kernel_size, config.conv_stride, config.conv_padding)
        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x += identity
        return x


# Convolutional Block with LoRA
class ConvLoraBlock(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channel, config):
        super(ConvLoraBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.mid_channel = mid_channel
        self.config = config

        self.conv1 = nn.Conv2d(in_channel, mid_channel, config.conv_kernel_size, config.conv_stride,
                               config.conv_padding)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channel, out_channel, config.conv_kernel_size, config.conv_stride,
                               config.conv_padding)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


# UNet Layer
class UNetLayer(nn.Module):
    def __init__(self, channel, upscale, attention_enable, config):
        super(UNetLayer, self).__init__()
        self.channel = channel
        self.upscale = upscale
        self.attention_enable = attention_enable
        self.config = config

        self.res_block1 = ResBlock(channel, config)
        self.res_block2 = ResBlock(channel, config)
        if upscale:
            self.conv_scale = nn.ConvTranspose2d(channel, channel // 2, kernel_size=4, stride=2, padding=1)
        else:
            self.conv_scale = nn.Conv2d(channel, channel * 2, kernel_size=3, stride=2, padding=1)
        if attention_enable:
            self.attention = MultiHeadAttention(embed_dim=channel, num_heads=config.num_heads,
                                                drop_rate=config.drop_rate, qkv_bias=config.qkv_bias,
                                                seq_first=config.seq_first, config=config)
        else:
            self.attention = None

    def forward(self, x, embedding=None):
        if embedding is not None:
            x = x + embedding[:, :x.shape[1], :, :]
        x = self.res_block1(x)

        if self.attention is not None:
            x = self.attention(x)

        if embedding is not None:
            x = x + embedding[:, :x.shape[1], :, :]
        x = self.res_block2(x)
        y = self.conv_scale(x)
        residual = x
        return y, residual


# UNet Model
class UNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.num_layers == len(config.channels) == len(config.attention_enable) == len(config.upscale), \
            "Length of channels, attention_enable, upscale should be the same"
        self.num_layers = config.num_layers
        self.num_enc_layers = config.num_enc_layers
        self.num_dec_layers = config.num_dec_layers
        self.in_channel = config.in_channel
        self.out_channel = config.out_channel
        self.channels = config.channels
        self.attention_enable = config.attention_enable
        self.upscale = config.upscale

        self.embedding = SinusoidalPositionalEmbedding(config.context_len, max(self.channels), config)
        self.shallow_conv = nn.Conv2d(self.in_channel, self.channels[0], config.conv_kernel_size,
                                      config.conv_stride, config.conv_padding)

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(self.num_enc_layers):
            layer = UNetLayer(config.channels[i], config.upscale[i], config.attention_enable[i], config)
            self.encoder.append(layer)
        for i in range(self.num_dec_layers):
            idx = i + self.num_enc_layers
            layer = UNetLayer(config.channels[idx], config.upscale[idx], config.attention_enable[idx], config)
            self.decoder.append(layer)
        out_channels = config.channels[-1]//2 + config.channels[0]
        self.output = ConvLoraBlock(out_channels,  self.out_channel, out_channels//2, config)

    def forward(self, x, t):
        x = self.shallow_conv(x)
        residuals = []
        for i in range(self.num_enc_layers):
            embedding = self.embedding(x, t) if self.embedding is not None else None
            x, residual = self.encoder[i](x, embedding)
            residuals.append(residual)
        for i in range(self.num_dec_layers):
            embedding = self.embedding(x, t) if self.embedding is not None else None
            x, _ = self.decoder[i](x, embedding)
            residual = residuals.pop()
            x = torch.cat((x, residual), dim=1)
        x = self.output(x)
        return x


# Set random seed for reproducibility
def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Beta schedule for DDPM
class BETA_SCHEDULE:
    LINEAR = "linear"


# DDPM Model
class DDPM(nn.Module):
    def __init__(self, model, time_steps=1000, beta_schedule=BETA_SCHEDULE.LINEAR, linear_start=1e-4, linear_end=2e-2):
        super(DDPM, self).__init__()
        self.time_steps = time_steps
        self.model = model
        self.betas = torch.tensor(DDPM.make_beta_schedule(beta_schedule, time_steps, linear_start, linear_end), dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return super().to(device)

    @staticmethod
    def make_beta_schedule(schedule, num_timesteps, start, end):
        if schedule == BETA_SCHEDULE.LINEAR:
            betas = np.linspace(start, end, num_timesteps)
        return betas

    def forward(self, x, t):
        return self.p_losses(x, t)

    def p_losses(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_previous = self.q_sample(x_start=x_start, t=t-1, noise=noise)
        model_out = self.model(x_noisy, t)
        loss = nn.functional.mse_loss(model_out, x_previous)
        return loss

    def q_sample(self, x_start, t, noise):
        return self.sqrt_alphas_cumprod[t][:, None, None, None] * x_start + self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None] * noise

def enable_debugging():
    torch.autograd.set_detect_anomaly(True)

def enable_tf32():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def performance():
    enable_tf32()
    # Enable cuDNN benchmark mode and other fast options
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.fastest = True
    torch.backends.cudnn.deterministic = False


performance()

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)
set_seed(42)

# Load MNIST dataset and resize to 32x32
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

model_path = 'stable_diffusion_saved_model.pth'

# using bfloat16
base_type = torch.bfloat16
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

# Training setup
config = StableDiffusionConfig()
device = config.device
unet = UNet(config)
ddpm = DDPM(unet, time_steps=config.time_steps)
ddpm.to(device).to(base_type)
optimizer = torch.optim.Adam(ddpm.parameters(), lr=config.lr)

# Load saved model if exists
if os.path.isfile(model_path):
    checkpoint = torch.load(model_path)
    ddpm.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Loaded saved model from epoch {start_epoch}")
else:
    start_epoch = 0

# Function to visualize results
def show_images(images, title="Images"):
    images = (images + 1) / 2  # unnormalize
    grid = torchvision.utils.make_grid(images.cpu().detach())
    plt.figure(figsize=(32, 32))
    plt.title(title)
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)), cmap="gray")
    plt.show()

# Training loop
for epoch in range(config.epochs):
    for i, data in enumerate(trainloader, 0):
        inputs, _ = data
        inputs = inputs.to(device).to(base_type)
        optimizer.zero_grad()
        time = torch.randint(0, config.time_steps, (inputs.shape[0],), device=device).long()
        loss = ddpm(inputs, time)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"Epoch {epoch + 1}, Step {i + 1}, Loss: {loss.item()}")

    # Save model after each epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': ddpm.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, model_path)
    print(f"Model saved after epoch {epoch + 1}")

    # Generate and display images after each epoch
    with torch.no_grad():
        ddpm.eval()
        noise = torch.randn(1, 1, 32, 32).to(device).to(base_type)
        images = torch.tensor([]).cpu()
        # 0 to config.time_steps - 1
        for i in reversed(range(config.time_steps)):
            # one element list
            a_time = torch.tensor([i], device=device).long()
            generated_image = ddpm.model(noise, a_time)
            noise = generated_image
            # append to input
            images = torch.cat((images, generated_image.cpu()), dim=0)

        show_images(images, f"Generated Images at Epoch {epoch + 1}")
