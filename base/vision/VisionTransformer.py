import os

from base.config.Config import GPT2_CONFIG_124M
from base.vision.Util import  save_images

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# base on https://github.com/google-research/vision_transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

def weight_standardize(w, axis, eps):
    """Subtracts mean and divides by standard deviation."""
    w_mean = torch.mean(w, dim=axis, keepdim=True)
    w_std = torch.std(w, dim=axis, keepdim=True)
    w = (w - w_mean) / (w_std + eps)
    return w


class StdConv(nn.Conv2d):
    """Convolution with weight standardization."""

    def reset_parameters(self):
        super().reset_parameters()
        self.weight.data = weight_standardize(self.weight.data, axis=[0, 1, 2], eps=1e-5)


class ResidualUnit(nn.Module):
    """Bottleneck ResNet block."""

    def __init__(self, in_channels, features, strides=(1, 1)):
        super(ResidualUnit, self).__init__()
        self.strides = strides
        self.features = features
        self.needs_projection = in_channels != features * 4 or strides != (1, 1)

        if self.needs_projection:
            self.conv_proj = StdConv(in_channels, features * 4, kernel_size=1, stride=strides, bias=False)
            self.gn_proj = nn.GroupNorm(32, features * 4)  # assuming 32 groups as in typical ResNet

        self.conv1 = StdConv(in_channels, features, kernel_size=1, stride=1, bias=False)
        self.gn1 = nn.GroupNorm(32, features)
        self.conv2 = StdConv(features, features, kernel_size=3, stride=strides, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(32, features)
        self.conv3 = StdConv(features, features * 4, kernel_size=1, stride=1, bias=False)
        self.gn3 = nn.GroupNorm(32, features * 4)

    def forward(self, x):
        residual = x
        if self.needs_projection:
            residual = self.conv_proj(residual)
            residual = self.gn_proj(residual)

        y = self.conv1(x)
        y = self.gn1(y)
        y = nn.ReLU(inplace=True)(y)
        y = self.conv2(y)
        y = self.gn2(y)
        y = nn.ReLU(inplace=True)(y)
        y = self.conv3(y)
        y = self.gn3(y)

        y += residual
        y = nn.ReLU(inplace=True)(y)
        return y

class ResNetStage(nn.Module):
    """A ResNet stage."""

    def __init__(self, in_channels, block_size, nout, first_stride):
        super(ResNetStage, self).__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(ResidualUnit(in_channels, nout, strides=first_stride))
        for _ in range(1, block_size):
            self.blocks.append(ResidualUnit(nout * 4, nout, strides=(1, 1)))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class IdentityLayer(nn.Module):
    """Identity layer, convenient for giving a name to an array."""

    def forward(self, x):
        return x

class AddPositionEmbs(nn.Module):
    """Adds learned positional embeddings to the inputs."""

    def __init__(self, num_embeddings, embedding_dim):
        super(AddPositionEmbs, self).__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_embeddings, embedding_dim))

    def forward(self, inputs):
        """Applies the AddPositionEmbs module.

        Args:
            inputs: Inputs to the layer.

        Returns:
            Output tensor with shape `(batch_size, timesteps, embedding_dim)`.
        """
        assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                                  ' but it is: %d' % inputs.ndim)
        return inputs + self.pos_embedding


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    def __init__(self, mlp_dim, out_dim=None, dropout_rate=0.1):
        super(MlpBlock, self).__init__()
        self.mlp_dim = mlp_dim
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate

        self.fc1 = nn.Linear(mlp_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim if out_dim else mlp_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs, deterministic=True):
        """Applies Transformer MlpBlock module."""
        x = F.gelu(self.fc1(inputs))
        if not deterministic:
            x = self.dropout(x)
        output = self.fc2(x)
        if not deterministic:
            output = self.dropout(output)
        return output

class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(self, num_layers, mlp_dim, num_heads, dropout_rate=0.1, attention_dropout_rate=0.1, add_position_embedding=True, num_patches=None, hidden_size=None, cls_token_size=None):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.add_position_embedding = add_position_embedding

        if add_position_embedding:
            self.pos_embed = AddPositionEmbs(num_embeddings=num_patches+cls_token_size, embedding_dim=hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder_blocks = nn.ModuleList([
            Encoder1DBlock(mlp_dim, num_heads, dropout_rate, attention_dropout_rate, embedding_dim=hidden_size)
            for _ in range(num_layers)
        ])
        self.layernorm = nn.LayerNorm(hidden_size)

    def forward(self, x, train=True):
        assert x.ndim == 3, f'Expected (batch, len, emb) got {x.shape}'

        if self.add_position_embedding:
            x = self.pos_embed(x)
        x = self.dropout(x) if train else x

        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, deterministic=not train)

        encoded = self.layernorm(x)
        return encoded



class Encoder1DBlock(nn.Module):
    """Transformer encoder layer."""

    def __init__(self, mlp_dim, num_heads, dropout_rate=0.1, attention_dropout_rate=0.1, embedding_dim=None):
        super(Encoder1DBlock, self).__init__()
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

        self.layernorm1 = nn.LayerNorm(embedding_dim)
        self.mha = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=attention_dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.layernorm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MlpBlock(embedding_dim, dropout_rate=dropout_rate)

    def forward(self, x, deterministic=True):
        # Attention block
        assert x.ndim == 3, f'Expected (batch, seq, hidden) got {x.shape}'
        x_norm = self.layernorm1(x)
        attn_output, _ = self.mha(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_output)

        # MLP block
        y = self.layernorm2(x)
        y = self.mlp(y, deterministic=deterministic)

        return x + y


class VisionTransformer(nn.Module):
    """Vision Transformer."""

    def __init__(self, num_classes, patches, transformer, hidden_size, resnet=None, representation_size=None,
                 classifier='token', head_bias_init=0.0, model_name=None):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.patches = patches
        self.hidden_size = hidden_size
        self.resnet = resnet
        self.representation_size = representation_size
        self.classifier = classifier
        self.head_bias_init = head_bias_init
        self.model_name = model_name

        if resnet is not None:
            width = int(64 * resnet['width_factor'])
            self.resnet_stem = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=width, kernel_size=7, stride=2, padding=3, bias=False),
                nn.GroupNorm(32, width),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            self.resnet_stages = nn.ModuleList()
            for i, block_size in enumerate(resnet['num_layers']):
                stage = ResNetStage(
                    block_size=block_size,
                    nout=width * (2 ** i),
                    first_stride=(1, 1) if i == 0 else (2, 2)
                )
                self.resnet_stages.append(stage)

        self.embedding = nn.Conv2d(in_channels=3, out_channels=hidden_size, kernel_size=patches['size'],
                                   stride=patches['size'], padding=0)
        num_patches = (patches['image_size'] // patches['size']) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_size))
        cls_token_size = 1
        self.cls_token = nn.Parameter(torch.zeros(1, cls_token_size, hidden_size))
        self.pos_drop = nn.Dropout(p=transformer['dropout_rate'])

        self.encoder = Encoder(
            num_layers=transformer['num_layers'],
            mlp_dim=transformer['mlp_dim'],
            num_heads=transformer['num_heads'],
            dropout_rate=transformer['dropout_rate'],
            attention_dropout_rate=transformer['attention_dropout_rate'],
            num_patches=num_patches,
            hidden_size=hidden_size,
            cls_token_size=cls_token_size
        )

        if representation_size is not None:
            self.pre_logits = nn.Linear(hidden_size, representation_size)
            self.tanh = nn.Tanh()
        else:
            self.pre_logits = IdentityLayer()

        self.head = nn.Linear(representation_size if representation_size is not None else hidden_size, num_classes)
        nn.init.constant_(self.head.bias, head_bias_init)

    def forward(self, x):
        B = x.shape[0]

        if self.resnet is not None:
            x = self.resnet_stem(x)
            for stage in self.resnet_stages:
                x = stage(x)

        x = self.embedding(x)  # (B, hidden_size, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, hidden_size)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.encoder(x, train=self.training)

        if self.classifier == 'token':
            x = x[:, 0]
        elif self.classifier == 'gap':
            x = x.mean(dim=1)
        else:
            raise ValueError(f'Invalid classifier={self.classifier}')

        if self.representation_size is not None:
            x = self.tanh(self.pre_logits(x))
        else:
            x = self.pre_logits(x)

        x = self.head(x)
        return x


# CIFAR-10 Data Preparation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = datasets.CIFAR10(root='/workspace/data/cifar10', train=True, download=True, transform=transform_train)
val_dataset = datasets.CIFAR10(root='/workspace/data/cifar10', train=False, download=True, transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

# Model, Loss, Optimizer, and Device Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VisionTransformer(
    num_classes=10,  # Number of classes in CIFAR-10
    patches={'size': 4, 'image_size': 32},
    transformer={
        'num_layers': 12,
        'mlp_dim': 3072,
        'num_heads': 12,
        'dropout_rate': 0.1,
        'attention_dropout_rate': 0.1
    },
    hidden_size=768
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
scaler = GradScaler()


# Training Function
def train(model, train_loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in tqdm(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / total, correct / total


# Validation Function
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return running_loss / total, correct / total


# Training Loop
num_epochs = 1000
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, scaler)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    scheduler.step()
    print(
        f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    # Save checkpoint
    checkpoint_path = f'./checkpoint_epoch_{epoch + 1}.pth'
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
    }, checkpoint_path)
    print(f'Checkpoint saved at {checkpoint_path}')


# Function to display an image (for debugging/visualization purposes)
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()