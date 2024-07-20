import os
from datasets import load_dataset
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


def save_images(split, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataset = load_dataset('imagenet-21k', split=split)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    for i, item in enumerate(tqdm(dataset)):
        image = Image.open(item['file']).convert('RGB')
        image = transform(image)
        label = item['label']

        image_path = os.path.join(save_dir, f"{i}.pt")
        label_path = os.path.join(save_dir, f"{i}.txt")

        torch.save(image, image_path)
        with open(label_path, 'w') as f:
            f.write(str(label))