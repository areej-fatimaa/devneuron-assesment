import torch
from torchvision import transforms

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    tensor = transform(image).unsqueeze(0)
    return tensor
