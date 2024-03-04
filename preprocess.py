from torchvision import transforms
from typing import Any, Callable

def to_tensor_transform(pic: Any) -> Any:
    return transforms.ToTensor()(pic)

def resize_transform(img: Any) -> Any:
    return transforms.Resize((256, 256))(img)

def normalize_transform(tensor: Any) -> Any:
    return transforms.Normalize(mean=[0.5], std=[0.5])(tensor)

def random_horizontal_flip_transform(img: Any) -> Any:
    return transforms.RandomHorizontalFlip()(img)
    