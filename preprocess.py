from torchvision import transforms
from typing import Any, Callable

class ToTensorTransform:
    def __call__(self, pic: Any) -> Any:
        return transforms.ToTensor()(pic)

class ResizeTransform:
    def __init__(self, size: int):
        self.size = size
    
    def __call__(self, img: Any) -> Any:
        return transforms.Resize((self.size, self.size))(img)

class NormalizeTransform:
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor: Any) -> Any:
        return transforms.Normalize(mean=[self.mean], std=[self.std])(tensor)

class RandomHorizontalFlipTransform:
    def __call__(self, img: Any) -> Any:
        return transforms.RandomHorizontalFlip()(img)
    