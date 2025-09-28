# datasets.py
import random
from PIL import Image
from pathlib import Path
import torch
from torchvision import transforms
from torch.utils.data import Dataset

IMG_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

def list_images(folder):
    p = Path(folder)
    return [x for x in sorted(p.glob('*')) if x.suffix.lower() in IMG_EXTS]

class UnalignedImageDataset(Dataset):
    """
    Expects:
      root/trainA, root/trainB, root/testA, root/testB
    Returns a tuple (A_image, B_image)
    """
    def __init__(self, root, phase='train', transform=None):
        self.root = Path(root)
        self.phase = phase
        self.dir_A = self.root / f"{phase}A"
        self.dir_B = self.root / f"{phase}B"
        self.A_paths = list_images(self.dir_A)
        self.B_paths = list_images(self.dir_B)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = transform or self.default_transform()
        if self.A_size == 0 or self.B_size == 0:
            raise RuntimeError(f"No images found in {self.dir_A} or {self.dir_B}")

    def default_transform(self):
        # match common CycleGAN preprocessing: resize -> random crop -> normalize to [-1,1]
        transform_list = [
            transforms.Resize(286, interpolation=Image.BICUBIC),
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ]
        return transforms.Compose(transform_list)

    def __len__(self):
        return max(self.A_size, self.B_size)

    def __getitem__(self, idx):
        A_path = self.A_paths[idx % self.A_size]
        B_path = self.B_paths[random.randint(0, self.B_size - 1)]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        A = self.transform(A_img)
        B = self.transform(B_img)
        return {'A': A, 'B': B, 'A_path': str(A_path), 'B_path': str(B_path)}
