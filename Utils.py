# utils.py
import torch
import itertools
import os
from torchvision.utils import save_image
import numpy as np

def weights_init(net):
    for m in net.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, torch.nn.InstanceNorm2d):
            if m.weight is not None:
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)

class ImagePool():
    """Image buffer for discriminator as in original CycleGAN."""
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.images = []

    def query(self, images):
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if len(self.images) < self.pool_size:
                self.images.append(image)
                return_images.append(image)
            else:
                p = np.random.uniform()
                if p > 0.5:
                    # use image from pool
                    idx = int(np.random.uniform(0, len(self.images)))
                    tmp = self.images[idx].clone()
                    self.images[idx] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return torch.cat(return_images, 0)

def save_sample(A, B, fake_B, fake_A, out_dir, step):
    os.makedirs(out_dir, exist_ok=True)
    # Denormalize from [-1,1] to [0,1] for saving
    def denorm(x): return (x + 1) / 2
    # Save a grid: A | fake_B | B | fake_A
    grid = torch.cat([A.cpu(), fake_B.cpu(), B.cpu(), fake_A.cpu()], dim=0)
    save_image(denorm(grid), os.path.join(out_dir, f'sample_{step:06d}.png'), nrow=A.size(0))
