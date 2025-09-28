# explain.py
import torch
from torchvision.utils import save_image
import os
import torch.nn.functional as F
from PIL import Image
import numpy as np

def denorm(t): return (t + 1) / 2

def save_explain_maps(real_A, fake_B, prompt, generator, clip_loss, encoder=None, out_dir='./explain', device='cpu', step=0):
    """
    Saves:
    - pixel difference map (abs(real_A - fake_B))
    - CLIP saliency map: grad of similarity w.r.t input pixels (absolute, aggregated channels)
    - latent map: per-channel variance (if encoder provided)
    """
    os.makedirs(out_dir, exist_ok=True)
    B = real_A.size(0)
    # 1) pixel difference
    diff = torch.abs(real_A - fake_B)
    diff_grid = diff.cpu()
    save_image(denorm(diff_grid), os.path.join(out_dir, f'diff_{step:06d}.png'), nrow=B)

    # 2) CLIP saliency (if available)
    if clip_loss is not None:
        # compute grad of similarity between fake_B and prompt text w.r.t. fake_B
        # enable grads
        fake = fake_B.clone().detach().requires_grad_(True)
        text_feats = clip_loss.encode_text([prompt]).to(device)
        imgs = (fake + 1)/2
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1,3,1,1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1,3,1,1)
        imgs_norm = (imgs - mean)/std
        img_feats = clip_loss.model.encode_image(imgs_norm)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        txt = text_feats
        sim = (img_feats @ txt.t()).mean()
        sim.backward()
        saliency = fake.grad.abs().sum(dim=1, keepdim=True)  # (B,1,H,W)
        # normalize each image
        saliency = (saliency - saliency.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0])
        saliency = saliency / (saliency.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-8)
        save_image(saliency.cpu(), os.path.join(out_dir, f'clip_saliency_{step:06d}.png'), nrow=B)

    # 3) latent stats (if encoder available)
    if encoder is not None:
        with torch.no_grad():
            mu, logvar = encoder(fake_B.detach())
            # compute per-channel magnitude and save as a heatmap-like image by repeating
            stats = mu.abs().mean(dim=1, keepdim=True)  # (B,1)
            # make a small tiled image representing magnitude
            stats_map = stats.unsqueeze(-1).unsqueeze(-1).repeat(1,1,64,64)
            save_image(stats_map.cpu(), os.path.join(out_dir, f'latent_stats_{step:06d}.png'), nrow=B)
