# losses.py
import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F

# ---------- VGG perceptual loss ----------
class VGGPerceptualLoss(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features.to(device).eval()
        self.device = device
        # use layers up to relu4_2 for perceptual features
        self.slice = nn.Sequential()
        for x in range(21):  # relu4_2
            self.slice.add_module(str(x), vgg[x])
        for p in self.slice.parameters():
            p.requires_grad = False

    def forward(self, x, y):
        # x, y in [-1,1] -> convert to VGG input range [0,1] then normalize with ImageNet stats
        def prep(t):
            t = (t + 1) / 2  # [0,1]
            mean = torch.tensor([0.485, 0.456, 0.406], device=t.device).view(1,3,1,1)
            std = torch.tensor([0.229, 0.224, 0.225], device=t.device).view(1,3,1,1)
            return (t - mean) / std
        fx = self.slice(prep(x))
        fy = self.slice(prep(y))
        return F.l1_loss(fx, fy)

# ---------- KL divergence ----------
def kl_loss(mu, logvar):
    # returns batch mean KL
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

# ---------- CLIP intent loss wrapper ----------
# NOTE: requires the 'clip' package (pip install git+https://github.com/openai/CLIP.git)
try:
    import clip
    CLIP_AVAILABLE = True
except Exception:
    CLIP_AVAILABLE = False

class CLIPIntentLoss:
    """
    Encodes a text prompt via CLIP and computes cosine distance between 
    CLIP(image) and CLIP(text). Minimizing 1 - cos similarity is the loss.
    """
    def __init__(self, device='cpu', clip_model_name="ViT-B/32"):
        if not CLIP_AVAILABLE:
            raise RuntimeError("CLIP not installed. pip install git+https://github.com/openai/CLIP.git")
        self.device = device
        self.model, self.preprocess = clip.load(clip_model_name, device=device)
        self.model.eval()

    @torch.no_grad()
    def encode_text(self, text_list):
        tokens = clip.tokenize(text_list).to(self.device)
        with torch.no_grad():
            txt_feats = self.model.encode_text(tokens)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
        return txt_feats

    def loss_image_text(self, images, text_feats):
        """
        images: tensor [B,3,H,W] in [-1,1]
        text_feats: [T, D] (we'll allow single prompt or list)
        We compute average over prompts if multiple.
        """
        imgs = (images + 1) / 2  # [0,1]
        # preprocess as CLIP expects: the loaded preprocess does transforms on PIL images,
        # but we can manually normalize to CLIP imagenet-like stats:
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device).view(1,3,1,1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device).view(1,3,1,1)
        imgs = (imgs - mean) / std
        img_feats = self.model.encode_image(imgs)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        # compute cosine similarity: (B, D) x (T, D) -> (B, T)
        sim = img_feats @ text_feats.t()
        # maximize similarity -> minimize (1 - sim)
        # if multiple text prompts, take the max sim for each image (best matching prompt),
        # or take mean. We'll use mean(sim across prompts).
        sim_mean = sim.mean(dim=1)
        loss = torch.mean(1 - sim_mean)
        return loss
