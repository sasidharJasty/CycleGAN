# models_latent.py
import torch
import torch.nn as nn
from models import ResnetBlock  # reuse ResnetBlock from your previous models.py if present
import torch.nn.functional as F

# ---------- Encoder (VAE-style) ----------
class ConvEncoder(nn.Module):
    """
    Simple convolutional encoder that outputs mu and logvar.
    Input 3x256x256 -> downsample -> produce latent dim z_dim (vector).
    """
    def __init__(self, input_nc=3, ngf=64, z_dim=256):
        super().__init__()
        self.z_dim = z_dim
        layers = []
        # downsample 256 -> 128 -> 64 -> 32 -> 16 -> 8
        ch = ngf
        layers += [nn.Conv2d(input_nc, ch, kernel_size=7, stride=1, padding=3), nn.InstanceNorm2d(ch), nn.ReLU(True)]
        for i in range(4):
            layers += [nn.Conv2d(ch, ch*2, kernel_size=4, stride=2, padding=1, bias=False),
                       nn.InstanceNorm2d(ch*2), nn.ReLU(True)]
            ch *= 2
        self.conv = nn.Sequential(*layers)
        # global pooling -> latent
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_mu = nn.Linear(ch, z_dim)
        self.fc_logvar = nn.Linear(ch, z_dim)

    def forward(self, x):
        h = self.conv(x)
        g = self.pool(h).view(h.size(0), -1)
        mu = self.fc_mu(g)
        logvar = self.fc_logvar(g)
        return mu, logvar

def reparameterize(mu, logvar):
    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    return mu + eps * std

# ---------- Generator that accepts z ----------
class ResnetGeneratorLatent(nn.Module):
    """
    Modified Resnet generator: original input concatenated with broadcasted latent z
    (Replicate z spatially and concat as extra channels).
    """
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9, z_dim=256):
        super().__init__()
        self.z_dim = z_dim
        # initial conv uses input_nc + z_channels
        # we'll broadcast z to small channel map (e.g., 16 channels) to keep memory reasonable
        self.z_channels = min(64, z_dim // 4)  # heuristic
        in_ch = input_nc + self.z_channels
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(in_ch, ngf, kernel_size=7, padding=0, bias=False),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]
        # downsampling
        n_down = 2
        mult = 1
        for i in range(n_down):
            mult_prev = mult
            mult = 2 ** (i+1)
            model += [nn.Conv2d(ngf * mult_prev, ngf * mult, kernel_size=3, stride=2, padding=1, bias=False),
                      nn.InstanceNorm2d(ngf * mult),
                      nn.ReLU(True)]
        # Resnet blocks
        mult = 2 ** n_down
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult)]
        # upsampling
        for i in range(n_down):
            mult_prev = 2 ** (n_down - i)
            mult = 2 ** (n_down - i - 1)
            model += [nn.ConvTranspose2d(ngf * mult_prev, ngf * mult, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                      nn.InstanceNorm2d(ngf * mult),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]
        self.model = nn.Sequential(*model)

        # small MLP to map z vector -> z_channels spatial map
        self.z2map = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, self.z_channels * 16 * 16),  # produce 16x16 spatial map (will be upsampled)
            nn.ReLU(True)
        )

    def forward(self, x, z):
        # z: (B, z_dim) -> produce (B, z_channels, Hz, Wz) and upsample to x spatial size
        B, C, H, W = x.shape
        zmap = self.z2map(z)  # (B, z_channels*16*16)
        zmap = zmap.view(B, self.z_channels, 16, 16)
        zmap = F.interpolate(zmap, size=(H, W), mode='bilinear', align_corners=False)
        xin = torch.cat([x, zmap], dim=1)
        return self.model(xin)

# Small test instantiation
if __name__ == "__main__":
    enc = ConvEncoder()
    gen = ResnetGeneratorLatent()
    x = torch.randn(2,3,256,256)
    mu, logvar = enc(x)
    z = reparameterize(mu, logvar)
    out = gen(x, z)
    print(out.shape)  # expected (2,3,256,256)
