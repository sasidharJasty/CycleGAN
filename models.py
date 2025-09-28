# models.py
import torch
import torch.nn as nn
import functools

# ---------- helper blocks ----------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, padding, norm=True, activation=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=not norm)]
        if norm:
            layers.append(nn.InstanceNorm2d(out_ch))
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(dim),
        )
    def forward(self, x):
        return x + self.conv_block(x)

# ---------- Generator: ResNet-based (9 blocks) ----------
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9):
        assert(n_blocks >= 0)
        super().__init__()
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=False),
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

    def forward(self, input):
        return self.model(input)

# ---------- Discriminator: PatchGAN ----------
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super().__init__()
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
