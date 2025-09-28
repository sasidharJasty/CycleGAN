# test.py
import torch
from models import ResnetGenerator
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from pathlib import Path
import argparse
import os

def load_img(path, size=256):
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=Image.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    return transform(img).unsqueeze(0)

def denorm(t):
    return (t + 1) / 2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='./datasets/renaissance')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--which_direction', type=str, default='A2B')  # or B2A
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--out_dir', type=str, default='./test_results')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    netG_A2B = ResnetGenerator().to(device)
    netG_B2A = ResnetGenerator().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    netG_A2B.load_state_dict(ckpt['netG_A2B'])
    netG_B2A.load_state_dict(ckpt['netG_B2A'])
    netG_A2B.eval(); netG_B2A.eval()

    src_dir = Path(args.dataroot) / f'{args.phase}A' if args.which_direction == 'A2B' else Path(args.dataroot) / f'{args.phase}B'
    os.makedirs(args.out_dir, exist_ok=True)
    for p in src_dir.glob('*'):
        if p.suffix.lower() not in ['.jpg','.jpeg','.png']: continue
        img = load_img(p).to(device)
        with torch.no_grad():
            if args.which_direction == 'A2B':
                fake = netG_A2B(img)
            else:
                fake = netG_B2A(img)
        out_path = Path(args.out_dir) / p.name
        save_image(denorm(fake.cpu().squeeze(0)), str(out_path))

    print("Done. Results in:", args.out_dir)

if __name__ == '__main__':
    main()
