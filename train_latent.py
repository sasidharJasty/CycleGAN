# train_latent.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import UnalignedImageDataset
from models_latent import ConvEncoder, reparameterize, ResnetGeneratorLatent
from models import NLayerDiscriminator  # reuse from your models.py
from utils import weights_init, ImagePool, save_sample
from losses import VGGPerceptualLoss, kl_loss, CLIPIntentLoss, CLIP_AVAILABLE
import argparse
from tqdm import tqdm
import itertools
import os
from explain import save_explain_maps  # helper to save saliency and diffs

# Make sure save dir exists


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataroot', type=str, default='./datasets/renaissance')
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--n_epochs', type=int, default=60)
    p.add_argument('--n_epochs_decay', type=int, default=60)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--lambda_cycle', type=float, default=10.0)
    p.add_argument('--lambda_id', type=float, default=0.5)
    p.add_argument('--lambda_kl', type=float, default=0.01)
    p.add_argument('--lambda_latent', type=float, default=1.0)
    p.add_argument('--lambda_perc', type=float, default=1.0)
    p.add_argument('--lambda_clip', type=float, default=1.0)
    p.add_argument('--pool_size', type=int, default=50)
    p.add_argument('--checkpoint_dir', type=str, default='./checkpoints/renaissance_latent')
    p.add_argument('--sample_dir', type=str, default='./samples_latent')
    p.add_argument('--prompt', type=str, default="a renaissance oil painting")
    return p.parse_args()

def make_valid_tensor(bs, device, feat_shape=(1,30,30)):
    return torch.ones((bs, *feat_shape), device=device)

def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    dataset = UnalignedImageDataset(args.dataroot, phase='train')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    # models: encoders and gens
    enc_A = ConvEncoder().to(device)
    enc_B = ConvEncoder().to(device)
    netG_A2B = ResnetGeneratorLatent().to(device)
    netG_B2A = ResnetGeneratorLatent().to(device)
    netD_A = NLayerDiscriminator().to(device)
    netD_B = NLayerDiscriminator().to(device)

    for net in [enc_A, enc_B, netG_A2B, netG_B2A, netD_A, netD_B]:
        weights_init(net)

    # losses
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_id = nn.L1Loss()
    perc_loss = VGGPerceptualLoss(device=device)
    clip_loss = None
    if CLIP_AVAILABLE:
        clip_loss = CLIPIntentLoss(device=device)
        text_feats = clip_loss.encode_text([args.prompt]).to(device)
    else:
        print("Warning: CLIP not available. Install it to use intent loss.")

    # optimizers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters(),
                                                  enc_A.parameters(), enc_B.parameters()),
                                   lr=args.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # schedulers
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch - args.n_epochs) / float(args.n_epochs_decay + 1)
        return lr_l
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
    scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=lambda_rule)
    scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=lambda_rule)

    fake_pool_A = ImagePool(args.pool_size)
    fake_pool_B = ImagePool(args.pool_size)

    step = 0
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    for epoch in range(1, args.n_epochs + args.n_epochs_decay + 1):
        loop = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        for batch in loop:
            real_A = batch['A'].to(device)
            real_B = batch['B'].to(device)
            bs = real_A.size(0)
            valid = make_valid_tensor(bs, device)

            # ---------- encode to latents ----------
            mu_A, logvar_A = enc_A(real_A)
            z_A = reparameterize(mu_A, logvar_A)
            mu_B, logvar_B = enc_B(real_B)
            z_B = reparameterize(mu_B, logvar_B)

            # ------------------
            #  Train Generators + Encoders (VAE parts)
            # ------------------
            optimizer_G.zero_grad()

            # identity losses (if used)
            idt_B = netG_A2B(real_B, z_B)
            idt_A = netG_B2A(real_A, z_A)
            loss_idt_B = criterion_id(idt_B, real_B) * args.lambda_cycle * args.lambda_id
            loss_idt_A = criterion_id(idt_A, real_A) * args.lambda_cycle * args.lambda_id

            # GAN forward
            fake_B = netG_A2B(real_A, z_A)  # A (photo) -> B (renaissance)
            fake_A = netG_B2A(real_B, z_B)

            pred_fake_B = netD_B(fake_B)
            pred_fake_A = netD_A(fake_A)
            loss_GAN_A2B = criterion_GAN(pred_fake_B, valid)
            loss_GAN_B2A = criterion_GAN(pred_fake_A, valid)

            # cycle reconstruction
            rec_A = netG_B2A(fake_B, reparameterize(*enc_B(fake_B.detach())))  # encode fake_B with enc_B (detach to prevent trivial)
            rec_B = netG_A2B(fake_A, reparameterize(*enc_A(fake_A.detach())))
            loss_cycle = criterion_cycle(rec_A, real_A) * args.lambda_cycle + criterion_cycle(rec_B, real_B) * args.lambda_cycle

            # latent cycle consistency: re-encode fake_B and match latent z_A (we use mu of re-encode)
            mu_fakeB, logvar_fakeB = enc_B(fake_B)
            z_fakeB = reparameterize(mu_fakeB, logvar_fakeB)
            loss_latent_A = torch.mean(torch.abs(z_fakeB - z_A)) * args.lambda_latent

            mu_fakeA, logvar_fakeA = enc_A(fake_A)
            z_fakeA = reparameterize(mu_fakeA, logvar_fakeA)
            loss_latent_B = torch.mean(torch.abs(z_fakeA - z_B)) * args.lambda_latent

            # KL losses for both encoders
            loss_kl = kl_loss(mu_A, logvar_A) + kl_loss(mu_B, logvar_B)
            loss_kl = loss_kl * args.lambda_kl

            # perceptual loss (VGG) - enforce structural similarity
            loss_perc = perc_loss(fake_B, real_B) * args.lambda_perc + perc_loss(fake_A, real_A) * args.lambda_perc

            # CLIP intent loss (if available)
            if clip_loss is not None:
                # compute loss of generated images to prompt
                loss_clip_A2B = clip_loss.loss_image_text(fake_B, text_feats) * args.lambda_clip
                loss_clip_B2A = clip_loss.loss_image_text(fake_A, text_feats) * args.lambda_clip
            else:
                loss_clip_A2B = 0
                loss_clip_B2A = 0

            loss_G = (loss_GAN_A2B + loss_GAN_B2A + loss_cycle +
                      loss_idt_A + loss_idt_B + loss_latent_A + loss_latent_B +
                      loss_kl + loss_perc + loss_clip_A2B + loss_clip_B2A)
            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminators
            # -----------------------
            optimizer_D_A.zero_grad()
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, valid)
            fake_A_detached = fake_pool_A.query([fake_A.detach()])[0]
            pred_fake = netD_A(fake_A_detached)
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()
            optimizer_D_A.step()

            optimizer_D_B.zero_grad()
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, valid)
            fake_B_detached = fake_pool_B.query([fake_B.detach()])[0]
            pred_fake = netD_B(fake_B_detached)
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()
            optimizer_D_B.step()

            # logging
            loop.set_postfix(G=loss_G.item(), D_A=loss_D_A.item(), D_B=loss_D_B.item())
            if step % 200 == 0:
                save_sample(real_A, real_B, fake_B, fake_A, args.sample_dir, step)
            # save explainability images occasionally
            if step % 1000 == 0:
                # compute and save saliency and diffs
                save_explain_maps(real_A, fake_B, args.prompt, netG_A2B, clip_loss, encoder=enc_B, out_dir=os.path.join(args.sample_dir, "explain"), device=device, step=step)

            step += 1

        # update learning rates
        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()

        # save checkpoint
        torch.save({
            'enc_A': enc_A.state_dict(),
            'enc_B': enc_B.state_dict(),
            'netG_A2B': netG_A2B.state_dict(),
            'netG_B2A': netG_B2A.state_dict(),
            'netD_A': netD_A.state_dict(),
            'netD_B': netD_B.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_D_A': optimizer_D_A.state_dict(),
            'optimizer_D_B': optimizer_D_B.state_dict(),
            'epoch': epoch
        }, os.path.join(args.checkpoint_dir, f'ckpt_epoch_{epoch:03d}.pth'))

if __name__ == "__main__":
    main()
