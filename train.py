# train.py
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import UnalignedImageDataset
from models import ResnetGenerator, NLayerDiscriminator
from utils import weights_init, ImagePool, save_sample
import argparse
from tqdm import tqdm
import itertools
import os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataroot', type=str, default='./datasets/renaissance')
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--n_epochs', type=int, default=100)
    p.add_argument('--n_epochs_decay', type=int, default=100)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--lambda_cycle', type=float, default=10.0)
    p.add_argument('--lambda_id', type=float, default=0.5)
    p.add_argument('--pool_size', type=int, default=50)
    p.add_argument('--checkpoint_dir', type=str, default='./checkpoints/renaissance')
    p.add_argument('--sample_dir', type=str, default='./samples')
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # dataset & loader
    dataset = UnalignedImageDataset(args.dataroot, phase='train')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    # models
    netG_A2B = ResnetGenerator().to(device)
    netG_B2A = ResnetGenerator().to(device)
    netD_A = NLayerDiscriminator().to(device)
    netD_B = NLayerDiscriminator().to(device)

    for net in [netG_A2B, netG_B2A, netD_A, netD_B]:
        weights_init(net)

    # losses
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # optimizers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                   lr=args.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # schedulers (linear decay after n_epochs)
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch - args.n_epochs) / float(args.n_epochs_decay + 1)
        return lr_l
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
    scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=lambda_rule)
    scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=lambda_rule)

    fake_pool_A = ImagePool(args.pool_size)
    fake_pool_B = ImagePool(args.pool_size)

    real_label = 1.0
    fake_label = 0.0

    step = 0
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    # training loop
    for epoch in range(1, args.n_epochs + args.n_epochs_decay + 1):
        loop = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        for batch in loop:
            real_A = batch['A'].to(device)
            real_B = batch['B'].to(device)
            bs = real_A.size(0)
            valid = torch.ones((bs, 1, 30, 30), device=device)  # patch size depends; 30 approx at 256
            fake = torch.zeros_like(valid)

            # ------------------
            #  Train Generators
            # ------------------
            optimizer_G.zero_grad()

            # identity loss
            if args.lambda_id > 0:
                idt_B = netG_A2B(real_B)
                idt_A = netG_B2A(real_A)
                loss_idt_B = criterion_identity(idt_B, real_B) * args.lambda_cycle * args.lambda_id
                loss_idt_A = criterion_identity(idt_A, real_A) * args.lambda_cycle * args.lambda_id
            else:
                loss_idt_A = 0
                loss_idt_B = 0

            # GAN loss A->B
            fake_B = netG_A2B(real_A)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, valid)

            # GAN loss B->A
            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, valid)

            # cycle loss
            rec_A = netG_B2A(fake_B)
            rec_B = netG_A2B(fake_A)
            loss_cycle_A = criterion_cycle(rec_A, real_A) * args.lambda_cycle
            loss_cycle_B = criterion_cycle(rec_B, real_B) * args.lambda_cycle

            # total generator loss
            loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator A
            # -----------------------
            optimizer_D_A.zero_grad()
            # real
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, valid)
            # fake (from pool)
            fake_A_detached = fake_pool_A.query([fake_A.detach()])[0]
            pred_fake = netD_A(fake_A_detached)
            loss_D_fake = criterion_GAN(pred_fake, fake)
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()
            optimizer_D_A.step()

            # -----------------------
            #  Train Discriminator B
            # -----------------------
            optimizer_D_B.zero_grad()
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, valid)
            fake_B_detached = fake_pool_B.query([fake_B.detach()])[0]
            pred_fake = netD_B(fake_B_detached)
            loss_D_fake = criterion_GAN(pred_fake, fake)
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()
            optimizer_D_B.step()

            # logging
            loop.set_postfix(G=loss_G.item(), D_A=loss_D_A.item(), D_B=loss_D_B.item())
            if step % 200 == 0:
                save_sample(real_A, real_B, fake_B, fake_A, args.sample_dir, step)
            step += 1

        # update learning rates
        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()

        # save checkpoints each epoch
        torch.save({
            'netG_A2B': netG_A2B.state_dict(),
            'netG_B2A': netG_B2A.state_dict(),
            'netD_A': netD_A.state_dict(),
            'netD_B': netD_B.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_D_A': optimizer_D_A.state_dict(),
            'optimizer_D_B': optimizer_D_B.state_dict(),
            'epoch': epoch
        }, os.path.join(args.checkpoint_dir, f'ckpt_epoch_{epoch:03d}.pth'))

if __name__ == '__main__':
    main()
