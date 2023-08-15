import argparse
import os

from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
from torch import autograd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F 
from pytorch_msssim import ssim
import numpy as np

from dataloader import TrainSet, TestSet, TrainOutFolder, TestOut, TrainRepoFolder
from loss import *
from models.dehazeformer import VDG, dehazeformer_b
from utils import utils


def train(args):
    torch.cuda.empty_cache()
    os.makedirs(args.ckpt, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)
    model = dehazeformer_b()
    model = nn.DataParallel(model).cuda()
    print('Loaded Model')
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr = args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=args.lr * 1e-2)
    if args.resume:
        ckpt = f'{args.ckpt}/{str(args.resume)}.pth'
        ckpt = torch.load(ckpt)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
        start_epoch = args.resume 
    else:
        # model = utils.he_init(model)
        start_epoch = 0
    trainset = TrainRepoFolder(args)
    trainset = DataLoader(trainset, shuffle=True, batch_size=args.batch_size,
                            num_workers=4, pin_memory=True)
    validation = TestOut(args)
    validation = DataLoader(validation, shuffle=False, batch_size=1,
                            num_workers=1, pin_memory=True)
    print('Loaded Data')

    clip_grad_D = args.clip_grad_D
    clip_grad_S = args.clip_grad_S

    train_len, val_len = len(trainset), len(validation)
    print(f'Train length: {train_len}, Validation length: {val_len}')
    # param_D = [x for name, x in model.named_parameters() if 'dnet' in name.lower()]
    # param_S = [x for name, x in model.named_parameters() if 'tnet' in name.lower()]
    best_psnr=0
    for epoch in range(start_epoch, args.epoch):
        model.train()
        running_loss = lh_loss = trans_loss = dehaze_loss = val_loss = PSNR = 0
        grad_norm_D = grad_norm_S = 0
        with tqdm(total=len(trainset), desc=f'Epoch {epoch+1}', ncols=60) as pbar:
            for i, batch in enumerate(trainset):
                clear, hazy= [x.cuda().float() for x in batch]
                optimizer.zero_grad()
                dehaze_est = model(hazy)
                loss = criterion(dehaze_est, clear)
                # loss, lh, kl_dehaze, kl_trans = criterion(
                #     hazy, dehaze_est, trans_est, clear, trans, A,
                    # args.sigma, args.eps1, args.eps2, args.kl_j, args.kl_t)
                loss.backward()
                # total_norm_D = nn.utils.clip_grad_norm_(param_D, clip_grad_D)
                # total_norm_S = nn.utils.clip_grad_norm_(param_S, clip_grad_S)
                # grad_norm_D = (grad_norm_D*(i/(i+1)) + total_norm_D/(i+1))
                # grad_norm_S = (grad_norm_S*(i/(i+1)) + total_norm_S/(i+1))
                optimizer.step()
                running_loss += loss.item() / train_len
                # lh_loss += lh.item() / train_len
                # trans_loss += kl_trans.item() / train_len
                # dehaze_loss += kl_dehaze.item() / train_len
                pbar.update(1)
        model.eval()
        PSNR =[]
        SSIM =[]
        for batch in validation:
            clear, hazy = [x.cuda().float() for x in batch]
            with torch.no_grad():
                est= model(hazy)
                # val_dehaze = loss_val_value(dehaze_est, clear,
                                            # args.eps1, args.kl_j)
                # val_loss += val_dehaze / train_len
            psnr_val = 10 * torch.log10(1 / F.mse_loss(est.clamp_(0,1), clear)).item()
            _, _, H, W = est.size()
            down_ratio = max(1, round(min(H, W) / 256))		# Zhou Wang
            ssim_val = ssim(F.adaptive_avg_pool2d(est, (int(H / down_ratio), int(W / down_ratio))), 
                            F.adaptive_avg_pool2d(clear, (int(H / down_ratio), int(W / down_ratio))), 
                            data_range=1, size_average=False).item()

            SSIM.append(ssim_val)
            PSNR.append(psnr_val)
        print(np.mean(PSNR), np.mean(SSIM))
        scheduler.step()

        writer.add_scalar('Train Loss', running_loss, epoch+1)
        # writer.add_scalar('Train Likelihood', lh_loss, epoch+1)
        # writer.add_scalar('Train Transmission', trans_loss, epoch+1)
        # writer.add_scalar('Train Dehazer', dehaze_loss, epoch+1)
        # writer.add_scalar('Validation Loss', val_loss, epoch+1)
        writer.add_scalar('PSNR', np.mean(PSNR), epoch+1)
        writer.add_scalar("SSIM", np.mean(SSIM), epoch+1)
        # writer.add_scalar('Gradient Norm_D Iter', total_norm_D, epoch+1)
        # writer.add_scalar('Gradient Norm_S Iter', total_norm_S, epoch+1)
    
        # clip_grad_D = min(clip_grad_D, grad_norm_D)
        # clip_grad_S = min(clip_grad_S, grad_norm_S)
    
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': scheduler.state_dict()
                    }, f'{args.ckpt}/{epoch+1}.pth')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='dehazeformer-b', type=str, help='model name')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--train_path', type=str, default='/data/dehaze/reside_out/')
    parser.add_argument('--test_path', type=str, default='/data/dehaze/test_out_former/')
    parser.add_argument('--ckpt', type=str, default='/home/junsung/nas/vhrn_ckpt/vdg/outdoor-repo')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--log_dir', type=str, default='/home/junsung/nas/vhrn_log/vdg/outdoor-repo')
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--augmentation', type=bool, default=True)
    parser.add_argument('--eps1', type=float, default=1e-5)
    parser.add_argument('--eps2', type=float, default=1e-6)
    parser.add_argument('--sigma', type=float, default=1e-5)
    parser.add_argument('--kl_j', type=str, default='laplace')
    parser.add_argument('--kl_t', type=str, default='lognormal')

    parser.add_argument('--clip_grad_D', type=float, default=1e4,
                                            help="Cliping the gradients for D-Net, (default: 1e4)")
    parser.add_argument('--clip_grad_S', type=float, default=1e3,
                                            help="Cliping the gradients for S-Net, (default: 1e3)")

    parser.add_argument('--cuda', type=str, default='2,3')
    parser.add_argument('--resume', type=int, default=15)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    print(args)
    train(args)