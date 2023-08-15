import argparse
import os

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim 
from skimage.metrics import mean_squared_error as mse 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataloader import *
from networks.GCANet import GCANet
from networks.FFANet import FFANet, VGF
from networks.DehazeFormer import VGD
from utils import utils

def train(args):
    torch.cuda.empty_cache()
    os.makedirs(args.ckpt, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)
    model = VGD()
    model = nn.DataParallel(model).cuda()
    print('Loaded Model')
    criterion = utils.vlb_loss
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                milestones=args.milestones, gamma=args.gamma)
    if args.resume:
        ckpt = f'{args.ckpt}/{str(args.resume).zfill(3)}.pth'
        ckpt = torch.load(ckpt)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
        start_epoch = args.resume
    else:
        start_epoch = 0
    trainset = TrainHaze4k(args)
    trainset = DataLoader(trainset, shuffle=True, batch_size=args.batch_size,
                            num_workers=4, pin_memory=True)
    validation = TestHaze4k(args)
    validation = DataLoader(validation, shuffle=False, batch_size=4,
                            num_workers=1, pin_memory=True)
    print('Loaded Data')

    clip_grad_D = args.clip_grad_D
    clip_grad_S = args.clip_grad_S

    param_D = [x for name, x in model.named_parameters() if 'dnet' in name.lower()]
    param_S = [x for name, x in model.named_parameters() if 'tnet' in name.lower()]
    
    train_len, val_len = len(trainset), len(validation)
    for epoch in range(start_epoch, args.epoch):
        model.train()
        grad_norm_D = grad_norm_S = 0
        running_loss = lh_loss = trans_loss = dehaze_loss= 0
        with tqdm(total=len(trainset), desc=f'Epoch {epoch+1}', ncols=60) as pbar:
            for i, batch in enumerate(trainset):
                clear, hazy, trans, A, edge = [x.cuda().float() for x in batch]
                optimizer.zero_grad()
                dehaze_est, trans_est = model(edge, 'train')
                loss, lh, kl_dehaze, kl_trans = criterion(
                    hazy, dehaze_est, trans_est, clear, trans, A,
                    args.sigma, args.eps1, args.eps2, args.kl_j, args.kl_t)
                loss.backward()
                
                total_norm_D = nn.utils.clip_grad_norm_(param_D, clip_grad_D)
                total_norm_S = nn.utils.clip_grad_norm_(param_S, clip_grad_S)
                grad_norm_D = (grad_norm_D*(i/(i+1)) + total_norm_D/(i+1))
                grad_norm_S = (grad_norm_S*(i/(i+1)) + total_norm_S/(i+1))
                optimizer.step()

                running_loss += loss.item() / train_len
                lh_loss += lh.item() / train_len
                trans_loss += kl_trans.item() / train_len
                dehaze_loss += kl_dehaze.item() / train_len
                pbar.update(1)
        model.eval()
        PSNR = []
        for batch in validation:
            clear, hazy = [x.cuda().float() for x in batch]
            with torch.no_grad():
                dehaze_est = model(hazy, 'test')    
            psnr_val = 10 * torch.log10(1 / F.mse_loss(dehaze_est.clamp_(0,1), clear)).item()
            PSNR.append(psnr_val)
        scheduler.step()

        writer.add_scalar('Train Loss', running_loss, epoch+1)
        writer.add_scalar('Train Likelihood', lh, epoch+1)
        writer.add_scalar('Train Transmission', trans_loss, epoch+1)
        writer.add_scalar('Train Dehazer', dehaze_loss, epoch+1)
        # writer.add_scalar('Val T-MSE', np.mean(val_mses), epoch+1)
        # writer.add_scalar('Val T-SSIM', np.mean(val_ssims), epoch+1)
        writer.add_scalar('PSNR', np.mean(np.array(PSNR)), epoch+1)
        clip_grad_D = min(clip_grad_D, grad_norm_D)
        clip_grad_S = min(clip_grad_S, grad_norm_S)
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': scheduler.state_dict()
                    }, f'{args.ckpt}/{str(epoch+1).zfill(3)}.pth')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--edge', type=bool,default=True)
    parser.add_argument('--lr', type=float, default=1e-4)
    # parser.add_argument('--train_path', type=str, default='/home/junsung/my_nas/datasets/nh/nh_train.h5')
    # parser.add_argument('--test_path', type=str, default='/home/junsung/my_nas/datasets/nh/nh_test.h5')
    parser.add_argument('--train_path', type=str, default='/data/dehaze/haze4k/haze4k_train.h5')
    parser.add_argument('--test_path', type=str, default='/data/dehaze/haze4k/haze4k_test.h5')
    parser.add_argument('--ckpt', type=str, default='/home/junsung/nas/vhrn_ckpt/gca_dehaze_vlb/former_t')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--milestones', type=list, default=[40,80])
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--clip_grad_D', type=float, default=1e4,
                                            help="Cliping the gradients for D-Net, (default: 1e4)")
    parser.add_argument('--clip_grad_S', type=float, default=1e3,
                                            help="Cliping the gradients for S-Net, (default: 1e3)")
    parser.add_argument('--eps1', type=float, default=1e-5)
    parser.add_argument('--eps2', type=float, default=1e-5)
    parser.add_argument('--sigma', type=float, default=1e-5)
    parser.add_argument('--kl_j', type=str, default='laplace')
    parser.add_argument('--kl_t', type=str, default='lognormal')
    parser.add_argument('--log_dir', type=str, default='/home/junsung/nas/vhrn_log/gca_dehaze_vlb/former_t')
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--augmentation', type=bool, default=True)


    parser.add_argument('--cuda', type=str, default='1')
    parser.add_argument('--resume', type=int, default=0)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    train(args)
