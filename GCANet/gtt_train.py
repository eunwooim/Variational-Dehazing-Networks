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

from dataloader import TrainSet, TestSet
from loss import *
from GCANet import *
import utils


def train(args):
    torch.cuda.empty_cache()
    os.makedirs(args.ckpt, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)
    model = GCANet(in_c=4)
    model = nn.DataParallel(model).cuda()
    print('Loaded Model')
    criterion = vlb_loss_value
    optimizer = optim.AdamW(model.parameters(), lr = args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                milestones=[40,80], gamma=0.1)
    if args.resume:
        ckpt = f'{args.ckpt}/{str(args.resume).zfill(3)}.pth'
        ckpt = torch.load(ckpt)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
        start_epoch = args.resume
    else:
        # model = utils.he_init(model)
        start_epoch = 0
    trainset = TrainSet(args)
    trainset = DataLoader(trainset, shuffle=True, batch_size=args.batch_size,
                            num_workers=8, pin_memory=True)
    validation = TestSet(args)
    validation = DataLoader(validation, shuffle=False, batch_size=1,
                            num_workers=1, pin_memory=True)
    print('Loaded Data')

    train_len, val_len = len(trainset), len(validation)
    print(f'Train length: {train_len}, Validation length: {val_len}')
    for epoch in range(start_epoch, args.epoch):
        model.train()
        running_loss = lh_loss = trans_loss = dehaze_loss = val_loss = PSNR = 0
        with tqdm(total=len(trainset), desc=f'Epoch {epoch+1}', ncols=60) as pbar:
            for i, batch in enumerate(trainset):
                clear, hazy, trans, A, edge = [x.cuda().float() for x in batch]
                optimizer.zero_grad()
                dehaze_est = model(edge)
                loss, lh, kl_dehaze, kl_trans = criterion(
                    hazy, dehaze_est, trans, clear, trans, A,
                    args.sigma, args.eps1, args.eps2, args.kl_j, args.kl_t)
                loss.backward()
                nn.utils.clip_grad_norm_([p for p in model.parameters()], args.grad_clip)

                optimizer.step()
                running_loss += loss.item() / train_len
                lh_loss += lh.item() / train_len
                trans_loss += kl_trans.item() / train_len
                dehaze_loss += kl_dehaze.item() / train_len
                pbar.update(1)
        model.eval()
        for batch in validation:
            clear, hazy = [x.cuda().float() for x in batch]
            # _, _, H, W = clear.shape
            # if H % 16: clear, hazy = clear[:,:,H%16:,:], hazy[:,:,H%16:,:]
            # if W % 16: clear, hazy = clear[:,:,:,W%16:], hazy[:,:,:,W%16:]
            with torch.no_grad():
                dehaze_est = model(hazy)
                val_dehaze = loss_val_value(dehaze_est, clear,
                                            args.eps1, args.kl_j)
                val_loss += val_dehaze / train_len
            dehaze_est = utils.postprocess(dehaze_est[:,:3])
            clear = utils.postprocess(clear)
            PSNR += psnr(dehaze_est, clear)
        print(PSNR/val_len)
        scheduler.step()

        writer.add_scalar('Train Loss', running_loss, epoch+1)
        writer.add_scalar('Train Likelihood', lh_loss, epoch+1)
        writer.add_scalar('Train Transmission', trans_loss, epoch+1)
        writer.add_scalar('Train Dehazer', dehaze_loss, epoch+1)
        writer.add_scalar('Validation Loss', val_loss, epoch+1)
        writer.add_scalar('PSNR', PSNR / val_len, epoch+1)

        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': scheduler.state_dict()
                    }, f'{args.ckpt}/{str(epoch+1).zfill(3)}.pth')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--train_path', type=str, default='/home/eunu/nas/reside/in_train_with_A.h5')
    parser.add_argument('--test_path', type=str, default='/home/eunu/nas/reside/in_test.h5')
    parser.add_argument('--ckpt', type=str, default='/home/eunu/nas/vhrn_ckpt/gca/gt/l1')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--milestones', type=list, default=[40,80])
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--grad_clip', type=float, default=1e3)
    parser.add_argument('--log_dir', type=str, default='/home/eunu/nas/vhrn_log/gca/gt/l1')
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--augmentation', type=bool, default=True)
    parser.add_argument('--eps1', type=float, default=1e-5)
    parser.add_argument('--eps2', type=float, default=1e-5)
    parser.add_argument('--sigma', type=float, default=1e-5)
    parser.add_argument('--kl_j', type=str, default='laplace')
    parser.add_argument('--kl_t', type=str, default='lognormal')

    parser.add_argument('--cuda', type=int, default=7)
    parser.add_argument('--resume', type=int, default=0)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if isinstance(args.cuda, int):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in list(args.cuda))
    train(args)

'''
Loss = kl_d + alpha*lh
tmux1: gaussian, alpha=1, gpu7, tensorboard ./gca/gt/alpha1
tmux0: gaussian, alpha=10, gpu6, tensorboard ./gca/gt/alpha10
'''

'''
gaussian: G, laplace: L, lognormal: N
Target: 30.23dB

ln, gca
sigma5, epsj5, epst5: 28.85
sigma5, epsj6, epst5: 28.60
sigma5, epsj6, epst6: 28.82, 28.55
sigma5, epsj6, epst6: 30.28, 30.16 (value loss, clip)
sigma5, epsj6, epst6: 30.01 (value loss, divide sigma)
sigma5, epsj6, epst6: 29.19 (value loss)
sigma5, epsj7, epst6: 28.72
sigma5, epsj7, epst6: 30.33 (value loss)
sigma5, epsj7, epst5: 28.85

ll, gca
sigma5, epsj6, epst6: 29.72
sigma5, epsj6, epst6: 30.38 (value loss)
sigma5, epsj7, epst6: 29.00
sigma5, epsj7, epst6: 30.32 (value loss)
''' 