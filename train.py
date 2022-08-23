import argparse
import os

from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataloader import TrainSet, TestSet
from loss import *
from networks.VHRN import *
from utils import utils


def train(args):
    torch.cuda.empty_cache()
    os.makedirs(args.ckpt, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)
    model = VHRN()
    model = nn.DataParallel(model).cuda()
    print('Loaded Model')
    criterion = vlb_loss
    optimizer = optim.AdamW(model.parameters(), lr = args.lr)
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
    trainset = TrainSet(args)
    trainset = DataLoader(trainset, shuffle=True, batch_size=args.batch_size,
                            num_workers=8, pin_memory=True)
    validation = TestSet(args)
    validation = DataLoader(validation, shuffle=False, batch_size=1,
                            num_workers=1, pin_memory=True)
    print('Loaded Data')

    train_len, val_len = len(trainset), len(validation)
    for epoch in range(start_epoch, args.epoch):
        model.train()
        running_loss = lh_loss = trans_loss = dehaze_loss = val_loss = PSNR = 0
        with tqdm(total=len(trainset), desc=f'Epoch {epoch+1}', ncols=60) as pbar:
            for i, batch in enumerate(trainset):
                clear, hazy, trans, A = [x.cuda().float() for x in batch]
                optimizer.zero_grad()
                dehaze_est, trans_est = model(hazy, 'train')
                loss, lh, kl_dehaze, kl_trans = criterion(hazy, dehaze_est, trans_est, clear, trans, A, sigma=args.sigma, eps1=args.eps1, eps2=args.eps2, kl_j=args.kl_j, kl_t=args.kl_t)

                nn.utils.clip_grad_norm_([x for name, x in model.named_parameters() if 'dnet' in name.lower()], args.grad_clip)
                nn.utils.clip_grad_norm_([x for name, x in model.named_parameters() if 'tnet' in name.lower()], args.grad_clip)
                loss.backward()

                optimizer.step()
                running_loss += loss.item() / train_len
                lh_loss += lh.item() / train_len
                trans_loss += kl_trans.item() / train_len
                dehaze_loss += kl_dehaze.item() / train_len
                pbar.update(1)
        model.eval()
        for batch in validation:
            clear, hazy = [x.cuda().float() for x in batch]
            with torch.no_grad():
                dehaze_est, _ = model(hazy, 'train')
                val_dehaze = loss_val(dehaze_est, clear, eps1=args.eps1, kl_j=args.kl_j)
                val_loss += val_dehaze / train_len
            dehaze_est = utils.postprocess(dehaze_est[:,:3])
            clear = utils.postprocess(clear)
            PSNR += psnr(dehaze_est, clear)
        scheduler.step()

        writer.add_scalar('Train Loss', running_loss, epoch+1)
        writer.add_scalar('Train Likelihood', lh, epoch+1)
        writer.add_scalar('Train Transmission', trans_loss, epoch+1)
        writer.add_scalar('Train Dehazer', dehaze_loss, epoch+1)
        writer.add_scalar('Validation Loss', dehaze_val, epoch+1)
        writer.add_scalar('PSNR', PSNR/val_len, epoch+1)

        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': scheduler.state_dict()
                    }, f'{args.ckpt}/{str(epoch+1).zfill(3)}.pth')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--train_path', type=str, default='/home/eunu/nas/reside/in_train_with_A.h5')
    parser.add_argument('--test_path', type=str, default='/home/eunu/nas/reside/in_val.h5')
    parser.add_argument('--ckpt', type=str, default='/home/eunu/nas/vhrn_ckpt/dist/gg/1e-7')
    parser.add_argument('--epoch', type=int, default=80)
    parser.add_argument('--milestones', type=list, default=[10,20,30,45,60])
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--grad_clip', type=float, default=0.01)
    parser.add_argument('--log_dir', type=str, default='/home/eunu/nas/vhrn_log/dist/gg/1e-7')
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--augmentation', type=bool, default=True)
    parser.add_argument('--eps', type=float, default=1e-7)
    parser.add_argument('--sigma', type=float, default=1e-6)
    parser.add_argument('--kl_j', type=str, default='gaussian')
    parser.add_argument('--kl_t', type=str, default='gaussian')

    parser.add_argument('--cuda', type=int, default=2)
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
