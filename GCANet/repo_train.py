import argparse
import os

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataloader import BaseTrainSet, TestSet
from GCANet import GCANet
import utils


def train(args):
    torch.cuda.empty_cache()
    os.makedirs(args.ckpt, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)
    model = GCANet(in_c=4)
    model = nn.DataParallel(model).cuda()
    print('Loaded Model')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                step_size=args.step_size, gamma=args.gamma)
    if args.resume:
        ckpt = f'{args.ckpt}/{str(args.resume).zfill(3)}.pth'
        ckpt = torch.load(ckpt)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
        start_epoch = args.resume
    else:
        start_epoch = 0
    valset = TestSet(args)
    valset = DataLoader(valset, shuffle=False, batch_size=1,
                            num_workers=1, pin_memory=True)
    trainset = BaseTrainSet(args)
    trainset = DataLoader(trainset, shuffle=True, batch_size=args.batch_size,
                            num_workers=8, pin_memory=True)
    print('Loaded Data')

    model.train()
    train_len, val_len = len(trainset), len(valset)
    for epoch in range(start_epoch, args.epoch):
        running_loss = PSNR = val_loss = 0
        with tqdm(total=train_len, desc=f'Epoch {epoch+1}', ncols=70) as pbar:
            for i, batch in enumerate(trainset):
                clear, hazy = [x.cuda().float() for x in batch]
                optimizer.zero_grad()
                dehaze_est = model(hazy)
                loss = criterion(dehaze_est, clear)
                loss.backward()

                # norms = [p.grad.detach().abs().mean() for p in param]
                # print(f'D avg: {sum(norms)/len(norms)}')

                optimizer.step()
                running_loss += loss.item()/train_len
                pbar.update(1)
        model.eval()
        for batch in valset:
            clear, hazy = [x.cuda().float() for x in batch]
            with torch.no_grad():
                dehaze_est = model(hazy)
                val_loss += criterion(dehaze_est, clear) / val_len
            dehaze_est = utils.postprocess(dehaze_est)
            clear = utils.postprocess(clear)
            PSNR += psnr(dehaze_est, clear) / val_len
        scheduler.step()
        writer.add_scalar('Train Loss', running_loss, epoch+1)
        writer.add_scalar('PSNR', PSNR, epoch+1)
        writer.add_scalar('Validation Loss', val_loss, epoch+1)
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
    parser.add_argument('--ckpt', type=str, default='/home/eunu/nas/vhrn_ckpt/gca/noedge/edge')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step_size', type=list, default=40)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--log_dir', type=str, default='/home/eunu/nas/vhrn_log/gca/noedge/edge')

    parser.add_argument('--cuda', type=int, default=6)
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