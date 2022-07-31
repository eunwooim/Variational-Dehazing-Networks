import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataloader import Train_DnCNN
from loss import loss_fn
from networks.Unet import UNet
from utils import utils


def train(args):
    os.makedirs(args.ckpt, exist_ok=True)
    trainset = Train_DnCNN(args)
    trainset = DataLoader(trainset, shuffle=True, batch_size=args.batch_size,
                            num_workers=8, pin_memory=True)
    writer = SummaryWriter(args.log_dir)
    model = UNet(n_channels=3, n_classes=3)
    model = nn.DataParallel(model)
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr = args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                milestones=args.milestones, gamma=args.gamma)
    if args.resume:
        ckpt = f'{args.ckpt}/{str(args.resume).zfill(3)}.pth'
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
        start_epoch = args.resume
    else:
        model = utils.he_init(model)
        start_epoch = args.resume
    model.train()
    for epoch in range(start_epoch, args.epoch):
        running_loss = 0
        with tqdm(total=len(trainset), desc=f'Epoch {epoch+1}', ncols=70) as pbar:
            for _, batch in enumerate(trainset):
                target, inputx = [x.cuda().float() for x in batch]
                optimizer.zero_grad()
                out = model(inputx)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.update(1)
        scheduler.step()
        writer.add_scalar('Loss', running_loss, epoch)
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': scheduler.state_dict()
                    }, f'{args.ckpt}/{str(epoch+1).zfill(3)}.pth')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--h5_path', type=str, default='')
    parser.add_argument('--train_path', type=str, default='')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--log_dir', type=str, default='')
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--milestones', type=list, default=[20,40,70,100,150])
    parser.add_argument('--gamma', type=float, default=0.5)

    parser.add_argument('--cuda', type=int, default=2)
    parser.add_argument('--resume', type=str, default='')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICE'] = str(args.cuda)
    train(args)
