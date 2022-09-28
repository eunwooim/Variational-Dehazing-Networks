import argparse
import json
import os

from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import dataloader
from utils import utils


def train(args, opts):
    torch.cuda.empty_cache()
    os.makedirs(args.ckpt, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)
    writer = SummaryWriter(args.logdir)
    model = utils.get_model(args)
    model = nn.DataParallel(model).cuda()
    print('Loaded Model')
    criterion = utils.vlb_loss
    if opts['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=opts['lr'])
    elif opts['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=opts['lr'])
    
    if opts['scheduler'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                            step_size=opts['step_size'], gamma=opts['gamma'])
    elif opts['scheduler'] == 'CA':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                T_max=opts['epoch'], eta_min=opts['eta_min'])
    
    if args.resume:
        ckpt = f'{args.ckpt}/{str(args.resume).zfill(3)}.pth'
        ckpt = torch.load(ckpt)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
        start_epoch = args.resume
    else:
        start_epoch = 0
    
    if args.dataset == 'indoor':
        trainset = dataloader.IndoorTrain(args)
        validation = dataloader.IndoorTest(args)
    elif args.dataset == 'outdoor':
        trainset = dataloader.OutdoorTrain(args)
        validation = dataloader.OutdoorTest(args)
    
    trainset = DataLoader(trainset, shuffle=True,
                batch_size=opts['batch_size'], num_workers=8, pin_memory=True)
    validation = DataLoader(validation, shuffle=False, batch_size=1,
                                            num_workers=1, pin_memory=True)

    print('Loaded Data')

    train_len, val_len = len(trainset), len(validation)
    train_len = len(trainset)
    print(f'Train length: {train_len}, Validation length: {val_len}')
    for epoch in range(start_epoch, opts['epoch']):
        model.train()
        tot_loss = lh_loss = trans_loss = dehaze_loss = val_loss = PSNR = 0
        with tqdm(total=train_len, desc=f'Epoch {epoch+1}', ncols=60) as pbar:
            for batch in trainset:
                clear, hazy, trans, A, edge = [x.cuda().float() for x in batch]
                optimizer.zero_grad()
                dehaze_est, trans_est = model(edge)
                loss, lh, kl_dehaze, kl_trans = criterion(
                    hazy, dehaze_est, trans_est, clear, trans, A,
                    args.sigma, args.eps1, args.eps2, args.kl_j, args.kl_t)
                loss.backward()
                nn.utils.clip_grad_norm_([p for p in model.parameters()], 1e3)

                optimizer.step()
                tot_loss += loss.item() / train_len
                lh_loss += lh.item() / train_len
                trans_loss += kl_trans.item() / train_len
                dehaze_loss += kl_dehaze.item() / train_len
                pbar.update(1)
        model.eval()
        for batch in validation:
            clear, hazy = [x.cuda().float() for x in batch]
            with torch.no_grad():
                dehaze_est, _ = model(hazy)
                val_dehaze = utils.loss_val(dehaze_est, clear,
                                            args.eps1, args.kl_j)
                val_loss += val_dehaze / train_len
            dehaze_est = utils.postprocess(dehaze_est[:,:3])
            clear = utils.postprocess(clear)
            PSNR += psnr(dehaze_est, clear)
        print(PSNR/val_len)
        scheduler.step()

        writer.add_scalar('Train Loss', tot_loss, epoch+1)
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
    parser.add_argument('--model', type=str, default='gca')
    parser.add_argument('--dataset', type=str, default='indoor')
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--augmentation', type=bool, default=True)
    parser.add_argument('--sigma', type=float, default=1e-5)
    parser.add_argument('--eps1', type=float, default=1e-5)
    parser.add_argument('--eps2', type=float, default=1e-5)
    parser.add_argument('--kl_j', type=str, default='laplace')
    parser.add_argument('--kl_t', type=str, default='lognormal')

    parser.add_argument('--cuda', type=str, default='7')
    parser.add_argument('--resume', type=int, default=0)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    print(args)
    opt = os.path.join('./config', args.dataset, args.model + '.json')
    with open(opt, 'r') as f:
        opts = json.load(f)
        args.patch_size = opts['patch_size']
    
    print(opts)
    train(args, opts)