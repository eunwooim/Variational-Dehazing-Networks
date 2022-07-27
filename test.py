import argparse
import os

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch
from torch.utils.data import DataLoader

from data.dataloader import TestSet
from networks.VHRN import VHRN
from utils import utils


def test(args, testset):
    model = utils.load_model(args, VHRN()).cuda()
    model.eval()
    PSNR, SSIM = [], []
    if args.save_img:
        buffer = []

    for i, data in enumerate(testset):
        gt, corrupt = [x.cuda().float() for x in data]
        with torch.no_grad():
            est = model(corrupt, mode='test')
        est = utils.postprocess(est.squeeze()[:3])
        gt = utils.postprocess(gt.squeeze())
        PSNR.append(psnr(gt, est))
        SSIM.append(ssim(gt, est, channel_axis=2))
        if i+1 in args.save_img:
            est = cv2.cvtColor(est, cv2.COLOR_BGR2RGB)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            buffer.append([gt, est])
            print(f'saved {i+1}')

    if args.save_img:
        for idx, items in zip(args.save_img, buffer):
            cv2.imwrite(f'{args.save_path}/gt_{idx}.jpg', items[0])
            cv2.imwrite(f'{args.save_path}/est_{idx}.jpg', items[1])
    
    return (np.mean(PSNR), np.mean(SSIM))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, default='/home/eunu/nas/reside/in_test.h5')
    parser.add_argument('--save_path', type=str, default='/home/eunu/nas/vhrn_test')

    parser.add_argument('--save_img', type=tuple, default=())
    parser.add_argument('--ckpt', type=str, default='/home/eunu/nas/vhrn_ckpt/1e-7/026.pth')
    parser.add_argument('--cuda', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    testset = TestSet(args)
    testset = DataLoader(testset, batch_size=1, shuffle=False)
    PSNR, SSIM = test(args, testset)
    
    print(f'{args.ckpt} PSNR: {PSNR}, SSIM: {SSIM}')