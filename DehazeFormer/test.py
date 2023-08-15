import argparse
import os

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.metrics import structural_similarity as ssim
from pytorch_msssim import ssim 
import torch
from torch.utils.data import DataLoader
import torch.nn as nn 
import torch.nn.functional as F
# from networks.FFANet import FFANet, VHRN
# from networks.GCANet import GCANet
from models.dehazeformer import VDG
from dataloader import TestSet, TestOut

from utils import utils

# class VHRN(nn.Module):
#     def __init__(self): 
#         super(VHRN, self).__init__()
#         self.DNet = GCANet(4,3,mode='d')
#         self.TNet = GCANet(4,1,mode='t')

#     def forward(self, x, mode = 'train'):
#         if mode.lower() == 'train':
#             phi_Z = self.DNet(x)
#             phi_T = self.TNet(x)
#             return phi_Z, phi_T
#         if mode.lower() == 'test':
#             phi_Z = self.DNet(x)
#             return phi_Z
#         if mode.lower() == 'transmission':
#             phi_T= self.TNet(x)
#             return phi_T


def test(args, testset, model):
    #model = utils.load_model(args, VHRN()).cuda()
    # model = utils.load_model(args, VHRN()).cuda()
    # model.eval()
    PSNR, SSIM = [], []
    if args.save_img:
        buffer = []

    for i, data in enumerate(testset):
        gt, corrupt = [x.cuda().float() for x in data]
        # _, _, h, w = corrupt.shape
        # if h % 16: h -= h % 16
        # if w % 16: w -= w % 16
        # gt, inp = gt[:,:,:h,:w], corrupt[:,:,:h,:w]
        with torch.no_grad():
            # est, trans = model(corrupt, mode='train')
            # est = model(corrupt)
            est, trans = model(corrupt)

        if args.mode == 'former': 
            psnr_val = 10 * torch.log10(1 / F.mse_loss(est.clamp_(0,1), gt)).item()
            _, _, H, W = est.size()
            down_ratio = max(1, round(min(H, W) / 256))		# Zhou Wang
            ssim_val = ssim(F.adaptive_avg_pool2d(est, (int(H / down_ratio), int(W / down_ratio))), 
                            F.adaptive_avg_pool2d(gt, (int(H / down_ratio), int(W / down_ratio))), 
                            data_range=1, size_average=False).item()				
            SSIM.append(ssim_val)
            PSNR.append(psnr_val)
        else:           
            est = utils.postprocess(est.squeeze())
            gt = utils.postprocess(gt.squeeze())
            # trans = (torch.log(trans)*255).cpu().numpy().astype(np.uint8)
            PSNR.append(psnr(gt, est))
            SSIM.append(ssim(gt, est,channel_axis=2))
        if i+1 in args.save_img:
            est = cv2.cvtColor(est, cv2.COLOR_BGR2RGB)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            corrupt = cv2.cvtColor(utils.postprocess(corrupt[:,:3].squeeze()), cv2.COLOR_BGR2RGB)
            buffer.append([gt, est, trans, corrupt])
            print(f'saved {i+1}')

    if args.save_img:
        for idx, items in zip(args.save_img, buffer):
            cv2.imwrite(f'{args.save_path}/gt_{idx}.jpg', items[0])
            cv2.imwrite(f'{args.save_path}/est_{idx}.jpg', items[1])
            cv2.imwrite(f'{args.save_path}/trans_{idx}.jpg', items[2])
            cv2.imwrite(f'{args.save_path}/corrupt_{idx}.jpg', items[3])
    
    # for i, val in enumerate(PSNR): 
    #     print(i, val)

    return (np.mean(PSNR), np.mean(SSIM))

def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--test_path', type=str, default='/home/junsung/nas/haze4k/haze4k_test.h5')
    # parser.add_argument('--test_path', type=str, default='/home/junsung/nas/reside/in_test.h5')
    # parser.add_argument('--test_path', type=str, default='/home/junsung/my_nas/datasets/reside_out_test')
    parser.add_argument('--test_path', type=str, default='/data/dehaze/test_out_former/')
    parser.add_argument('--save_path', type=str, default='/home/junsung/my_nas/results/gca/out')

    parser.add_argument('--ckpt', type=str, default='/home/junsung/nas/vhrn_ckpt/vdg/outdoor_onlyhfilp/former_out.pth')
    parser.add_argument('--save_img', type=tuple, default='')
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--mode', type=str, default = 'former')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    os.makedirs(args.save_path, exist_ok = True)
    # testset = TestHaze4k(args)
    testset = TestOut(args)
    # testset = TestFolder(args)
    testset = DataLoader(testset, batch_size=1, shuffle=False)
    # model = utils.load_model(args, FFANet(3,19)).cuda()
    # model = utils.load_model(args, GCANet(4,3,mode='d')).cuda()
    # model = utils.load_model(args, VDG()).cuda()
    # model_name = 'out_vlb'
    # hazy_path = os.path.join(args.save_path, model_name,'hazy')
    # dehaze_path = os.path.join(args.save_path, model_name,'dehaze')
    # trans_path = os.path.join(args.save_path, model_name,'trans')
    # gt_path = os.path.join(args.save_path, model_name, 'gt')
    # os.makedirs(hazy_path, exist_ok=True)
    # os.makedirs(trans_path, exist_ok=True)
    # os.makedirs(dehaze_path, exist_ok=True)
    # os.makedirs(gt_path, exist_ok=True)
    # model.eval()
    # for i,batch in enumerate(testset):
    #     clear, hazy = batch
    #     claer, hazy = clear.cuda().float(), hazy.cuda().float()
    #     _, _, h, w = hazy.shape
    #     if h % 2: h -= h % 2
    #     if w % 2: w -= w % 2
    #     clear, hazy = clear[:,:,:h,:w], hazy[:,:,:h,:w]        
    #     with torch.no_grad():
    #         dehaze, trans= model(hazy)
    #         # dehaze = model(hazy)
        
    #     clear = utils.postprocess(clear.squeeze())
    #     hazy = utils.postprocess(hazy[:,:3].squeeze())
    #     trans = np.clip(((trans)*255).squeeze().detach().cpu().numpy(), 0,255).astype(np.uint8)
    #     dehaze = utils.postprocess(dehaze.squeeze().detach())

    #     cv2.imwrite(os.path.join(args.save_path, model_name,'gt', str(i)+'.jpg'), clear[:,:,::-1])
    #     cv2.imwrite(os.path.join(args.save_path, model_name,'hazy', str(i)+'.jpg'), hazy[:,:,::-1])
    #     cv2.imwrite(os.path.join(args.save_path, model_name,'dehaze', str(i)+'.jpg'), dehaze[:,:,::-1])
    #     cv2.imwrite(os.path.join(args.save_path, model_name,'trans', str(i)+'.jpg'), trans)
    
    # PSNR, SSIM = test(args, testset, model)
    # print(PSNR, SSIM)
    
    
    root_ckpt = args.ckpt
    ckpts = os.listdir(args.ckpt)
    for ckpt in ckpts: 
        if (24 < int(ckpt[:2])):
            args.ckpt = root_ckpt + ckpt
            model = utils.load_model(args, VDG()).cuda()
            PSNR, SSIM = test(args, testset, model)
            print(ckpt, PSNR, SSIM)