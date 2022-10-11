from math import pi, log, sqrt

import cv2
import numpy as np
import torch
import torch.nn as nn


def postprocess(output):
    output = torch.clamp(output, min=0, max=1)
    if len(output.shape) == 3:
        return (output * 255).permute(1,2,0).cpu().numpy().astype(np.uint8)
    elif len(output.shape) == 4:
        return (output * 255).permute(0,2,3,1).cpu().numpy().astype(np.uint8)

def load_model(args, model):
    model = nn.DataParallel(model)
    ckpt = torch.load(args.ckpt)
    model.load_state_dict(ckpt['model_state_dict'])
    return model

def get_A(img, p=0.001):
    dc = np.amin(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(31,31))
    dc = cv2.erode(dc, kernel)
    num_pixels = np.prod(dc.shape)
    flat_img, flat_dc = img.reshape(num_pixels,3), dc.ravel()
    idx = (-flat_dc).argsort()[:int(num_pixels * p)]
    A = np.max(flat_img.take(idx, axis=0), axis=0)
    return (0.2126 * A[0] + 0.7152 * A[1] + 0.0722 * A[2]) / 255

def edge_compute(x):
    x_diffx = torch.abs(x[:,:,1:] - x[:,:,:-1])
    x_diffy = torch.abs(x[:,1:,:] - x[:,:-1,:])
    y = x.new(x.size())
    y.fill_(0)
    y[:,:,1:] += x_diffx
    y[:,:,:-1] += x_diffx
    y[:,1:,:] += x_diffy
    y[:,:-1,:] += x_diffy
    y = torch.sum(y,0,keepdim=True)/3
    y /= 4
    return y

def vlb_loss(inp, d_out, t_out, d_gt, t_gt, A, sigma, eps1, eps2, kl_j, kl_t):
    # Likelihood
    lh = 0.5 * torch.mean((inp - (d_out*t_out) - A*(1-t_out))**2)/sigma
    
    # KL divergence for dehazer
    if kl_j == 'gaussian':
        kl_dehaze = 0.5 * torch.mean((d_out-d_gt)**2)/eps1
    if kl_j == 'laplace':
        kl_dehaze = torch.mean(torch.exp(-torch.abs(d_out - d_gt)/eps1) + torch.abs(d_out - d_gt)/eps1)
    
    # KL divergence for transmission
    if kl_t == 'gaussian':
        kl_transmission = torch.mean((t_out-t_gt)**2)/eps2
    if kl_t == 'lognormal':
        kl_transmission = torch.div(torch.mean((torch.log(t_out) - torch.log(t_gt))**2),2*eps2)

    total_loss = lh + kl_dehaze + kl_transmission
    return total_loss, lh, kl_dehaze, kl_transmission

def loss_val(d_out, d_gt, eps1, kl_j):
    # KL divergence for dehazer
    if kl_j == 'gaussian':
        kl_dehaze = torch.mean((d_out-d_gt)**2 / eps1)
    elif kl_j == 'laplace':
        kl_dehaze = torch.mean(torch.exp(-torch.abs(d_out-d_gt)/eps1) + torch.abs(d_out-d_gt)) / eps1

    return kl_dehaze

def get_model(args):
    if args.model.lower() == 'gca':
        from networks.GCANet import GCANet
        model = GCANet()
    elif args.model.lower() == 'ffa':
        from networks.FFANet import FFA
        model = FFA()
    elif args.model.lower() == 'dehazeformer':
        from networks.DehazeFormer import dehazeformer_b
        model = dehazeformer_b()
    elif args.model.lower() == 'gunet':
        from networks.gUNet import gunet_b
        model = gunet_b()
    return model

def get_vhrn(args):
    if args.model.lower() == 'gca':
        from networks.GCANet import VHRN
        model = VHRN()
    elif args.model.lower() == 'ffa':
        from networks.FFANet import VHRN
        model = VHRN()
    elif args.model.lower() == 'dehazeformer':
        from networks.DehazeFormer import VHRN
        model = VHRN()
    elif args.model.lower() == 'gunet':
        from networks.gUNet import VHRN
        model = VHRN()
    return model
