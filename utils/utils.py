import cv2
import numpy as np
import torch
import torch.nn as nn

from networks.VHRN import VHRN


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

def he_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if not m.bias is None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return model
