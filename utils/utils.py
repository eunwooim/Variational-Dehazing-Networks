import cv2
import numpy as np
import torch


def postprocess(output):
    output = torch.clamp(output, min=0, max=1)
    return (output * 255).permute(1,2,0).numpy()


def get_A(img, p=0.001):
    dc = np.amin(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(31,31))
    dc = cv2.erode(dc, kernel)
    num_pixels = np.prod(dc.shape)
    flat_img, flat_dc = img.reshape(num_pixels,3), dc.ravel()
    idx = (-flat_dc).argsort()[:int(num_pixels * p)]
    return np.max(flat_img.take(idx, axis=0), axis=0)    