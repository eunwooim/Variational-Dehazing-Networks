import cv2
import numpy as np
import torch


def postprocess(output):
    output = torch.clamp(output, min=0, max=1)
    return (output * 255).permute(1,2,0).numpy()


def get_A(img):
    img = np.amin(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(31,31))
    dc = cv2.erode(img, kernel)
    ##################