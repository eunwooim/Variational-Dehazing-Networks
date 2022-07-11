from tkinter import Grid
import torch
import torch.nn as nn
from Unet import UNet
from Griddehaze import GridDehazeNet

class VHRN(nn.Module):
    def __init__(self): 
        super(VHRN, self).__init__()
        self.DNet = GridDehazeNet()
        self.TNet = UNet()

    def forward(self, x, mode = 'train'): 
        if mode.lower() == 'train':
            phi_Z = self.DNet(x)    
            phi_sigma = self.TNet(x)
            return phi_Z, phi_sigma
        if mode.lower() == 'test': 
            phi_Z = self.DNet(x)
            return phi_Z
        if mode.lower() == 'transmission':
            transmission= self.TNet(x)
            return transmission 
            