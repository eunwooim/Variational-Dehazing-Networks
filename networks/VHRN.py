import torch
import torch.nn as nn

from networks.FFANet import FFA
from networks.Unet import UNet


class VHRN(nn.Module):
    def __init__(self): 
        super(VHRN, self).__init__()
        #self.DNet = GridDehazeNet()
        self.DNet = UNet(n_channels=3,n_classes=6)
        self.TNet = UNet(n_channels=3,n_classes=2)

    def forward(self, x, mode = 'train'): 
        if mode.lower() == 'train':
            phi_Z = self.DNet(x)    
            phi_T = self.TNet(x)
            return phi_Z, phi_T
        if mode.lower() == 'test': 
            phi_Z = self.DNet(x)
            return phi_Z
        if mode.lower() == 'transmission':
            phi_T= self.TNet(x)
            return phi_T

class VGU(nn.Module): 
    def __init__(self): 
        super(VGU, self).__init__()
        #self.DNet = GridDehazeNet()
        self.DNet = GridDehazeNet()
        self.TNet = UNet()

    def forward(self, x, mode = 'train'):
        if mode.lower() == 'train':
            phi_Z = self.DNet(x)    
            phi_T = self.TNet(x)
            return phi_Z, phi_T
        if mode.lower() == 'test':
            phi_Z = self.DNet(x)
            return phi_Z
        if mode.lower() == 'transmission':
            phi_T= self.TNet(x)
            return phi_T

class VFFA(nn.Module):
    def __init__(self):
        super(VFFA, self).__init__()
        self.DNet = FFA(in_c=3, out_c=6)
        self.TNet = UNet(n_channels=3, n_classes=2)
    
    def forward(self, x, mode = 'train'):
        if mode.lower() == 'train':
            phi_Z = self.DNet(x)    
            phi_T = self.TNet(x)
            return phi_Z, phi_T
        if mode.lower() == 'test':
            phi_Z = self.DNet(x)
            return phi_Z
        if mode.lower() == 'transmission':
            phi_T= self.TNet(x)
            return phi_T