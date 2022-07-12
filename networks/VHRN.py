import torch
import torch.nn as nn
from networks.Unet import UNet
from networks.Griddehaze import GridDehazeNet

class VHRN(nn.Module):
    def __init__(self): 
        super(VHRN, self).__init__()
        self.DNet = GridDehazeNet()
        self.TNet = UNet()

    def forward(self, x, mode = 'train'): 
        if mode.lower() == 'train':
            phi_Z = self.DNet(x)    
            phi_trans = self.TNet(x)
            return phi_Z, phi_trans
        if mode.lower() == 'test': 
            phi_Z = self.DNet(x)
            return phi_Z
        if mode.lower() == 'transmission':
            phi_trans= self.TNet(x)
            return phi_trans