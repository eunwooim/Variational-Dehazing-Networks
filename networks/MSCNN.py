import torch 
import torch.nn as nn 
from networks.Unet import UNet

class Coarse_Scale_Net(nn.Module): 
    def __init__(self, in_channels = 3): 
        super(Coarse_Scale_Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels= in_channels, out_channels=5, kernel_size=11, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Upsample(scale_factor = 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=11, padding=5),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Upsample(scale_factor = 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=9, padding=4),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Upsample(scale_factor = 2)
        )
        self.linear_combine = nn.Conv2d(in_channels=10, out_channels=1, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): 
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.sigmoid(self.linear_combine(out))
        return out 

class MSCNN(nn.Module): 
    def __init__(self): 
        super(MSCNN, self).__init__()
        self.coarse_scale_net = Coarse_Scale_Net()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=7, padding=3),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Upsample(scale_factor=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Upsample(scale_factor=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1),
            nn.ReLU(True), 
            nn.MaxPool2d(2),
            nn.Upsample(scale_factor=2)
        )
        self.linear_combine = nn.Conv2d(in_channels=10, out_channels=2, kernel_size=1, stride=1)
        
        
        self.conv4 = nn.Conv2d(in_channels=10, out_channels=1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): 
        out = self.layer1(x)
        y = self.coarse_scale_net(x) 
        out = self.layer2((torch.cat([out, y], dim=1)))
        out = self.layer3(out)
        out = self.sigmoid(self.linear_combine(out))
        return out

class VMSCNN(nn.Module):
    def __init__(self): 
        super(VMSCNN, self).__init__()
        #self.DNet = GridDehazeNet()
        self.DNet = UNet(n_channels=3,n_classes=6)
        self.TNet = MSCNN()

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
