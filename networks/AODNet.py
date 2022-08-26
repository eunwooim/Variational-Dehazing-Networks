import torch
import torch.nn as nn


class AODNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, mode='dehazer'):
        super(AODNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(in_channels,3,1,1,0,bias=True)
        self.e_conv2 = nn.Conv2d(3,3,3,1,1,bias=True)
        self.e_conv3 = nn.Conv2d(6,3,5,1,2,bias=True)
        self.e_conv4 = nn.Conv2d(6,3,7,1,3,bias=True)
        self.e_conv5 = nn.Conv2d(12,out_channels,3,1,1,bias=True)

        self.pono = PONO(affine=False)
        
    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        concat1 = torch.cat((x1,x2), 1)
        x1, mean1, std1 = self.pono(x1)
        x2, mean2, std2 = self.pono(x2)
        x3 = self.relu(self.e_conv3(concat1))
        n3 = x3 * std1 + mean1
        concat2 = torch.cat((x2, x3), 1)
        x4 = self.relu(self.e_conv4(concat2))
        n4 = x4 * std2 + mean2
        concat3 = torch.cat((x1,x2,n3,n4),1)
        x5 = self.relu(self.e_conv5(concat3))
        
        if mode == 'dehazer':
            return (x5 * x) - x5 + 1
        else:
            return x5

class PONO(nn.Module):
    def __init__(self, input_size=None, return_stats=False, affine=True, eps=1e-5):
        super(PONO, self).__init__()
        self.return_stats = return_stats
        self.input_size = input_size
        self.eps = eps
        self.affine = affine

        if affine:
            self.beta = nn.Parameter(torch.zeros(1, 1, *input_size))
            self.gamma = nn.Parameter(torch.ones(1, 1, *input_size))
        else:
            self.beta, self.gamma = None, None

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = (x.var(dim=1, keepdim=True) + self.eps).sqrt()
        x = (x - mean) / std
        if self.affine:
            x = x * self.gamma + self.beta
        return x, mean, std
