import torch
import torch.nn as nn


def conv3x3(in_c, out_c):
    return nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)

def conv1x1(in_c, out_c):
    return nn.Conv2d(in_c, out_c, kernel_size=1)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        out = self.block(x)
        return self.relu(out + x)

class ResNet(nn.Module):
    def __init__(self, in_c=3, out_c=6, dim=32):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3,dim, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(dim, 2*dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(2*dim, 4*dim, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(4*dim, 2*dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(2*dim, dim, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(dim, out_c, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.block1 = ResBlock(dim)
        self.block2 = ResBlock(dim)
        self.block3 = ResBlock(2*dim)
        self.block4 = ResBlock(2*dim)
        self.block5 = ResBlock(4*dim)
        self.block6 = ResBlock(4*dim)
        self.block7 = ResBlock(2*dim)
        self.block8 = ResBlock(dim)
        self.block9 = ResBlock(out_c)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.block2(self.block1(out))
        out = self.relu(self.conv2(out))
        out = self.block4(self.block3(out))
        out = self.relu(self.conv3(out))
        out = self.block6(self.block5(out))
        out = self.relu(self.conv4(out))
        out = self.relu(self.conv5(self.block7(out)))
        out = self.relu(self.conv6(self.block8(out)))
        out = self.block9(out)
        return out

if __name__ == '__main__':
    model = ResNet(in_c=3, out_c=6)
    inp = torch.rand(1,3,460,620)
    print(model(inp).shape)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
