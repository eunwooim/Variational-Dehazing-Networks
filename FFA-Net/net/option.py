import torch,os,sys,torchvision,argparse
import torchvision.transforms as tfs
import time,math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch,warnings
from torch import nn
import torchvision.utils as vutils
warnings.filterwarnings('ignore')

parser=argparse.ArgumentParser()
parser.add_argument('--steps',type=int,default=100000)
parser.add_argument('--device',type=str,default='Automatic detection')
parser.add_argument('--resume',type=bool,default=False)
parser.add_argument('--eval_step',type=int,default=5000)
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--model_dir',type=str,default='/data/dehaze/ffa/ckpt/out_repo/')
parser.add_argument('--trainset',type=str,default='/data/dehaze/reside_out/')
parser.add_argument('--testset',type=str,default='/data/dehaze/test_out_git/')
parser.add_argument('--net',type=str,default='ffa')
parser.add_argument('--gps',type=int,default=3,help='residual_groups')
parser.add_argument('--blocks',type=int,default=20,help='residual_blocks')
parser.add_argument('--bs',type=int,default=4,help='batch size')
parser.add_argument('--crop',action='store_true')
parser.add_argument('--crop_size',type=int,default=240,help='Takes effect when using --crop ')
parser.add_argument('--no_lr_sche',default=False,help='no lr cos schedule')
parser.add_argument('--perloss',action='store_true',help='perceptual loss')
parser.add_argument('--log_dir', type=str, default='/data/dehaze/ffa/log/out_repo/')
parser.add_argument('--premodel', type=str, default='/home/junsung/FFA-Net/net/ots_train_ffa_3_19.pk')
parser.add_argument('--eps1', type=float, default=1e-6)
parser.add_argument('--eps2', type=float, default=1e-5)
parser.add_argument('--sigma', type=float, default=1e-5)
parser.add_argument('--kl_j', type=str, default='laplace')
parser.add_argument('--kl_t', type=str, default='lognormal')
opt=parser.parse_args()
model_name='ffa'

if not os.path.exists(opt.model_dir):
	os.mkdir(opt.model_dir)

opt.model_dir=opt.model_dir+model_name+'.pk'
log_dir=opt.log_dir

print(opt)
print('model_dir:',opt.model_dir)



if not os.path.exists(log_dir):
	os.mkdir(log_dir)
