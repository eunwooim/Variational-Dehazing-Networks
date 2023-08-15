import argparse
import os

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
# from pytorch_msssim import ssim as former_ssim
import torch
from torch.utils.data import DataLoader
import torch.nn as nn 
import torch.nn.functional as F
from networks.FFANet import FFANet, VHRN
from networks.GCANet import GCANet
from networks.DehazeFormer import VDG, dehazeformer_b

# from metrics import ssim as ssim_ffa
# from metrics import psnr as psnr_ffa
# from utils import utils
from torchvision.transforms import ToTensor
from PIL import Image
import utils

def to_tensor(img):
    return torch.as_tensor(img/255, dtype=torch.float32).permute(2,0,1).contiguous()
def postprocess(output):
    output = torch.clamp(output, min=0, max=1)
    if len(output.shape) == 3:
        return (output * 255).permute(1,2,0).cpu().numpy().astype(np.uint8)
    elif len(output.shape) == 4:
        return (output * 255).permute(0,2,3,1).cpu().numpy().astype(np.uint8)

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

ckpt_dict = {
    "ffa_repo_h4k" : '/home/junsung/my_nas/model/ffa/h4k_repo/ffa200.pk', 
    "ffa_vlb_h4k" : '/home/junsung/my_nas/model/ffa/h4k_vlb/ffa200.pk',
    "ffa_vlb_in" : '/home/junsung/my_nas/model/ffa/in_vlb/ffa480.pk',
    'ffa_vlb_out' : '/data/dehaze/ffa/ckpt/out_vlb/ffa200.pk',
    'ffa_repo_out' : '/data/dehaze/ffa/ckpt/out_repo/ffa200.pk',
    'ffa_repo_in' : '/home/junsung/my_nas/model/ffa/in_repo/ffa480.pk',

    'gca_vlb_in' : '/home/junsung/nas/vhrn_ckpt/eps/565/091.pth',
    'gca_repo_in' : '/home/junsung/nas/vhrn_ckpt/gca/repo/089.pth',
    'gca_vlb_out' : '/home/junsung/nas/vhrn_ckpt/out/gca_555/010.pth',
    'gca_repo_out' : '/home/junsung/nas/vhrn_ckpt/out/gca_repo/006.pth',
    'gca_vlb_h4k' : '/home/junsung/nas/vhrn_ckpt/gca_h4k/ll565/099.pth',
    'gca_repo_h4k' : '/home/junsung/nas/vhrn_ckpt/gca_h4k/repo2/best.pth',

    'former_vlb_in' : '/home/junsung/nas/vhrn_ckpt/vdg/b565.5/clip/276.pth',
    'former_repo_in' : '/home/junsung/nas/vhrn_ckpt/former-b/repo/indoor/dehazeformer-b.pth',
    'former_out_official' : '/home/junsung/nas/vhrn_ckpt/vdg/dehazeformer-out-b.pth',
    'former_vlb_out' : '/home/junsung/nas/vhrn_ckpt/vdg/outdoor/former_out.pth',
    'former_repo_out':'/home/junsung/nas/vhrn_ckpt/vdg/outdoor-repo/27.pth',
    "former_vlb_h4k":'/home/junsung/nas/vhrn_ckpt/vdg/h4k_vlb/former_out.pth',
    'former_repo_h4k':'/home/junsung/nas/vhrn_ckpt/haze4k/dehazeformer/repo/271.pth'
}
dense_per = '/home/junsung/nas/vhrn_ckpt/extra_loss/per/098.pth'
dense_cont = '/home/junsung/nas/vhrn_ckpt/extra_loss/cont2/099.pth'

data_path = "/home/junsung/nas/real_data"
data_list = os.listdir('/home/junsung/nas/real_data')
# data_list.remove('ny17_input.png')
# data_list.remove('forest_input.png')


#####################3

model_name = 'ffa'
mode = 'repo'

######################
if model_name == 'ffa': 
    if mode == 'repo':
        model = FFANet(3,19)
    else: 
        from networks.FFANet import VHRN
        model = VHRN()

elif model_name == 'former': 
    if mode == 'repo':
        model = dehazeformer_b()
    else: 
        model = VDG()
elif model_name == 'gca': 
    if mode == 'repo': 
        model = GCANet(4,3)
    else: 
        from networks.GCANet import VHRN
        model =VHRN()
    

model = nn.DataParallel(model)
ckpt = torch.load(ckpt_dict[model_name + '_' + mode + '_h4k'])

if model_name == 'ffa':
    model.load_state_dict(ckpt['model'])
else:
    model.load_state_dict(ckpt['model_state_dict'])

model.cuda()

for name in data_list: 
    img = cv2.imread(os.path.join(data_path, name))[:,:,::-1]
    img = to_tensor(img)

    if model_name == 'gca': 
        edge = utils.edge_compute(img)
        img = torch.cat([img,edge],dim=0)

    img = img.unsqueeze(0)
    with torch.no_grad():
        if mode == 'repo':
            dehaze = model(img)
        else: 
            dehaze, _ = model(img)
    dehaze = postprocess(dehaze.detach())
    cv2.imwrite(os.path.join('/home/junsung/nas/vhrn_real_test/'+model_name+'_'+mode, name), dehaze.squeeze()[:,:,::-1])