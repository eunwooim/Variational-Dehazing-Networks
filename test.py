import os 
import numpy as np
import cv2 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
import argparse 

from data.dataloader import TestSet
from networks.VHRN import VHRN
from networks.DnCNN import DnCNN

from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio

def set_opts():
    parser = argparse.ArgumentParser()
    
    # dataset setttings 
    parser.add_argument('--train_path', type = str, default= '/home/eunu/nas/reside/in_train.h5')
    parser.add_argument('--test_path', type = str, default='/home/eunu/nas/reside/in_test.h5')
    parser.add_argument('--batch_size', type = int, default=1)

    # gpu 
    parser.add_argument('--gpu_id', type = int, default = 3)

    # model 
    parser.add_argument('--model_dir', type = str, default='/home/junsung/nas/model/vhrn/model_21.pth')
    parser.add_argument('--dncnn_dir', type = str, default='/home/junsung/VHRN/model/DnCNN/model_31.pth')
    parser.add_argument('--mode', type=str, default='')
    parser.add_argument('--save_img', type = bool, default=True)

    args = parser.parse_args()

    return args 

 
def test(): 
    # Dataset setting
    dataset = TestSet(args)
    dataloader = DataLoader(dataset = dataset, batch_size = args.batch_size, shuffle=False)

    # model setting 
    net = VHRN()
    net = nn.DataParallel(net).cuda()
    print(sum(p.numel() for p in net.parameters()))
    input('i')

    checkpoint = torch.load(args.model_dir)

    net.load_state_dict(checkpoint)
    net.eval()

    total_psnr = 0

    for ii, data in enumerate(dataloader): 
        im_clear, im_hazy = [x.cuda().float() for x in data]

        with torch.set_grad_enabled(False): 
            im_dehaze = net(im_hazy, 'test')
            im_dehaze = im_dehaze[:, :3]
        # tensor to numpy
        im_dehaze = im_dehaze.cpu().numpy()
        im_clear = im_clear.cpu().numpy()

        im_dehaze = np.transpose(im_dehaze.squeeze(), (1,2,0))
        im_clear = np.transpose(im_clear.squeeze(), (1,2,0))

        # to ubyte 
        im_dehaze = img_as_ubyte(im_dehaze.clip(0,1))
        im_clear = img_as_ubyte(im_clear.clip(0,1))

        psnr_val = peak_signal_noise_ratio(im_clear, im_dehaze)

        total_psnr += psnr_val

    print('psnr : ', total_psnr / dataset.__len__() )
    
    if args.save_img: 
        _, im_hazy = [x.cuda().float() for x in data]
        
        with torch.no_grad(): 
            _, im_trans = net(im_hazy, 'train')
            im_trans = im_trans[:, :3]

        im_hazy = im_hazy.cpu().numpy()
        im_trans = im_trans.cpu().numpy()

        im_hazy = np.transpose(im_hazy.squeeze(), (1,2,0))
        im_trans = np.transpose(im_trans.squeeze(), (1,2,0))
        
        im_hazy = img_as_ubyte(im_hazy)
        im_trans = img_as_ubyte(im_trans.clip(0,1))
        
        cv2.imwrite('/home/junsung/nas/results/hazy.png', im_hazy)
        cv2.imwrite('/home/junsung/nas/results/gt.png', im_clear)
        cv2.imwrite('/home/junsung/nas/results/trans.png', im_trans)
        cv2.imwrite('/home/junsung/nas/results/dehazy.png', im_dehaze)

def test_dncnn(): 
    # Dataset setting
    dataset = TestSet(args)
    dataloader = DataLoader(dataset = dataset, batch_size = args.batch_size, shuffle=False)

    # model setting 
    net = DnCNN(in_channels=3, out_channels=3)
    net = nn.DataParallel(net).cuda()

    checkpoint = torch.load(args.dncnn_dir)

    net.load_state_dict(checkpoint)
    net.eval()

    total_psnr = 0

    for ii, data in enumerate(dataloader): 
        im_clear, im_hazy = [x.cuda().float() for x in data]

        with torch.set_grad_enabled(False): 
            im_dehaze = net(im_hazy)
            # im_dehaze = im_dehaze[:, :3]
        # tensor to numpy
        im_dehaze = im_dehaze.cpu().numpy()
        im_clear = im_clear.cpu().numpy()

        im_dehaze = np.transpose(im_dehaze.squeeze(), (1,2,0))
        im_clear = np.transpose(im_clear.squeeze(), (1,2,0))

        # to ubyte 
        im_dehaze = img_as_ubyte(im_dehaze.clip(0,1))
        im_clear = img_as_ubyte(im_clear.clip(0,1))

        psnr_val = peak_signal_noise_ratio(im_clear, im_dehaze)

        total_psnr += psnr_val

    print('psnr : ', total_psnr / dataset.__len__() )
    if args.save_img: 
        cv2.imwrite('/home/junsung/nas/results/test_dncnn.png', im_dehaze)
if __name__ == '__main__': 
    args = set_opts()

    if isinstance(args.gpu_id, int): 
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    else: 
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x)for x in list(args.gpu_id))

    test() 
    
    if args.mode == 'dncnn': 
        test_dncnn()    

    