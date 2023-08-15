import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os,sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from metrics import *
from option import opt
import h5py 
import cv2
BS=opt.bs
print(BS)

crop_size='whole_img'
if opt.crop:
    crop_size=opt.crop_size

def get_A(img, p=0.001):
    dc = np.amin(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(31,31))
    dc = cv2.erode(dc, kernel)
    num_pixels = np.prod(dc.shape)
    flat_img, flat_dc = img.reshape(num_pixels,3), dc.ravel()
    idx = (-flat_dc).argsort()[:int(num_pixels * p)]
    A = np.max(flat_img.take(idx, axis=0), axis=0)
    return (0.2126 * A[0] + 0.7152 * A[1] + 0.0722 * A[2]) / 255

def tensorShow(tensors,titles=None):
        '''
        t:BCWH
        '''
        fig=plt.figure()
        for tensor,tit,i in zip(tensors,titles,range(len(tensors))):
            img = make_grid(tensor)
            npimg = img.numpy()
            ax = fig.add_subplot(211+i)
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(tit)
        plt.show()
class RESIDE_Dataset(data.Dataset):
    def __init__(self,path,train,size=crop_size,format='.png'):
        super(RESIDE_Dataset,self).__init__()
        self.size=size
        self.train = train
        self.path = path
        print('crop size',size)
        self.hazy_imgs = os.listdir(os.path.join(path, 'hazy'))

    def __getitem__(self, idx):
        haze=Image.open(os.path.join(self.path, 'hazy', self.hazy_imgs[idx]))
        clear = Image.open(os.path.join(self.path, 'clear', self.hazy_imgs[idx].split('_')[0]+'.jpg'))
        trans = Image.open(os.path.join(self.path, 'trans', self.hazy_imgs[idx]))
        A=get_A(np.array(haze)).reshape(1,1,1)
        clear=tfs.CenterCrop(haze.size[::-1])(clear)
        if not isinstance(self.size,str):
            i,j,h,w=tfs.RandomCrop.get_params(haze,output_size=(self.size,self.size))
            haze=FF.crop(haze,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w)
            trans = FF.crop(trans, i,j,h,w)
        haze,clear,trans=self.augData(haze.convert("RGB") ,clear.convert("RGB"), trans.convert("L"))
        return haze,clear,trans,A
    def augData(self,data,target,trans):
        if self.train:
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
            trans = tfs.RandomHorizontalFlip(rand_hor)(trans)
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)
                trans = FF.rotate(trans,90*rand_rot)
        data=tfs.ToTensor()(data)
        # data_norm=tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(data)
        target=tfs.ToTensor()(target)
        trans=tfs.ToTensor()(trans)
        return  data,target,trans
    def __len__(self):
        return len(self.hazy_imgs)

class RESIDE_Testset(data.Dataset):
    def __init__(self,path,train,size=crop_size,format='.png'):
        super(RESIDE_Testset,self).__init__()
        self.size=size
        self.train = train
        self.path = path
        print('crop size',size)
        self.hazy_imgs = os.listdir(os.path.join(path, 'hazy'))

    def __getitem__(self, idx):
        haze=Image.open(os.path.join(self.path, 'hazy', self.hazy_imgs[idx]))
        clear = Image.open(os.path.join(self.path, 'clear', self.hazy_imgs[idx].split('_')[0] + '.png'))
        haze,clear=self.augData(haze.convert("RGB") ,clear.convert("RGB"))
        return haze,clear

    def augData(self,data,target):
        data=tfs.ToTensor()(data)
        # data=tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(data)
        target=tfs.ToTensor()(target)
        
        return  data ,target
    def __len__(self):
        return len(self.hazy_imgs)

class RESIDE_IN_Dataset(data.Dataset):
    def __init__(self,path,train,size=crop_size,format='.png'):
        super(RESIDE_IN_Dataset,self).__init__()
        self.size=size
        self.train = train
        print('crop size',size)
        with h5py.File(path, 'r') as f: 
            self.clear = np.array(f['clear'])
            self.haze = np.array(f['hazy'])
            self.trans = np.array(f['trans'])


    def __getitem__(self, index):
        A=get_A(self.haze[index]).reshape(1,1,1)
        haze=Image.fromarray(self.haze[index])
        clear = Image.fromarray(self.clear[index//10])
        trans = Image.fromarray(self.trans[index])
        if not isinstance(self.size,str):
            i,j,h,w=tfs.RandomCrop.get_params(haze,output_size=(self.size,self.size))
            haze=FF.crop(haze,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w)
            trans = FF.crop(trans,i,j,h,w)
        haze,clear,trans=self.augData(haze.convert("RGB") ,clear.convert("RGB"), trans.convert("L"))
        return haze,clear,trans,A
    def augData(self,data,target,trans):
        if self.train:
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
            trans = tfs.RandomHorizontalFlip(rand_hor)(trans)
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)
                trans = FF.rotate(trans,90*rand_rot)
        data=tfs.ToTensor()(data)
        #data=tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(data)
        target=tfs.ToTensor()(target)
        trans=tfs.ToTensor()(trans)
        return  data ,target, trans
    def __len__(self):
        return len(self.haze)


class RESIDE_IN_Testset(data.Dataset):
    def __init__(self,path,train,size=crop_size,format='.png'):
        super(RESIDE_IN_Testset,self).__init__()
        self.size=size
        self.train = train
        self.path = path
        print('crop size',size)
        with h5py.File(path, 'r') as f: 
            self.clear = np.array(f['clear'])
            self.haze = np.array(f['hazy'])
    def __getitem__(self, index):
        haze=Image.fromarray(self.haze[index])
        clear = Image.fromarray(self.clear[index//10])

        haze,clear=self.augData(haze.convert("RGB") ,clear.convert("RGB"))
        return haze,clear
    def augData(self,data,target):
        data=tfs.ToTensor()(data)
        #data=tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(data)
        target=tfs.ToTensor()(target)
        return  data ,target
    def __len__(self):
        return len(self.haze)