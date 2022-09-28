import h5py
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import cv2 

from utils import utils


def to_tensor(img):
    return torch.as_tensor(img/255, dtype=torch.float32).permute(2,0,1).contiguous()

class IndoorBase(Dataset):
    def __init__(self, args):
        self.args = args
        self.patch_size = args.patch_size
        with h5py.File(indoor_train, 'r') as f:
            self.clear = np.array(f['clear'])
            self.hazy = np.array(f['hazy'])

    def __len__(self):
        return (10 * len(self.clear))

    def __getitem__(self, idx):
        clear, hazy = self.random_crop(self.clear[idx//10], self.hazy[idx])
        if np.random.choice([0,1]):
            clear, hazy= np.flip(clear,1), np.flip(hazy,1)
        if np.random.choice([0,1]):
            clear, hazy= np.flip(clear,0), np.flip(hazy,0)
        clear, hazy = to_tensor(clear), to_tensor(hazy)
        edge = utils.edge_compute(hazy)
        hazy = torch.cat([hazy,edge],dim=0)
        return (clear, hazy)

    def random_crop(self, *im):
        H, W = im[0].shape[:2]
        if H < self.patch_size or W < self.patch_size:
            H = max(self.patch_size, H)
            W = max(self.patch_size, W)
            im = cv2.resize(im, (W, H))
        ind_H = random.randint(0, H-self.patch_size)
        ind_W = random.randint(0, W-self.patch_size)
        return [x[ind_H:ind_H+self.patch_size, ind_W:ind_W+self.patch_size] for x in im]

class IndoorTrain(Dataset):
    def __init__(self, args):
        self.args = args
        self.patch_size = args.patch_size
        with h5py.File(indoor_train, 'r') as f:
            self.clear = np.array(f['clear'])
            self.hazy = np.array(f['hazy'])
            self.trans = np.expand_dims(np.array(f['trans']), -1)

    def __len__(self):
        return (10 * len(self.clear))

    def __getitem__(self, idx):
        clear, hazy, trans = self.clear[idx//10], self.hazy[idx], self.trans[idx]
        A = torch.tensor(utils.get_A(hazy)).reshape(1,1,1).float()
        if self.args.patch_size:
            clear, hazy, trans = self.random_crop(clear, hazy, trans)
        if np.random.choice([0,1]):
            clear, hazy, trans = np.flip(clear,1), np.flip(hazy,1), np.flip(trans,1)
        if np.random.choice([0,1]):
            clear, hazy, trans = np.flip(clear,0), np.flip(hazy,0), np.flip(trans,0)
        clear, hazy, trans = to_tensor(clear), to_tensor(hazy), to_tensor(trans)
        edge = utils.edge_compute(hazy)
        edge = torch.cat([hazy,edge],dim=0)
        return (clear, hazy, trans, A, edge)

    def random_crop(self, *im):
        H, W = im[0].shape[:2]
        if H < self.patch_size or W < self.patch_size:
            H = max(self.patch_size, H)
            W = max(self.patch_size, W)
            im = cv2.resize(im, (W, H))
        ind_H = random.randint(0, H-self.patch_size)
        ind_W = random.randint(0, W-self.patch_size)
        return [x[ind_H:ind_H+self.patch_size, ind_W:ind_W+self.patch_size] for x in im]

class IndoorTest(Dataset):
    def __init__(self, args):
        self.args = args
        with h5py.File(indoor_test, 'r') as f:
            self.clear = np.array(f['clear'])
            self.hazy = np.array(f['hazy'])
    
    def __len__(self):
        return (10 * len(self.clear))

    def __getitem__(self, idx):
        clear, hazy = to_tensor(self.clear[idx//10]), to_tensor(self.hazy[idx])
        hazy = torch.cat([hazy,utils.edge_compute(hazy)], dim=0)
        return (clear, hazy)

class OutdoorTrain(Dataset): 
    def __init__(self, args): 
        self.args = args
        f = h5py.File(outdoor_train, 'r')
        self.keys = f.keys()
        self.clear = [f[x]['clear'] for x in self.keys]
        self.trans = [f[x]['trans'][y] for x in self.keys for y in range(35)]
        self.hazy = [f[x]['hazy'][y] for x in self.keys for y in range(35)]

        self.patch_size = args.patch_size
    
    def __getitem__(self, idx):
        clear = self.clear[idx//35]
        trans = np.expand_dims(self.trans[idx], -1)
        hazy = self.hazy[idx]
        A = torch.tensor(utils.get_A(hazy)).reshape(1,1,1).float()
        clear, hazy, trans = self.random_crop(clear, hazy, trans)
        if np.random.choice([0,1]):
            clear, hazy, trans = np.flip(clear,1), np.flip(hazy,1), np.flip(trans,1)
        if np.random.choice([0,1]):
            clear, hazy, trans = np.flip(clear,0), np.flip(hazy,0), np.flip(trans,0)
        clear, hazy, trans = to_tensor(clear), to_tensor(hazy), to_tensor(trans)
        edge = utils.edge_compute(hazy)
        edge = torch.cat([hazy,edge],dim=0)
        return (clear, hazy, trans, A, edge)
    
    def __len__(self):
        return len(self.keys)*35
         
    def random_crop(self, *img):
        H, W = img[0].shape[:2]
        if H < self.patch_size or W < self.patch_size:
            H = max(self.patch_size, H)
            W = max(self.patch_size, W)
            img = cv2.resize(img, (W, H))
        h_ind = random.randint(0, H-self.patch_size)
        w_ind = random.randint(0, W-self.patch_size)
        return [x[h_ind : h_ind + self.patch_size, w_ind : w_ind + self.patch_size] for x in img]

class OutdoorTest(Dataset):
    def __init__(self, args):
        self.args = args
        f = h5py.File(outdoor_test, 'r')
        self.keys = f.keys()
        self.clear = [f[x]['clear'] for x in self.keys]
        self.hazy = [f[x]['hazy'] for x in self.keys]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        clear, hazy = to_tensor(np.array(self.clear[idx])), to_tensor(np.array(self.hazy[idx]))
        hazy = torch.cat([hazy,utils.edge_compute(hazy)], dim=0)
        return (clear, hazy)