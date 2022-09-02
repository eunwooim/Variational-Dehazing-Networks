import h5py
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import cv2 

from utils import utils


def to_tensor(img):
    return torch.as_tensor(img/255, dtype=torch.float32).permute(2,0,1).contiguous()

class TrainSet(Dataset):
    def __init__(self, args):
        self.args = args
        self.pch_size = args.patch_size
        with h5py.File(args.train_path, 'r') as f:
            self.clear = np.array(f['clear'])
            self.hazy = np.array(f['hazy'])
            self.trans = np.expand_dims(np.array(f['trans']), -1)
            self.A = np.array(f['A'])

    def __len__(self):
        return (10 * len(self.clear))

    def __getitem__(self, idx):
        clear, hazy, trans = self.clear[idx//10], self.hazy[idx], self.trans[idx]
        A = torch.tensor(self.A[idx]).reshape(1,1,1).float()
        clear, hazy, trans = utils.random_crop(clear, hazy, trans)
        if np.random.choice([0,1]):
            clear, hazy, trans = np.flip(clear,0), np.flip(hazy,0), np.flip(trans,0)
        if np.random.choice([0,1]):
            clear, hazy, trans = np.flip(clear,1), np.flip(hazy,1), np.flip(trans,1)
        clear, hazy, trans = to_tensor(clear), to_tensor(hazy), to_tensor(trans)
        return (clear, hazy, trans, A)

class TestSet(Dataset):
    def __init__(self, args):
        self.args = args
        with h5py.File(args.test_path, 'r') as f:
            self.clear = np.array(f['clear'])
            self.hazy = np.array(f['hazy'])
    
    def __len__(self):
        return (10 * len(self.clear))

    def __getitem__(self, idx):
        return (to_tensor(self.clear[idx//10]), to_tensor(self.hazy[idx]))

class BaseTrainSet(Dataset): 
    def __init__(self, args):
        self.args = args
        self.pch_size = args.patch_size
        with h5py.File(args.train_path, 'r') as f:
            self.clear = np.array(f['clear'])
            self.hazy = np.array(f['hazy'])

    def __len__(self):
        return (10 * len(self.clear))

    def __getitem__(self, idx):
        clear, hazy = utils.random_crop(self.clear[idx//10], self.hazy[idx])
        if self.args.augmentation and np.random.choice([0,1]):
            clear, hazy= np.flip(clear,1), np.flip(hazy,1)
        clear, hazy= to_tensor(clear), to_tensor(hazy)
        return (clear, hazy)


class TransTrainSet(Dataset): 
    def __init__(self, args):
        self.args = args
        with h5py.File(args.train_path, 'r') as f:
            self.trans = np.array(f['trans'])
            self.hazy = np.array(f['hazy'])
        self.pch_size = args.patch_size

    def __len__(self):
        return len(self.hazy)

    def __getitem__(self, idx):
        trans, hazy = utils.random_crop(self.trans[idx], self.hazy[idx])
        if self.args.augmentation and np.random.choice([0,1]):
            trans, hazy = np.flip(trans,1), np.flip(hazy,1)
        trans = cv2.cvtColor(trans, cv2.COLOR_RGB2GRAY)
        trans, hazy= to_tensor(trans), to_tensor(hazy)
        return (trans, hazy)  
