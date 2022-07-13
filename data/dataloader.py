import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import utils


def to_tensor(img):
    return torch.as_tensor(img/255).permute(2,0,1).contiguous()

class TrainSet(Dataset):
    def __init__(self, args):
        self.args = args
        with h5py.File(args.train_path, 'r') as f:
            self.clear = np.array(f['clear'])
            self.hazy = np.array(f['hazy'])
            self.trans = np.array(f['trans'])

    def __len__(self):
        return (10 * len(self.clear))

    def __getitem__(self, idx):
        clear, hazy, trans = self.clear[idx//10], self.hazy[idx], self.trans[idx]
        A = utils.get_A(hazy)
        if self.args.augmentation and np.random.choice([0,1]):
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

class Train_DnCNN(Dataset): 
    def __init__(self, args):
        self.args = args
        with h5py.File(args.train_path, 'r') as f:
            self.clear = np.array(f['clear'])
            self.hazy = np.array(f['hazy'])

    def __len__(self):
        return (10 * len(self.clear))

    def __getitem__(self, idx):
        clear, hazy = self.clear[idx//10], self.hazy[idx]
        if self.args.augmentation and np.random.choice([0,1]):
            clear, hazy= np.flip(clear,1), np.flip(hazy,1)
        clear, hazy= to_tensor(clear), to_tensor(hazy)
        return (clear, hazy) 
