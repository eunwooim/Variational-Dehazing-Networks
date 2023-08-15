import h5py
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import cv2 
import torchvision.transforms.functional as TF
import os 
from torchvision.transforms import Normalize

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
        self.normalize = Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))

    def __len__(self):
        return (10 * len(self.clear))

    def __getitem__(self, idx):
        # aug-crop
        clear, hazy = self.clear[idx//10], self.hazy[idx]
        if np.random.choice([0,1]):
            clear, hazy= np.flip(clear,1), np.flip(hazy,1)
        # aug=flip
        if np.random.choice([0,1]):
            clear, hazy= np.flip(clear,0), np.flip(hazy,0)
        clear, hazy= to_tensor(clear), to_tensor(hazy)

        # # rot = int(np.random.choice([0,90,180,270]))
        # clear, hazy= TF.rotate(clear, rot), TF.rotate(hazy, rot)
        
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
        if self.args.edge:
            edge = utils.edge_compute(hazy)
            edge = torch.cat([hazy,edge],dim=0)
            return (clear, hazy, trans, A, edge)
        else:
            return (clear, hazy, trans, A)

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

indoor_train = '/home/junsung/nas/reside/in_train_with_A.h5'
indoor_test = '/home/junsung/nas/reside/in_test.h5'
outdoor_train = ''
outdoor_test = ''
rshaze_train = ''
rshaze_test = ''
haze4k_train = ''
haze4k_test = ''

class TrainHaze4k(Dataset): 
    def __init__(self, args): 
        self.args = args
        self.path = args.train_path
        self.patch_size = args.patch_size
        f = h5py.File(args.train_path, 'r')
        self.keys = f.keys()
        self.keys = f.keys()
        self.clear = [f[x]['clear'][y] for x in f.keys() for y in range(len(f[x]['clear']))]
        self.hazy = [f[x]['hazy'][y] for x in f.keys() for y in range(len(f[x]['hazy']))]
        self.trans = [f[x]['trans'][y] for x in f.keys() for y in range(len(f[x]['trans']))]

    
    def __getitem__(self, idx):
        clear = self.clear[idx]
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
        return len(self.hazy)
         
    def random_crop(self, *img):
        H, W = img[0].shape[:2]
        if H < self.patch_size or W < self.patch_size:
            H = max(self.patch_size, H)
            W = max(self.patch_size, W)
            img = cv2.resize(img, (W, H))
        h_ind = random.randint(0, H-self.patch_size)
        w_ind = random.randint(0, W-self.patch_size)
        return [x[h_ind : h_ind + self.patch_size, w_ind : w_ind + self.patch_size] for x in img]

class TestHaze4k(Dataset): 
    def __init__(self, args): 
        self.args = args
        f = h5py.File(args.test_path, 'r')
        self.keys = f.keys()
        self.clear = [f[x]['clear'][y] for x in f.keys() for y in range(len(f[x]['clear']))]
        self.hazy = [f[x]['hazy'][y] for x in f.keys() for y in range(len(f[x]['hazy']))]
        # self.trans = [f[x]['trans'][y] for x in self.keys for y in range(500)]
    
    def __getitem__(self, idx):
        # clear, trans, hazy =to_tensor(np.array(self.clear[idx])), to_tensor(np.array(np.expand_dims(self.trans[idx],-1))), to_tensor(np.array(self.hazy[idx]))
        clear, hazy =to_tensor(np.array(self.clear[idx])), to_tensor(np.array(self.hazy[idx]))
        edge = utils.edge_compute(hazy)
        edge = torch.cat([hazy,edge],dim=0)
        return (clear, edge)#, trans)
    
    def __len__(self):
        return len(self.hazy)

class TestSet(Dataset):
    def __init__(self, args):
        self.args = args
        with h5py.File(args.test_path, 'r') as f:
            self.clear = np.array(f['clear'])
            self.hazy = np.array(f['hazy'])

    def __len__(self):
        return (len(self.clear)*10)

    def __getitem__(self, idx):
        clear, hazy = to_tensor(self.clear[idx//10]), to_tensor(self.hazy[idx])
        hazy = torch.cat([hazy,utils.edge_compute(hazy)], dim=0)
        return (clear, hazy)

class TrainFolder(Dataset): 
    def __init__(self, args): 
        self.args = args
        self.path = args.train_path
        self.patch_size = args.patch_size
        self.hazy_imgs = os.listdir(os.path.join(self.path, 'hazy'))
    
    def __getitem__(self, idx):
        hazy = cv2.imread(os.path.join(self.path, 'hazy', self.hazy_imgs[idx]))[:,:,::-1]
        clear = cv2.imread(os.path.join(self.path, 'clear', self.hazy_imgs[idx].split('_')[0]+'.png'))[:,:,::-1]
        trans = np.expand_dims(cv2.imread(os.path.join(self.path, 'trans', self.hazy_imgs[idx].split('_')[0]+'.png'), cv2.IMREAD_GRAYSCALE), -1)
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
        return len(self.hazy_imgs)
         
    def random_crop(self, *img):
        H, W = img[0].shape[:2]
        if H < self.patch_size or W < self.patch_size:
            H = max(self.patch_size, H)
            W = max(self.patch_size, W)
            return [cv2.resize(x, (self.patch_size, self.patch_size)) for x in img]
        else:
            h_ind = random.randint(0, H-self.patch_size)
            w_ind = random.randint(0, W-self.patch_size)
            return [x[h_ind : h_ind + self.patch_size, w_ind : w_ind + self.patch_size] for x in img]

class TestFolder(Dataset): 
    def __init__(self, args): 
        self.args = args
        self.path = args.test_path
        self.hazy_imgs = os.listdir(os.path.join(self.path, 'hazy'))
    
    def __getitem__(self, idx):
        hazy = cv2.imread(os.path.join(self.path, 'hazy', self.hazy_imgs[idx]))[:,:,::-1]
        clear = cv2.imread(os.path.join(self.path, 'clear', self.hazy_imgs[idx]))[:,:,::-1]
        clear, hazy= to_tensor(clear) , to_tensor(hazy)
        # edge = utils.edge_compute(hazy)
        # edge = torch.cat([hazy,edge],dim=0)
        return (clear, hazy)
    
    def __len__(self):
        return len(self.hazy_imgs)

class BaseFolder(Dataset): 
    def __init__(self, args): 
        self.args = args
        self.path = args.train_path
        self.patch_size = args.patch_size
        self.hazy_imgs = os.listdir(os.path.join(self.path, 'hazy'))
    
    def __getitem__(self, idx):
        hazy = cv2.imread(os.path.join(self.path, 'hazy', self.hazy_imgs[idx]))[:,:,::-1]
        clear = cv2.imread(os.path.join(self.path, 'clear', self.hazy_imgs[idx].split('_')[0]+'.png'))[:,:,::-1]
        
        clear, hazy= self.random_crop(clear, hazy)
        if np.random.choice([0,1]):
            clear, hazy= np.flip(clear,1), np.flip(hazy,1)
        if np.random.choice([0,1]):
            clear, hazy= np.flip(clear,0), np.flip(hazy,0)
        clear, hazy= to_tensor(clear), to_tensor(hazy)
        edge = utils.edge_compute(hazy)
        edge = torch.cat([hazy,edge],dim=0)
        return (clear, edge)
    
    def __len__(self):
        return len(self.hazy_imgs)
         
    def random_crop(self, *img):
        H, W = img[0].shape[:2]
        if H < self.patch_size or W < self.patch_size:
            H = max(self.patch_size, H)
            W = max(self.patch_size, W)
            img = cv2.resize(img, (W, H))
        h_ind = random.randint(0, H-self.patch_size)
        w_ind = random.randint(0, W-self.patch_size)
        return [x[h_ind : h_ind + self.patch_size, w_ind : w_ind + self.patch_size] for x in img]

class TrainMultiDomain(Dataset): 
    def __init__(self, args): 
        self.args = args
        self.path = args.train_path
        self.patch_size = args.patch_size
        f = h5py.File(args.train_path, 'r')
        self.keys = f.keys()
        self.clear = [f[x]['clear'][y] for x in f.keys() for y in range(len(f[x]['clear']))]
        self.hazy = [f[x]['hazy'][y] for x in f.keys() for y in range(len(f[x]['hazy']))]

    
    def __getitem__(self, idx):
        clear = self.clear[idx]
        hazy = self.hazy[idx]
        clear, hazy = self.random_crop(clear, hazy)
        if np.random.choice([0,1]):
            clear, hazy = np.flip(clear,1), np.flip(hazy,1)
        if np.random.choice([0,1]):
            clear, hazy = np.flip(clear,0), np.flip(hazy,0)
        clear, hazy = to_tensor(clear), to_tensor(hazy)
        # edge = utils.edge_compute(hazy)
        # edge = torch.cat([hazy,edge],dim=0)
        return (clear, hazy)
    
    def __len__(self):
        return len(self.hazy)
         
    def random_crop(self, *img):
        H, W = img[0].shape[:2]
        if H < self.patch_size or W < self.patch_size:
            H = max(self.patch_size, H)
            W = max(self.patch_size, W)
            img = cv2.resize(img, (W, H))
        h_ind = random.randint(0, H-self.patch_size)
        w_ind = random.randint(0, W-self.patch_size)
        return [x[h_ind : h_ind + self.patch_size, w_ind : w_ind + self.patch_size] for x in img]

class TestMultiDomain(Dataset): 
    def __init__(self, args): 
        self.args = args
        self.path = args.test_path
        self.patch_size = args.patch_size
        f = h5py.File(args.test_path, 'r')
        self.keys = f.keys()
        self.clear = [f[x]['clear'][y] for x in f.keys() for y in range(len(f[x]['clear']))]
        self.hazy = [f[x]['hazy'][y] for x in f.keys() for y in range(len(f[x]['hazy']))]

    
    def __getitem__(self, idx):
        clear = self.clear[idx]
        hazy = self.hazy[idx]
        clear, hazy = to_tensor(clear), to_tensor(hazy)
        # edge = utils.edge_compute(hazy)
        # edge = torch.cat([hazy,edge],dim=0)
        return (clear, hazy)

    def __len__(self): 
        return len(self.hazy)