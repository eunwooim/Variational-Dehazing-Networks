import os
import random
import numpy as np
import cv2
import h5py

from torch.utils.data import Dataset
from utils import hwc_to_chw, read_img


def augment(imgs=[], size=256, edge_decay=0., only_h_flip=False):
	H, W, _ = imgs[0].shape
	Hc, Wc = [size, size]

	# simple re-weight for the edge
	if random.random() < Hc / H * edge_decay:
		Hs = 0 if random.randint(0, 1) == 0 else H - Hc
	else:
		Hs = random.randint(0, H-Hc)

	if random.random() < Wc / W * edge_decay:
		Ws = 0 if random.randint(0, 1) == 0 else W - Wc
	else:
		Ws = random.randint(0, W-Wc)

	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

	# horizontal flip
	if random.randint(0, 1) == 1:
		for i in range(len(imgs)):
			imgs[i] = np.flip(imgs[i], axis=1)

	if not only_h_flip:
		# bad data augmentations for outdoor
		rot_deg = random.randint(0, 3)
		for i in range(len(imgs)):
			imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1))
			
	return imgs


def align(imgs=[], size=256):
	H, W, _ = imgs[0].shape
	Hc, Wc = [size, size]

	Hs = (H - Hc) // 2
	Ws = (W - Wc) // 2
	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

	return imgs


class PairLoader(Dataset):
	def __init__(self, data_dir, sub_dir, mode, size=256, edge_decay=0, only_h_flip=False):
		assert mode in ['train', 'valid', 'test']

		self.mode = mode
		self.size = size
		self.edge_decay = edge_decay
		self.only_h_flip = only_h_flip
		self.data_dir = data_dir + sub_dir

		# self.root_dir = os.path.join(data_dir, sub_dir)
		# self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'GT')))
		# self.img_num = len(self.img_names)
        
		f = h5py.File(self.data_dir, 'r')

		self.clear = f['clear']
		self.hazy = f['hazy']
		if self.mode=='train':
			self.trans = np.expand_dims(f['trans'], -1) 
			self.A = f['A']

	def __len__(self):
		return len(self.hazy)

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		# read image, and scale [0, 1] to [-1, 1]
		
		# img_name = self.img_names[idx]
		# source_img = read_img(os.path.join(self.root_dir, 'hazy', img_name)) * 2 - 1
		# target_img = read_img(os.path.join(self.root_dir, 'GT', img_name)) * 2 - 1
		
		# source_img = self.hazy[idx].astype('float32') / 255.0 * 2 - 1
		# target_img = self.clear[idx//10].astype('float32') / 255.0 * 2 - 1
		# if self.mode=='train': 
		# 	trans_img = self.trans[idx].astype('float32') / 255.0 * 2 - 1
		# 	A = self.A[idx].reshape(1,1,1).astype('float32') * 2 -1 

		source_img = self.hazy[idx].astype('float32') / 255.0 
		target_img = self.clear[idx//10].astype('float32') / 255.0
		if self.mode=='train': 
			trans_img = self.trans[idx].astype('float32') / 255.0 
			A = self.A[idx].reshape(1,1,1).astype('float32')

		if self.mode == 'train':
			# [source_img, target_img] = augment([source_img, target_img], self.size, self.edge_decay, self.only_h_flip)
			# return {'source': hwc_to_chw(source_img), 'target': hwc_to_chw(target_img)}
			[source_img, target_img, trans_img] = augment([source_img, target_img, trans_img], self.size, self.edge_decay, self.only_h_flip)
			return {'source': hwc_to_chw(source_img), 'target': hwc_to_chw(target_img), 'trans' : hwc_to_chw(trans_img), 'A' : A}

		elif self.mode == 'valid':
			[source_img, target_img] = align([source_img, target_img], self.size)
			return {'source': hwc_to_chw(source_img), 'target': hwc_to_chw(target_img)}
		
		else: 
			return {'source': hwc_to_chw(source_img), 'target': hwc_to_chw(target_img)}
		


class SingleLoader(Dataset):
	def __init__(self, root_dir):
		self.root_dir = root_dir
		self.img_names = sorted(os.listdir(self.root_dir))
		self.img_num = len(self.img_names)

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		# read image, and scale [0, 1] to [-1, 1]
		img_name = self.img_names[idx]
		img = read_img(os.path.join(self.root_dir, img_name)) * 2 - 1

		return {'img': hwc_to_chw(img), 'filename': img_name}

class RepoTrain(Dataset): 
	def __init__(self, data_dir, sub_dir, mode, size=256, edge_decay=0, only_h_flip=False):
		assert mode in ['train', 'valid', 'test']

		self.mode = mode
		self.size = size
		self.edge_decay = edge_decay
		self.only_h_flip = only_h_flip

		self.root_dir = os.path.join(data_dir, sub_dir)
		self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'hazy')))
		self.img_num = len(self.img_names)
        

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		img_name = self.img_names[idx]
		source_img = cv2.imread(os.path.join(self.root, 'hazy', img_name))[:,:,::-1].astype('float32') / 255.0
		target_img = cv2.imread(os.path.join(self.root, 'clear', img_name.split('_')[0]+'.jpg'))[:,:,::-1].astype('float32') / 255.0
		
		if self.mode == 'train':
			[source_img, target_img] = augment([source_img, target_img], self.size, self.edge_decay, self.only_h_flip)

		elif self.mode == 'valid':

			[source_img, target_img] = align([source_img, target_img], self.size)
		return {'source': hwc_to_chw(source_img), 'target': hwc_to_chw(target_img)}