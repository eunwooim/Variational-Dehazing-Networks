import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils import AverageMeter
from datasets.loader import PairLoader
from models import *
from loss import *
import collections


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='dehazeformer-b', type=str, help='model name')
parser.add_argument('--num_workers', default=8, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='/home/junsung/nas/vhrn_ckpt/former-b/vdg/b566', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='/data/dehaze/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='/home/junsung/nas/vhrn_log/former-b/vdg/b566', type=str, help='path to logs')
parser.add_argument('--dataset', default='RESIDE-IN', type=str, help='dataset name')
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')
parser.add_argument('--gpu', default='0,1', type=str, help='GPUs used for training')
parser.add_argument('--eps1', type=float, default=1e-6)
parser.add_argument('--eps2', type=float, default=1e-5)
parser.add_argument('--sigma', type=float, default=1e-5)
parser.add_argument('--kl_j', type=str, default='laplace')
parser.add_argument('--kl_t', type=str, default='lognormal')
parser.add_argument('--resume', type = int, default=0)
# parser.add_argument('--sigma', default=1e-5, type=float)
# parser.add_argument('--eps', default=1e-6, type=float)
args = parser.parse_args()

os.environ['CUDA_DECICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

print()

def train(train_loader, network, criterion, optimizer, scaler):
	losses = AverageMeter()

	torch.cuda.empty_cache()
	
	network.train()
	lh_loss = kl_d = kl_t = 0
	for batch in train_loader:
		source_img = batch['source'].cuda()
		target_img = batch['target'].cuda()
		trans_img = batch['trans'].cuda()
		A_img = batch['A'].cuda()

		with autocast(args.no_autocast):
			# output = network(source_img)
			# loss = criterion(output, target_img)
			d_out, t_out = network(source_img)
			loss, lh, kl_dehaze, kl_trans = criterion(
				source_img, d_out, t_out, target_img, trans_img, A_img,
				args.sigma, args.eps1, args.eps2, args.kl_j, args.kl_t)

		losses.update(loss.item())

		lh_loss += lh.item()
		kl_d += kl_dehaze.item()
		kl_t += kl_trans.item()

		optimizer.zero_grad()
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()
		
	return (losses.avg, lh_loss, kl_d, kl_t)
	# return losses.avg

def valid(val_loader, network):
	PSNR = AverageMeter()

	torch.cuda.empty_cache()

	network.eval()

	for batch in val_loader:
		source_img = batch['source'].cuda()
		target_img = batch['target'].cuda()

		with torch.no_grad():							# torch.no_grad() may cause warning
			#output = network(source_img).clamp_(0,1)
			output = network(source_img, 'test').clamp_(0, 1)		

		mse_loss = F.mse_loss(output , target_img, reduction='none').mean((1, 2, 3))
		psnr = 10 * torch.log10(1 / mse_loss).mean()
		PSNR.update(psnr.item(), source_img.size(0))

	return PSNR.avg


if __name__ == '__main__':
	setting_filename = os.path.join('configs', args.exp, args.model+'.json')
	if not os.path.exists(setting_filename):
		setting_filename = os.path.join('configs', args.exp, 'default.json')
	with open(setting_filename, 'r') as f:
		setting = json.load(f)

	#network = eval(args.model.replace('-', '_'))()
	
	network = VDG()
	network = nn.DataParallel(network).cuda()	
	
	# state_dict = network.state_dict()
	# network.load_state_dict(torch.load('/home/junsung/nas/vhrn_ckpt/former_test/indoor/dehazeformer-t_gaussian_0.01.pth')['state_dict'])
	# checkpoint = torch.load('/home/junsung/nas/vhrn_ckpt/former-b/pretrain/dehazeformer-b.pth')['state_dict']
	# new_checkpoint = collections.OrderedDict()
	# for name, param in checkpoint.items(): 
	# 	new_name = name[:7]+'DNet.'+name[7:]
	# 	new_checkpoint[new_name]= param
	# state_dict.update(new_checkpoint)
	# network.load_state_dict(state_dict)

	#criterion = nn.L1Loss()
	criterion = vlb_loss_value

	if setting['optimizer'] == 'adam':
		optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])
	elif setting['optimizer'] == 'adamw':
		optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
	else:
		raise Exception("ERROR: unsupported optimizer") 

	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'], eta_min=setting['lr'] * 1e-2)
	scaler = GradScaler()

	save_dir = os.path.join(args.save_dir, args.exp)
	os.makedirs(save_dir, exist_ok=True)
	
	if args.resume: 
		network.load_state_dict(torch.load(os.path.join(save_dir, args.model+'.pth'))['state_dict'])
		optimizer.load_state_dict(torch.load(os.path.join(save_dir, args.model+'.pth'))['optimizer_state_dict'])
		scheduler.load_state_dict(torch.load(os.path.join(save_dir, args.model+'.pth'))['lr_scheduler_state_dict'])
		start_epoch= args.resume
	else: 
		start_epoch=0


	train_dataset = PairLoader(data_dir=args.data_dir, sub_dir='in_train_with_A.h5', mode='train', 
								size=setting['patch_size'], edge_decay=setting['edge_decay'], only_h_flip=setting['only_h_flip'])
	train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)
	val_dataset = PairLoader(data_dir=args.data_dir, sub_dir='in_test.h5',mode=setting['valid_mode'], 
							  size = setting['patch_size'])
	val_loader = DataLoader(val_dataset,
                            batch_size=setting['batch_size'],
                            num_workers=args.num_workers,
                            pin_memory=True)



	#if not os.path.exists(os.path.join(save_dir, args.model+'.pth')):
	print('==> Start training, current model name: ' + args.model)
	# print(network)

	# writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model, args.mode+str(args.weight)))
	writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp))
	best_psnr = 0

	for epoch in tqdm(range(start_epoch,setting['epochs'] + 1)):
		# loss = train(train_loader, network, criterion, optimizer, scaler)
		loss, lh, kl_dehaze, kl_trans = train(train_loader, network, criterion, optimizer, scaler)

		writer.add_scalar('Train loss', loss, epoch+1)
		writer.add_scalar('Train Likelihood', lh/len(train_dataset), epoch+1)
		writer.add_scalar('Train Dehazer', kl_dehaze/len(train_dataset), epoch+1)
		writer.add_scalar('Train Transmission', kl_trans/len(train_dataset), epoch+1)
		scheduler.step()

		if epoch % setting['eval_freq'] == 0:
			avg_psnr = valid(val_loader, network)
			
			writer.add_scalar('valid_psnr', avg_psnr, epoch)

			if avg_psnr > best_psnr:
				best_psnr = avg_psnr
				torch.save({'state_dict': network.state_dict(),
                        	'optimizer_state_dict': optimizer.state_dict(),
                        	'lr_scheduler_state_dict': scheduler.state_dict()
                        	},os.path.join(save_dir, args.model+'.pth'))
			
			writer.add_scalar('best_psnr', best_psnr, epoch)