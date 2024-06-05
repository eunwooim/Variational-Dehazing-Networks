import torch,os,sys,torchvision,argparse
import torchvision.transforms as tfs
from metrics import psnr,ssim
from models.FFA import FFA 
import time,math
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import torch,warnings
from torch import nn

from models.FFA import VHRN
from loss import vlb_loss_value

import torchvision.utils as vutils
warnings.filterwarnings('ignore')
from option import opt
from data_utils import *
from torchvision.models import vgg16

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
log_dir = opt.log_dir

models_={
	# 'ffa':VHRN(),
	'ffa':FFA(3,19)
}
loaders_={
	'its_train': RESIDE_Dataset(path=opt.trainset, train=True, size=240),
	'its_test':RESIDE_Testset(path=opt.testset, train=False),
}
start_time=time.time()
T=opt.steps	
def lr_schedule_cosdecay(t,T,init_lr=opt.lr):
	lr=0.5*(1+math.cos(t*math.pi/T))*init_lr
	return lr

def train(net,loader_train,loader_test,optim,criterion):
	losses=[]
	start_step=0
	max_ssim=0
	max_psnr=0
	ssims=[]
	psnrs=[]
	if opt.resume and os.path.exists(opt.model_dir):
		print(f'resume from {opt.model_dir}')
		ckp=torch.load(opt.model_dir)
		# losses=ckp['losses']
		net.load_state_dict(ckp['model'])
		start_step= 400000
		# max_ssim=ckp['max_ssim']
		# max_psnr=ckp['max_psnr']
		# psnrs=ckp['psnrs']
		# ssims=ckp['ssims']
		print(f'start_step:{start_step} start training ---')
	else :
		print('train from scratch *** ')
	writer = SummaryWriter(log_dir)
	for step in range(start_step+1,opt.steps+1):
		net.train()
		lr=lr_schedule_cosdecay(step,T)
		for param_group in optim.param_groups:
			param_group["lr"] = lr  
		# batch=next(iter(loader_train))
		# hazy, clear, trans, A= [x.cuda().float() for x in batch]
		# dehaze_est, trans_est=net(hazy)
		# optim.zero_grad()
		# loss, lh, kl_dehaze, kl_trans = criterion(
		# 	hazy, dehaze_est, trans_est, clear, trans, A,
		# 	opt.sigma, opt.eps1, opt.eps2, opt.kl_j, opt.kl_t)
		optim.zero_grad()
		batch=next(iter(loader_train))
		x, y, trans, A= [x.cuda().float() for x in batch]
		out=net(x)
		loss = criterion(out, y)
		loss.backward()

		nn.utils.clip_grad_norm_([p for p in net.parameters()], 1e3)

		optim.step()
		# scheduler.step()
		losses.append(loss.item())
		print(f'\rtrain loss : {loss.item():.5f}| step :{step}/{opt.steps}|lr :{optim.param_groups[0]["lr"] :.7f} |time_used :{(time.time()-start_time)/60 :.1f}',end='',flush=True)

		writer.add_scalar('data/loss',loss,step)

		if step % opt.eval_step ==0 :
			net.eval()
			with torch.no_grad():
				ssim_eval,psnr_eval=test(net,loader_test, max_psnr,max_ssim,step)

			print(f'\nstep :{step} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}')

			writer.add_scalar('data/ssim',ssim_eval,step)
			writer.add_scalar('data/psnr',psnr_eval,step)
			writer.add_scalars('group',{
				'ssim':ssim_eval,
				'psnr':psnr_eval,
				'loss':loss
			},step)
			ssims.append(ssim_eval)
			psnrs.append(psnr_eval)

			torch.save({
						'step':step,
						'max_psnr':max_psnr,
						'max_ssim':max_ssim,
						'ssims':ssims,
						'psnrs':psnrs,
						'losses':losses,
						'model':net.state_dict()
			},opt.model_dir.split('.')[0] + str(step)[:3] +'.pk')

	# np.save(f'./numpy_files/{model_name}_{opt.steps}_losses.npy',losses)
	# np.save(f'./numpy_files/{model_name}_{opt.steps}_ssims.npy',ssims)
	# np.save(f'./numpy_files/{model_name}_{opt.steps}_psnrs.npy',psnrs)

def test(net,loader_test,max_psnr,max_ssim,step):
	net.eval()
	torch.cuda.empty_cache()
	ssims=[]
	psnrs=[]
	#s=True
	for i ,(inputs,targets) in enumerate(loader_test):
		inputs=inputs.cuda();targets=targets.cuda()
		pred=net(inputs)
		# # print(pred)
		# tfs.ToPILImage()(torch.squeeze(targets.cpu())).save('111.png')
		# vutils.save_image(targets.cpu(),'target.png')
		# vutils.save_image(pred.cpu(),'pred.png')
		ssim1=ssim(pred,targets).item()
		psnr1=psnr(pred,targets)
		ssims.append(ssim1)
		psnrs.append(psnr1)
		#if (psnr1>max_psnr or ssim1 > max_ssim) and s :
		#		ts=vutils.make_grid([torch.squeeze(inputs.cpu()),torch.squeeze(targets.cpu()),torch.squeeze(pred.clamp(0,1).cpu())])
		#		vutils.save_image(ts,f'samples/{model_name}/{step}_{psnr1:.4}_{ssim1:.4}.png')
		#		s=False
	return np.mean(ssims) ,np.mean(psnrs)


if __name__ == "__main__":
	loader_train=loaders_['its_train']
	loader_train = DataLoader(loader_train, batch_size=BS, shuffle=True)
	loader_test=loaders_['its_test']
	loader_test = DataLoader(loader_test, batch_size=1, shuffle=False)
	net=models_[opt.net]
	net=net.cuda()
	net=torch.nn.DataParallel(net)
	criterion=nn.L1Loss()
	# criterion=vlb_loss_value
	if opt.perloss:
			vgg_model = vgg16(pretrained=True).features[:16]
			vgg_model = vgg_model.to(opt.device)
			for param in vgg_model.parameters():
				param.requires_grad = False
			criterion.append(PerLoss(vgg_model).to(opt.device))
	optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()),lr=opt.lr, betas = (0.9, 0.999), eps=1e-08)
	# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T)
	optimizer.zero_grad()
	train(net,loader_train,loader_test,optimizer,criterion)
	

