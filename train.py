import os 
import numpy as np

import torch 
import torch.nn as nn
import torch.utils.data as uData
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from networks.VHRN import VHRN
from data.dataloader  import TrainSet
from options import set_opts
from loss import loss_fn
import time 
from math import pi, log 

import gc 
gc.collect()
torch.cuda.empty_cache()

args = set_opts()
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
if isinstance(args.gpu_id, int):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in list(args.gpu_id))

A = 0.5

def train_model(net, train_dataset, optimizer, lr_scheduler, criterion):
    clip_grad_D = args.clip_grad_D
    clip_grad_S = args.clip_grad_S

    # set dataloader 
    trian_loader = uData.DataLoader(dataset=train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    # set tensorboard 
    writer = SummaryWriter(args.log_dir)

    param_D = [x for name, x in net.named_parameters() if 'dnet' in name.lower()]
    param_S = [x for name, x in net.named_parameters() if 'tnet' in name.lower()]
    
    for epoch in range(args.epochs): 
        loss_per_epoch = lh_loss = trans_loss = dehaze_loss =  0
        tic = time.time()

        grad_norm_D = grad_norm_S = 0

        # train stage 
        net.train()

        for ii, data in enumerate(trian_loader): 
            im_clear, im_hazy, im_trans = data 
            im_clear = im_clear.to('cuda:0')
            im_hazy = im_hazy.to('cuda:0')
            im_trans = im_trans.to('cuda:0')
            optimizer.zero_grad()
            phi_Z, phi_T = net(im_hazy, 'train')
            loss , lh, kl_dehaze, kl_trans = criterion(im_hazy, phi_Z, phi_T, im_clear, im_trans, A, sigma = 1e-6, eps1= 1e-6, eps2=1e-6)
            loss.backward()

            # clip the gradnorm
            total_norm_D = nn.utils.clip_grad_norm_(param_D, clip_grad_D)
            total_norm_S = nn.utils.clip_grad_norm_(param_S, clip_grad_S)
            grad_norm_D = (grad_norm_D*(ii/(ii+1)) + total_norm_D/(ii+1))
            grad_norm_S = (grad_norm_S*(ii/(ii+1)) + total_norm_S/(ii+1))

            optimizer.step()

            loss_per_epoch += loss.item()/args.batch_size
            lh_loss += lh.item()/args.batch_size
            trans_loss += kl_trans.item()/args.batch_size
            dehaze_loss += kl_dehaze.item() / args.batch_size
        
        # tensorboard
        writer.add_scalar('Loss_epochs', loss_per_epoch, epoch)

        # adjust the learning rate 
        lr_scheduler.step()


        # save model state
        if epoch % args.save_model_freq == 0 or epoch+1 == args.epochs:  
            model_prefix = 'model_'
            save_model_path = os.path.join(args.model_dir, model_prefix+str(epoch+1)+'.pth')
            torch.save(net.state_dict(), save_model_path)
        
        clip_grad_D = min(clip_grad_D, grad_norm_D)
        clip_grad_S = min(clip_grad_S, grad_norm_S)
        
        print('Epoch : {} || Total_Loss : {:.6f} || LH : {:.6f} || KL_T : {:.6f} || KL_D : {:.6f}'.format(epoch, loss_per_epoch, lh_loss, trans_loss, dehaze_loss ))
        print('This epoch take time : {:.2f}'.format(time.time() - tic))
        print('-' * 100)

def main(): 
    # build the model 
    net = VHRN()
    # move the mode to GPU 
    net = nn.DataParallel(net)
    print('model loaded')

    # optimizer 
    optimizer = optim.Adam(net.parameters(), lr = args.lr)
    # scheduler 
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma= 0.5)

    # datset 
    train_dataset = TrainSet(args)
    print('dataset loaded ')
    train_model(net, train_dataset, optimizer, lr_scheduler, loss_fn)

if __name__ == '__main__': 
    main()