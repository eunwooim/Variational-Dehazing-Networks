import os 
import numpy as np

import torch 
import torch.nn as nn
import torch.utils.data as uData
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from networks.VHRN import VHRN
from data.dataloader  import TrainSet, TestSet
from options import set_opts
from loss import loss_fn
import time 

args = set_opts()
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
if isinstance(args.gpu_id, int):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in list(args.gpu_id))

A = 0.5

def train_model(net, train_dataset, optimizer, lr_scheduler, criterion):
    # set dataloader 
    trian_loader = uData.DataLoader(dataset=train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    # set tensorboard 
    writer = SummaryWriter(args.log_dir)

    for epoch in range(args.epochs): 
        loss_per_epoch = 0
        tic = time.time()

        # train stage 
        net.train()

        for ii, data in enumerate(trian_loader): 
            im_clear, im_hazy, im_trans = [x.cuda for x in data]
            optimizer.zero_grad()
            phi_Z, phi_T = net(im_hazy, 'train')
            loss = criterion(im_hazy, phi_Z, phi_T, im_clear, im_trans, A, simga = 1e-6, eps1= 1e-6, eps2=1e-6)

            loss.backward()

            optimizer.step()

            loss_per_epoch += (loss/args.batch_size)
        
        # tensorboard
        writer.add_scalar('Loss_epochs', loss_per_epoch, epoch)

        # adjust the learning rate 
        lr_scheduler.step()


        # save model state
        if epoch % args.save_model_preq == 0 or epoch+1 == args.epochs:  
            model_prefix = 'model_'
            save_model_path = os.path.join(args.model_dir, model_prefix, str(epoch+1)+'.pth')
            torch.save(net.state_dict(), save_model_path)

        print('Epoch : {} || Loss : {:.2f}'.format(epoch, loss_per_epoch))
        print(' This epoch take time : {:.2f}'.format(tic - time.time()))
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
    
    train_model(net, train_dataset, optimizer, lr_scheduler, loss_fn)

if __name__ == '__main__': 
    main()