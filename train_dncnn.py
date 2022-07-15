import os 
import numpy as np

import torch 
import torch.nn as nn
import torch.utils.data as uData
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from networks.DnCNN import DnCNN
from data.dataloader  import Train_DnCNN
from options import set_opts
from loss import loss_fn
import time 

import gc 
gc.collect()
torch.cuda.empty_cache()

args = set_opts()

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
if isinstance(args.gpu_id, int):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in list(args.gpu_id))

def train_model(net, train_dataset, optimizer, lr_scheduler, criterion):
    # set dataloader 
    trian_loader = uData.DataLoader(dataset=train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    num_data = train_dataset.__len__()

    # set tensorboard 
    writer = SummaryWriter(args.log_dir + '/DnCNN')

    for epoch in range(args.epochs): 
        loss_per_epoch = 0
        tic = time.time()

        # train stage 
        net.train()

        for ii, data in enumerate(trian_loader): 
            im_clear, im_hazy = [x.cuda() for x in data]
            optimizer.zero_grad()
            dehaze_est = net(im_hazy)
            loss = criterion(im_clear, dehaze_est)
            loss.backward()

            optimizer.step()

            loss_per_epoch += loss.item()/num_data
            
        # tensorboard
        writer.add_scalar('Loss_epochs', loss_per_epoch, epoch)

        # adjust the learning rate 
        lr_scheduler.step()


        # save model state
        if epoch + 1 % args.save_model_freq == 0 or epoch+1 == args.epochs:  
            model_prefix = 'model_'
            save_model_path = os.path.join(args.model_dir,'DnCNN', model_prefix+str(epoch+1)+'.pth')
            torch.save(net.state_dict(), save_model_path)

        print('Epoch : {} || Total_Loss : {:.6f}'.format(epoch, loss_per_epoch))
        print('This epoch take time : {:.2f}'.format(time.time() - tic))
        print('-' * 100)

def main(): 
    # build the model 
    net = DnCNN(in_channels=3, out_channels=3)
    # move the mode to GPU 
    net = nn.DataParallel(net)
    print('model loaded')

    # optimizer 
    optimizer = optim.Adam(net.parameters(), lr = args.lr)
    # scheduler 
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma= 0.5)

    # datset 
    train_dataset = Train_DnCNN(args)
    print('dataset loaded ')

    # Loss -> MSE 
    criterion = nn.L1Loss()
    
    # Train 
    train_model(net, train_dataset, optimizer, lr_scheduler, criterion)
if __name__ == '__main__': 
    main()