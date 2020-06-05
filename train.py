import numpy as np
import time
import torch
import torch.nn as nn
import os
import cv2
# import visdom
import wandb
import random
import argparse
from tqdm import tqdm as tqdm

from cannet import CANNet
from my_dataset import CrowdDataset


def main(args):
    wandb.init(project="crowd", config=args)
    args = wandb.config
    # print(args)

    # vis=visdom.Visdom()
    torch.cuda.manual_seed(args.seed)
    model=CANNet().to(args.device)
    criterion=nn.MSELoss(reduction='sum').to(args.device)
    # optimizer=torch.optim.SGD(model.parameters(), args.lr,
    #                           momentum=args.momentum,
    #                           weight_decay=0)
    optimizer=torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.decay)
    train_dataset = CrowdDataset(args.train_image_root, args.train_dmap_root, gt_downsample=8, phase='train')
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset   = CrowdDataset(args.val_image_root, args.val_dmap_root, gt_downsample=8, phase='test')
    val_loader    = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    
    min_mae = 10000
    min_epoch = 0
    for epoch in tqdm(range(0, args.epochs)):
        # training phase
        model.train()
        model.zero_grad()
        train_loss = 0
        train_mae = 0
        train_bar = tqdm(train_loader)
        for i, (img,gt_dmap) in enumerate(train_bar):
            # print(img.shape, gt_dmap.shape)
            img = img.to(args.device)
            gt_dmap = gt_dmap.to(args.device)
            
            # forward propagation
            et_dmap = model(img)
            # calculate loss
            # print(et_dmap.shape, gt_dmap.shape)
            loss = criterion(et_dmap, gt_dmap)
            train_loss += loss.item()
            train_mae += abs(et_dmap.data.sum()-gt_dmap.data.sum()).item()
            loss = loss/args.gradient_accumulation_steps
            loss.backward()
            if (i+1)%args.gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
            train_bar.set_postfix(loss=train_loss/(i+1), mae=train_mae/(i+1))
        optimizer.step()
        model.zero_grad()
#        print("epoch:",epoch,"loss:",epoch_loss/len(dataloader))
        torch.save(model.state_dict(),'./checkpoints/epoch_'+str(epoch)+".pth")
    
        # testing phase
        model.eval()
        val_loss = 0
        val_mae = 0
        for i, (img,gt_dmap) in enumerate((val_loader)):
            img = img.to(args.device)
            gt_dmap = gt_dmap.to(args.device)

            # forward propagation
            et_dmap = model(img)
            loss = criterion(et_dmap, gt_dmap)
            val_loss += loss.item()
            val_mae += abs(et_dmap.data.sum()-gt_dmap.data.sum()).item()
            del img,gt_dmap,et_dmap

        if val_mae/len(val_loader) < min_mae:
            min_mae = val_mae/len(val_loader)
            min_epoch = epoch
        # print("epoch:" + str(epoch) + " error:" + str(mae/len(val_loader)) + " min_mae:"+str(min_mae) + " min_epoch:"+str(min_epoch))
        wandb.log({"loss/train": train_loss/len(train_loader),
                   "mae/train": train_mae/len(train_loader),
                   "loss/val": val_loss/len(val_loader),
                   "mae/val": val_mae/len(val_loader),
        }, commit=False)

        # show an image
        index = random.randint(0, len(val_loader)-1)
        img, gt_dmap = val_dataset[index]
        gt_dmap = gt_dmap.squeeze(0).detach().cpu().numpy()
        wandb.log({"image/img": [wandb.Image(img)]}, commit=False)
        wandb.log({"image/gt_dmap": [wandb.Image(gt_dmap/(gt_dmap.max())*255, caption=str(gt_dmap.sum()))]}, commit=False)

        img = img.unsqueeze(0).to(args.device)
        et_dmap = model(img)
        et_dmap = et_dmap.squeeze(0).detach().cpu().numpy()
        wandb.log({"image/et_dmap": [wandb.Image(et_dmap/(et_dmap.max())*255, caption=str(et_dmap.sum()))]})
        
    
    import time
    print(time.strftime('%Y.%m.%d %H:%M:%S',time.localtime(time.time())))



if __name__=="__main__":

    # configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='ShanghaiTech/part_B')
    # parser.add_argument("--train_image_root", default='./data/NWPU/train_data/images')
    # parser.add_argument("--train_dmap_root", default='./data/NWPU/train_data/density')
    # parser.add_argument("--val_image_root", default='./data/NWPU/val_data/images')
    # parser.add_argument("--val_dmap_root", default='./data/NWPU/val_data/density')
    parser.add_argument("--lr", default=1e-4)
    parser.add_argument("--batch_size", default=1)
    parser.add_argument("--momentum", default=0.95)
    parser.add_argument("--decay", default=5*1e-4)
    parser.add_argument("--epochs", default=100)
    parser.add_argument("--print_freq", default=30)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.steps = [-1,1,100,150]
    args.scales = [1,1,1,1]
    args.workers = 4
    args.seed = time.time()
    args.train_image_root = './data/' + args.dataset + '/train_data/images'
    args.train_dmap_root = './data/' + args.dataset + '/train_data/density'
    args.val_image_root = './data/' + args.dataset + '/val_data/images'
    args.val_dmap_root = './data/' + args.dataset + '/val_data/density'

    main(args)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    