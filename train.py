import numpy as np
import time
import torch
import torch.nn as nn
import os
# import visdom
import wandb
import random
import argparse
from tqdm import tqdm as tqdm

from cannet import CANNet
from my_dataset import CrowdDataset


def main(args):
    wandb.init(project="crowd-counting", config=args)
    args = wandb.config
    # print(args)

    # vis=visdom.Visdom()
    torch.cuda.manual_seed(args.seed)
    model=CANNet().to(args.device)
    criterion=nn.MSELoss(size_average=False).to(args.device)
    optimizer=torch.optim.SGD(model.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=0)
#    optimizer=torch.optim.Adam(model.parameters(),args.lr)
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
        epoch_loss = 0
        for i, (img,gt_dmap) in enumerate((train_loader)):
            img = img.to(args.device)
            gt_dmap = gt_dmap.to(args.device)
            # forward propagation
            et_dmap = model(img)
            # calculate loss
            loss = criterion(et_dmap,gt_dmap)
            loss = loss/args.gradient_accumulation_steps
            loss.backward()
            epoch_loss += loss.item()
            if (i+1)%args.gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
        optimizer.step()
        model.zero_grad()
#        print("epoch:",epoch,"loss:",epoch_loss/len(dataloader))
        torch.save(model.state_dict(),'./checkpoints/epoch_'+str(epoch)+".pth")
    
        # testing phase
        model.eval()
        mae = 0
        for i, (img,gt_dmap) in enumerate((val_loader)):
            img = img.to(args.device)
            gt_dmap = gt_dmap.to(args.device)
            # forward propagation
            et_dmap = model(img)
            mae += abs(et_dmap.data.sum()-gt_dmap.data.sum()).item()
            del img,gt_dmap,et_dmap
        if mae/len(val_loader) < min_mae:
            min_mae = mae/len(val_loader)
            min_epoch = epoch
        print("epoch:" + str(epoch) + " error:" + str(mae/len(val_loader)) + " min_mae:"+str(min_mae) + " min_epoch:"+str(min_epoch))
        wandb.log({"loss": epoch_loss/len(train_loader),
                   "error": mae/len(val_loader),
        })

        # show an image
        index = random.randint(0, len(val_loader)-1)
        img, gt_dmap = val_dataset[index]
        wandb.log({"image/img": [wandb.Image(img)]})
        wandb.log({"image/gt_dmap": [wandb.Image(gt_dmap/(gt_dmap.max())*255, caption=str(gt_dmap.sum()))]})

        img = img.unsqueeze(0).to(args.device)
        gt_dmap = gt_dmap.unsqueeze(0)
        et_dmap = model(img)
        et_dmap = et_dmap.squeeze(0).detach().cpu().numpy()
        wandb.log({"image/et_dmap": [wandb.Image(et_dmap/(et_dmap.max())*255, caption=str(et_dmap.sum()))]})
        
    
    import time
    print(time.strftime('%Y.%m.%d %H:%M:%S',time.localtime(time.time())))



if __name__=="__main__":

    # configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='ShanghaiTech/part_A')
    # parser.add_argument("--train_image_root", default='./data/NWPU/train_data/images')
    # parser.add_argument("--train_dmap_root", default='./data/NWPU/train_data/density')
    # parser.add_argument("--val_image_root", default='./data/NWPU/val_data/images')
    # parser.add_argument("--val_dmap_root", default='./data/NWPU/val_data/density')
    parser.add_argument("--lr", default=1e-7)
    parser.add_argument("--batch_size", default=1)
    parser.add_argument("--momentum", default=0.95)
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    