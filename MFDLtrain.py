import os
import sys
import time
import torch
import torch.nn
import argparse
from PIL import Image
import numpy as np
from validate import validate
from data import create_dataloader
from networks.MFDLtrainer import Trainer
from options.train_options import TrainOptions
from options.test_options import TestOptions
from util import Logger
from tqdm import tqdm
from eval import eval
import math





if __name__ == '__main__':
    opt = TrainOptions().parse()
    Testdataroot = os.path.join(opt.dataroot, 'test')
    opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)
    print('  '.join(list(sys.argv)) )
    data_loader = create_dataloader(opt)

    
    model = Trainer(opt)
    model.train()
    acc=0
    ap=0
    for epoch in range(opt.niter):
        epoch_iter = 0
        newacc=0
        newap=0
        total_loss = 0  # 初始化总损失
        num=0.2*len(data_loader)
        num=math.floor(num)
        with tqdm(total=num, desc=f'Epoch {epoch + 1}/{opt.niter}', unit='batch') as pbar:
            for i, data in enumerate(data_loader):
                if(i>num):
                    break;
                
                model.total_steps += 1
                epoch_iter += opt.batch_size

                model.set_input(data)
                model.optimize_parameters()
                loss=model.get_loss()
                # 获取当前损
                # 更新进度条，显示损失
                pbar.set_postfix(loss=loss)
                pbar.update(1)

            # 计算平均损失
            avg_loss = total_loss / len(data_loader)
            print(f'Epoch {epoch + 1} Average Loss: {avg_loss}')

            if epoch % opt.delr_freq == 0 and epoch != 0:
                model.adjust_learning_rate()
        path=f'MFDL{epoch+1}.pth'
        if epoch+1==0:
            torch.save(model.model.state_dict(),f'MFDLsd_{epoch}.pth')
        # newacc,newap=eval(path)
        #     # newacc,newap=eval(model)
        # if newacc>acc:
        #     torch.save(model.model.state_dict(),f'MFDLbest{epoch+1}.pth')
        #     print(f'acc:{newacc},ap:{newap}')

    
    
