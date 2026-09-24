import sys
import time
import os
import csv
import torch
from util import Logger, printSet
from validate import validate
from networks.MFDL import mfdl
from options.test_options import TestOptions
import numpy as np
import random


DetectionTests = {


           
            # 'univer' :{'dataroot'   : '/root/xzl/NewData/univerFake',
            #              'no_resize'  :True,
            #              'no_crop'    : False,},                  
            # # 'Cross-GAN_crop':{
            # #     'dataroot'   : '/root/xzl/NewData/GANAdd/crop',
            # #     'no_resize'  : True,
            # #     'no_crop'    : False,
            # # },
            # 'Cross-GAN':{
            #     'dataroot'   : '/root/xzl/NewData/GANAdd',
            #     'no_resize'  : True,
            #     'no_crop'    : False,
            # },
            # 'Forennormal':{
            #   'dataroot'   : '/root/xzl/NewData/ForenSynths_4classtrain_val_test/',
            #   'no_resize'  : True,
            #   'no_crop'    : False,},

            #      },
            #                  'Forespecial':{
            #   'dataroot'   : '/root/xzl/NewData/ForenSynths_4classtrain_val_test/test/special',
            #   'no_resize'  : True,
            #   'no_crop'    : False,

            #      },
                #  'GenImagecrop':{
                #     'dataroot'   : '/root/xzl/NewData/GenImage/crop',
                #     'no_resize'  : True,
                #     'no_crop'    : False,
                #  },
                                  
                # 'GenImage':{
                #     'dataroot'   : '/docker_back/public/xzl_root/NewData/GenImage',
                #     'no_resize'  : True,
                #     'no_crop'    : False,
                #  },
                #  'DIRE': { 'dataroot'   : '/root/xzl/NewData/dire',
                #                  'no_resize'  : True,
                #                  'no_crop'    : False,
                #                },
                # 'robust':{
                #        'dataroot'   : '/root/xzl/NewData/robust',
                #        'no_resize'  : True,
                #        'no_crop'    : False,
                #    } 
                'little':{
                    'dataroot'   : '/data_1/ywk/dataset/data',
                    'no_resize'  : True,
                    'no_crop'    : False,
                 },
                 }


opt = TestOptions().parse(print_options=False)
print(f'Model_path {opt.model_path}')

# get model
model = mfdl(num_classes=1)

# from collections import OrderedDict
# from copy import deepcopy
# state_dict = torch.load(opt.model_path, map_location='cpu')['model']
# pretrained_dict = OrderedDict()
# for ki in state_dict.keys():
    # pretrained_dict[ki[7:]] = deepcopy(state_dict[ki])
# model.load_state_dict(pretrained_dict, strict=True)

model.load_state_dict(torch.load(opt.model_path, map_location='cpu'), strict=True)
model.cuda()
model.eval()

start_time=time.time()
for testSet in DetectionTests.keys():
    dataroot = DetectionTests[testSet]['dataroot']
    printSet(testSet)

    accs = [];aps = []
    # print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    for v_id, val in enumerate(os.listdir(dataroot)):
        opt.dataroot = '{}/{}'.format(dataroot, val)
        opt.classes  = '' #os.listdir(opt.dataroot) if multiclass[v_id] else ['']
        print("当前路径:", opt.dataroot)
        print("子目录:", os.listdir(opt.dataroot))
        opt.no_resize = DetectionTests[testSet]['no_resize']
        opt.no_crop   = DetectionTests[testSet]['no_crop']
        acc, ap, _, _, _, _ = validate(model, opt)
        accs.append(acc);aps.append(ap)
        print("({} {:12}) acc: {:.1f}; ap: {:.1f}".format(v_id, val, acc*100, ap*100))
    print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100));print('*'*25) 

# totaltime=time.time()-start_time
# print(totaltime)