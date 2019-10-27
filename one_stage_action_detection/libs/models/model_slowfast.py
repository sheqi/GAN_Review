import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

from libs.tools.cfg import *
from libs.models.cfam import CFAMBlock
from libs.models.backbones_2d import darknet
from libs.models.backbones_3d import mobilenet, shufflenet, mobilenetv2, shufflenetv2, resnext, resnet, slowfast
from collections import OrderedDict

"""
YOWO model used in spatialtemporal action localization
"""


class YOWO_slowfast(nn.Module):

    def __init__(self, opt):
        super(YOWO_slowfast, self).__init__()
        self.opt = opt
        
        ##### 2D Backbone #####
        if opt.backbone_2d == "darknet":
            self.backbone_2d = darknet.Darknet("configs/cfg/yolo.cfg")
            num_ch_2d = 425 # Number of output channels for backbone_2d
        else:
            raise ValueError("Wrong backbone_2d model is requested. Please select\
                              it from [darknet]")
        if opt.backbone_2d_weights:# load pretrained weights on COCO dataset
            self.backbone_2d.load_weights(opt.backbone_2d_weights) 

        ##### 3D Backbone #####
        self.backbone_3d = slowfast.resnet50()
        num_ch_3d = 2048 # Number of output channels for backbone_3d

        if opt.backbone_3d_weights:# load pretrained weights on Kinetics-600 dataset
            #self.backbone_3d = slowfast.loadPretrained(self.backbone_3d, opt.backbone_3d_weights)
            pretrained_3d_backbone = torch.load(opt.backbone_3d_weights)

            backbone_3d_dict = OrderedDict()

            for k, v in pretrained_3d_backbone['state_dict'].items():
                ks = k.split('.')
                k = '.'.join(ks[1:])
                if k not in self.backbone_3d.state_dict():
                    print(k)
                    continue
                backbone_3d_dict[k] = v

            self.backbone_3d.load_state_dict(backbone_3d_dict) # 3. load the new state dict
            self.backbone_3d = self.backbone_3d.cuda()
            self.backbone_3d = nn.DataParallel(self.backbone_3d, device_ids=None) # Because the pretrained backbone models are saved in Dataparalled mode
            self.backbone_3d = self.backbone_3d.module # remove the dataparallel wrapper


        ##### Attention & Final Conv #####
        #############################
        self.slow_conv = nn.Conv2d(2048 + 256, 2048, kernel_size=3, stride=2, padding=1)
        self.fast_conv = nn.Conv3d(256, 256, kernel_size=(8, 1, 1), stride=1, padding=0)
        #############################
        self.cfam = CFAMBlock(num_ch_2d+num_ch_3d, 1024)
        self.conv_final = nn.Conv2d(1024, 5*(opt.n_classes+4+1), kernel_size=1, bias=False)
        self.seen = 0



    def forward(self, input):
        x_3d = input # Input clip
        x_2d = input[:, :, -1, :, :] # Last frame of the clip that is read

        x_2d = self.backbone_2d(x_2d)
        slow, fast = self.backbone_3d(x_3d)
        fast = self.fast_conv(fast)
        #print(fast.size(), slow.size())
        x_3d = self.slow_conv(torch.cat((slow, fast), dim=1).squeeze())
        #x_3d = self.backbone_3d(x_3d)
        #x_3d = torch.squeeze(x_3d, dim=2)
        #print(x_3d.size(), x_2d.size())
        x = torch.cat((x_3d, x_2d), dim=1)
        x = self.cfam(x)

        out = self.conv_final(x)

        return out


def get_fine_tuning_parameters_slowfast(model, opt):
    ft_module_names = ['cfam', 'conv_final', 'slow_conv', 'fast_conv'] # Always fine tune 'cfam' and 'conv_final'
    if not opt.freeze_backbone_2d:
        ft_module_names.append('backbone_2d') # Fine tune complete backbone_3d
    else:
        ft_module_names.append('backbone_2d.models.29') # Fine tune only layer 29 and 30
        ft_module_names.append('backbone_2d.models.30') # Fine tune only layer 29 and 30

    if not opt.freeze_backbone_3d:
        ft_module_names.append('backbone_3d') # Fine tune complete backbone_3d
    else:
        ft_module_names.append('backbone_3d.slow_res5') # Fine tune only layer 4
        ft_module_names.append('backbone_3d.fast_res5') # Fine tune only layer 4


    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                print(k)
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})
    
    return parameters
