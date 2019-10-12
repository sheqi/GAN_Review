# --------------------------------------------------------
# CGNL Network
# Copyright (c) 2018 Kaiyu Yue
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Functions for model building.
   Based on https://github.com/pytorch/vision
"""

import math
import torch
import torch.nn as nn

from termcolor import cprint
from collections import OrderedDict
from model.snl_block import SNLStage

__all__ = ['ResNet', 'resnet50', 'resnet101', 'resnet152']



def model_hub(arch, pretrained=True, nl_type=None, nl_nums=None, stage_num=None,
              pool_size=7, div=2, isrelu=False):
    """Model hub.
    """
    if arch == '50':
        return resnet50(pretrained=pretrained,
                        nl_type=nl_type,
                        nl_nums=nl_nums,
                        stage_num = stage_num,
                        pool_size=pool_size, div=div, isrelu=isrelu)
    elif arch == '101':
        return resnet101(pretrained=pretrained,
                         nl_type=nl_type,
                         nl_nums=nl_nums,
                         pool_size=pool_size)
    elif arch == '152':
        return resnet152(pretrained=pretrained,
                         nl_type=nl_type,
                         nl_nums=nl_nums,
                         pool_size=pool_size)
    else:
        raise NameError("The arch '{}' is not supported yet in this repo. \
                You can add it by yourself.".format(arch))


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding.
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000,
                 nl_type=None, nl_nums=None, stage_num=None, pool_size=7, div=2, isrelu=False, aff_kernel='dot'):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.aff_kernel = aff_kernel
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.nl_type = nl_type
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        if not nl_nums:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           nl_type=nl_type, nl_nums=nl_nums, stage_num = stage_num, div = div, isrelu=isrelu)
        if nl_nums != 5:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        else:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           nl_type=nl_type, nl_nums=nl_nums, stage_num = stage_num, div = div, isrelu=isrelu)

        self.avgpool = nn.AvgPool2d(pool_size, stride=1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if nl_nums == 1:
            for name, m in self._modules['layer3'][-2].named_modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm3d):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.GroupNorm):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)


    def _addNonlocal(self, in_planes, sub_planes, nl_type='nl', stage_num=None, use_scale=False, groups=8, order=3, isrelu=False):
            if nl_type == 'snl':
                return SNLStage(
                    in_planes, sub_planes,
                    use_scale=False, stage_num=stage_num,
                    relu=isrelu, aff_kernel=self.aff_kernel)
            else:
                raise KeyError("Unsupported nonlocal type: {}".format(nl_type))

    def _make_layer(self, block, planes, blocks, stride=1, nl_type=None, nl_nums=None, stage_num=None, div=2, isrelu=False):
        
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        sub_planes = int(self.inplanes / div)
        
        for i in range(1, blocks):
            #######Add Nonlocal Block#######
            if nl_nums == 1 and (i == 5 and blocks == 6) or (i == 22 and blocks == 23) or (i == 35 and blocks == 36):
                layers.append(self._addNonlocal(self.inplanes,sub_planes, nl_type, stage_num, isrelu=isrelu))
            if nl_nums == 5 and (stride == 2 and ((i == 1 and blocks == 6) or (i == 3 and blocks == 6) or (i == 5 and blocks == 6)\
                                   or (i == 1 and blocks ==3))):
                layers.append(self._addNonlocal(self.inplanes, sub_planes, nl_type, stage_num, isrelu=isrelu))

            #######Add Res Block#######
            layers.append(block(self.inplanes, planes))

        if nl_nums == 5 and stride == 2 and blocks==3:
            layers.append(self._addNonlocal(self.inplanes, sub_planes, nl_type, stage_num, isrelu=isrelu))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.nl_type == 'nl_lim':
            return att, x
        return x


def load_partial_weight(model, pretrained, nl_nums, nl_layer_id):
    """Loads the partial weights for NL/CGNL network.
    """
    _pretrained = pretrained
    _model_dict = model.state_dict()
    _pretrained_dict = OrderedDict()
    for k, v in _pretrained.items():
        ks = k.split('.')
        layer_name = '.'.join(ks[0:2])
        if nl_nums == 1 and \
                layer_name == 'layer3.{}'.format(nl_layer_id):
            ks[1] = str(int(ks[1]) + 1)
            k = '.'.join(ks)
        _pretrained_dict[k] = v
    _model_dict.update(_pretrained_dict)
    return _model_dict


def init_pretrained_weights(model, pretrained):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    _pretrained = pretrained
    _model_dict = model.state_dict()
    _pretrained_dict = OrderedDict()
    _pretrain_dict = {k: v for k, v in pretrained.items() if k in _model_dict and _model_dict[k].size() == v.size()}
    _model_dict.update(_pretrained_dict)
    return _model_dict


def resnet50(pretrained=False, nl_type=None, nl_nums=None, stage_num=None, isrelu = False, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   nl_type=nl_type, nl_nums=nl_nums, stage_num=stage_num, isrelu=isrelu, **kwargs)
    if pretrained:
        _pretrained = torch.load('pretrained/resnet50-19c8e357.pth')
        _model_dict = load_partial_weight(model, _pretrained, nl_nums, 5)
        model.load_state_dict(_model_dict)
    return model


def resnet101(pretrained=False, nl_type=None, nl_nums=None, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                   nl_type=nl_type, nl_nums=nl_nums, **kwargs)
    if pretrained:
        _pretrained = torch.load('pretrained/resnet101-5d3b4d8f.pth')
        _model_dict = load_partial_weight(model, _pretrained, nl_nums, 22)
        model.load_state_dict(_model_dict)
    return model


def resnet152(pretrained=False, nl_type=None, nl_nums=None, **kwargs):
    """Constructs a ResNet-152 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3],
                   nl_type=nl_type, nl_nums=nl_nums, **kwargs)
    if pretrained:
        _pretrained = torch.load('pretrained/resnet152-b121ed2d.pth')
        _model_dict = load_partial_weight(model, _pretrained, nl_nums, 35)
        model.load_state_dict(_model_dict)
    return model
