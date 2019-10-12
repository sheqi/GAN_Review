import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

from model.snl_block import SNLStage
__all__ = ['PreResNet', 'preresnet20', 'preresnet32', 'preresnet44', 'preresnet56',
           'preresnet110', 'preresnet1202']


def model_hub(arch, pretrained=True, nl_type=None, nl_nums=None, stage_num=None,
              pool_size=7, div=2, nl_layer=['3'], relu=False, aff_kernel='dot'):
    """Model hub.
    """
    if arch == '56':
        return preresnet56(pretrained=pretrained,
                        nl_type=nl_type,
                        nl_nums=nl_nums,
                        stage_num = stage_num,
                        pool_size=pool_size, 
                        div=div,
                        nl_layer=nl_layer,
                        relu = relu,
                        aff_kernel=aff_kernel)
    elif arch == '20':
        return preresnet20(pretrained=pretrained,
                         nl_type=nl_type,
                         nl_nums=nl_nums,
                         stage_num = stage_num,
                         pool_size=pool_size, 
                         div = div,
                         nl_layer=nl_layer,
                         relu = relu)
    elif arch == '110':
        return preresnet110(pretrained=pretrained,
                         nl_type=nl_type,
                         nl_nums=nl_nums,
                         pool_size=pool_size, 
                         div = div,
                         nl_layer=nl_layer,
                         relu = relu)
    else:
        raise NameError("The arch '{}' is not supported yet in this repo. \
                You can add it by yourself.".format(arch))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class PreResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, nl_type=None, nl_nums=None, stage_num=None, pool_size=7, div=2, nl_layer=['3'], relu=False, aff_kernel='dot'):
        self.inplanes = 16
        self.aff_kernel = aff_kernel
        super(PreResNet, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        if '1' not in nl_layer:
            self.layer1 = self._make_layer(block, 16, layers[0])
        else:
            self.layer1 = self._make_layer(block, 16, layers[0], stride=1, layer='1', nl_type=nl_type, nl_nums=nl_nums, stage_num = stage_num, div = div,  relu=relu, nl_layer=nl_layer)
        
        if '2' not in nl_layer:
            self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        else:
            self.layer2 = self._make_layer(block, 32, layers[1], stride=2, layer='2', nl_type=nl_type, nl_nums=nl_nums, stage_num = stage_num, div = div,  relu=relu, nl_layer=nl_layer)

        if '3' not in nl_layer:
            self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        else:
            self.layer3 = self._make_layer(block, 64, layers[2], stride=2, layer='3', nl_type=nl_type, nl_nums=nl_nums, stage_num = stage_num, div = div,  relu=relu, nl_layer=nl_layer)
        self.bn1 = nn.BatchNorm2d(64*block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(64*block.expansion, 64*block.expansion, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64*block.expansion)
        self.fc2 = nn.Linear(64*block.expansion, num_classes)
        self.avgpool = nn.AvgPool2d(pool_size, stride=1)
        self.dropout = nn.Dropout(0.5)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1, nl_type=None, nl_nums=None, stage_num=None, div=2, layer = '3', nl_layer=['3'], relu=False):
 
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
            print(nl_layer)
            if nl_nums == 1 and (layer in nl_layer) and i == 2:
                layers.append(self._addNonlocal(self.inplanes,sub_planes, nl_type, stage_num, relu=relu))

            #######Add Res Block#######
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc1(self.relu(self.bn1(x)))

        x = self.relu(self.bn2(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


    def _addNonlocal(self, in_planes, sub_planes, nl_type='nl', stage_num=None, use_scale=False, groups=8, order=3, relu=False):
            if nl_type == 'snl':
                return SNLStage(
                    in_planes, sub_planes,
                    use_scale=False, stage_num=stage_num,
                    relu=relu, aff_kernel=self.aff_kernel)
            else:
                raise KeyError("Unsupported nonlocal type: {}".format(nl_type))


def preresnet20(pretrained=False, nl_type=None, nl_nums=None, stage_num=None, div=2, nl_layer=['3'], relu=False, **kwargs):
    """Constructs a PreResNet-20 model.
    """
    model = PreResNet(BasicBlock, [3, 3, 3], nl_type=nl_type, nl_nums=nl_nums, stage_num=stage_num, div=div, nl_layer=nl_layer, relu = relu, **kwargs)
    return model


def preresnet56(pretrained=False, nl_type=None, nl_nums=None, stage_num=None, div=2, nl_layer=['3'], relu=False, **kwargs):
    """Constructs a PreResNet-32 model.
    """
    model = PreResNet(Bottleneck, [9, 9, 9], nl_type=nl_type, nl_nums=nl_nums, stage_num=stage_num, div=div, nl_layer=nl_layer, relu = relu, **kwargs)
    return model



def preresnet110(pretrained=False, nl_type=None, nl_nums=None, stage_num=None, out_num=None, div=2, nl_layer=['3'],  relu=False, **kwargs):
    """Constructs a PreResNet-110 model.
    """
    model = PreResNet(Bottleneck, [18, 18, 18], nl_type=nl_type, nl_nums=nl_nums, stage_num=stage_num, div=div, nl_layer=nl_layer, relu = relu, **kwargs)
    return model

