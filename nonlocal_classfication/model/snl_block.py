import math
import torch
import torch.nn as nn

from termcolor import cprint
from collections import OrderedDict




#########################################Spatial SNL Block#######################################################################
class SNLStage(nn.Module):
    def __init__(self, inplanes, planes, stage_num=5, use_scale=False, relu=False, aff_kernel='dot'):
        super(SNLStage, self).__init__()
        self.down_channel = planes
        self.output_channel = inplanes
        self.num_block = stage_num
        self.input_channel = inplanes
        self.softmax = nn.Softmax(dim=2)
        self.aff_kernel = aff_kernel
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.p = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.use_scale = use_scale


        layers = []
        for i in range(stage_num):
            layers.append(SNLUnit(inplanes, planes, relu=relu))

        self.stages = nn.Sequential(*layers)
        self._init_params()


    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def DotKernel(self, x):

        t = self.t(x)
        p = self.p(x)

        b, c, h, w = t.size()

        t = t.view(b, c, -1).permute(0, 2, 1)
        p = p.view(b, c, -1)

        att = torch.bmm(t, p)
        att = (att + att.permute(0, 2, 1)) / 2

        att = self.softmax(att)

        return att


    def forward(self, x):

        if self.aff_kernel == 'dot':
            att = self.DotKernel(x)
        #elif self.aff_kernel == 'embedgassian':
        #    att = self.EbdedGassKernel(x)
        #elif self.aff_kernel == "gassian":
        #    att = self.GassKernel(x)
        #elif self.aff_kernel == 'rbf':
        #    att = self.RBFGassKeneral(x)
        else:
            raise KeyError("Unsupported nonlocal type: {}".format(nl_type))

        if self.use_scale:
            att = att.div(c**0.5)

        out = x

        for cur_stage in self.stages:
            out = cur_stage(out, att)

        return out



class SNLUnit(nn.Module):
    def __init__(self, inplanes, planes, use_scale=False, relu=False, aff_kernel='dot'):
        self.use_scale = use_scale

        super(SNLUnit, self).__init__()

        self.g = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(inplanes)
        self.w_1 = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1, bias=False)
        self.w_2 = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1, bias=False)

        self.is_relu = relu
        if self.is_relu:
            self.relu = nn.ReLU(inplace=True)

        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, att):
        residual = x

        g = self.g(x)

        b, c, h, w = g.size()

        g = g.view(b, c, -1).permute(0, 2, 1)

        x_1 = g.permute(0, 2, 1).contiguous().view(b,c,h,w)
        x_1 = self.w_1(x_1)

        out = x_1

        x_2 = torch.bmm(att, g)
        x_2 = x_2.permute(0, 2, 1)
        x_2 = x_2.contiguous()
        x_2 = x_2.view(b, c, h, w)
        x_2 = self.w_2(x_2)
        out = out + x_2

        out = self.bn(out)

        if self.is_relu:
            out = self.relu(out)

        out = out + residual

        return out
#################################################################################################################




#########################################Spatial gSNL Block######################################################################
class gSNLStage(nn.Module):
    def __init__(self, inplanes, planes, stage_num=5, use_scale=False, relu=False, aff_kernel='dot'):
        super(gSNLStage, self).__init__()
        self.down_channel = planes
        self.output_channel = inplanes
        self.num_block = stage_num
        self.input_channel = inplanes
        self.softmax = nn.Softmax(dim=2)
        self.aff_kernel = aff_kernel
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.p = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.use_scale = use_scale


        layers = []
        for i in range(stage_num):
            layers.append(gSNLUnit(inplanes, planes, relu=relu))

        self.stages = nn.Sequential(*layers)



        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def DotKernel(self, x):

        t = self.t(x)
        p = self.p(x)

        b, c, h, w = t.size()

        t = t.view(b, c, -1).permute(0, 2, 1)
        p = p.view(b, c, -1)

        att = torch.bmm(t, p)

        att = self.softmax(att)

        return att


    def forward(self, x):

        if self.aff_kernel == 'dot':
            att = self.DotKernel(x)
        #elif self.aff_kernel == 'embedgassian':
        #    att = self.EbdedGassKernel(x)
        #elif self.aff_kernel == "gassian":
        #    att = self.GassKernel(x)
        #elif self.aff_kernel == 'rbf':
        #    att = self.RBFGassKeneral(x)
        else:
            raise KeyError("Unsupported nonlocal type: {}".format(nl_type))

        if self.use_scale:
            att = att.div(c**0.5)

        out = x

        for cur_stage in self.stages:
            out = cur_stage(out, att)

        return out



class gSNLUnit(nn.Module):
    def __init__(self, inplanes, planes, use_scale=False, relu=False, aff_kernel='dot'):
        self.use_scale = use_scale

        super(gSNLUnit, self).__init__()

        self.g = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(inplanes)
        self.w_1 = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1, bias=False)
        self.w_2 = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1, bias=False)
        self.w_3 = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1, bias=False)

        self.is_relu = relu
        if self.is_relu:
            self.relu = nn.ReLU(inplace=True)

        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, att):
        residual = x

        g = self.g(x)

        b, c, h, w = g.size()

        g = g.view(b, c, -1).permute(0, 2, 1)

        x_1 = g.permute(0, 2, 1).contiguous().view(b,c,h,w)
        x_1 = self.w_1(x_1)

        out = x_1

        x_2 = torch.bmm(att, g)
        x_2 = x_2.permute(0, 2, 1)
        x_2 = x_2.contiguous()
        x_2 = x_2.view(b, c, h, w)
        x_2 = self.w_2(x_2)
        out = out + x_2

        I_n = torch.Tensor(torch.eye(g.size()[1])).cuda()
        I_n = I_n.expand(att.size())
        x_3 = torch.bmm(2*att-I_n, g)
        x_3 = x_3.permute(0, 2, 1)
        x_3 = x_3.contiguous()
        x_3 = x_3.view(b, c, h, w)
        x_3 = self.w_3(x_3)
        out = out + x_3

        out = self.bn(out)

        if self.is_relu:
            out = self.relu(out)

        out = out + residual

        return out
###############################################################################################################









#########################################Spatial Temporal SNL Block#######################################################################
class st_SNLStage(nn.Module):
    def __init__(self, inplanes, planes, stage_num=5, use_scale=False, relu=False, aff_kernel='dot'):
        super(st_SNLStage, self).__init__()
        self.down_channel = planes
        self.output_channel = inplanes
        self.num_block = stage_num
        self.input_channel = inplanes
        self.softmax = nn.Softmax(dim=2)
        self.aff_kernel = aff_kernel
        self.t = nn.Conv3d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.p = nn.Conv3d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.use_scale = use_scale


        layers = []
        for i in range(stage_num):
            layers.append(st_SNLUnit(inplanes, planes, relu=relu))

        self.stages = nn.Sequential(*layers)
        self._init_params()


    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def DotKernel(self, x):

        t = self.t(x)
        p = self.p(x)

        b, c, n, h, w = t.size()

        t = t.view(b, c, -1).permute(0, 2, 1)
        p = p.view(b, c, -1)

        att = torch.bmm(t, p)

        att = self.softmax(att)

        return att


    def forward(self, x):

        if self.aff_kernel == 'dot':
            att = self.DotKernel(x)
        #elif self.aff_kernel == 'embedgassian':
        #    att = self.EbdedGassKernel(x)
        #elif self.aff_kernel == "gassian":
        #    att = self.GassKernel(x)
        #elif self.aff_kernel == 'rbf':
        #    att = self.RBFGassKeneral(x)
        else:
            raise KeyError("Unsupported nonlocal type: {}".format(nl_type))

        if self.use_scale:
            att = att.div(c**0.5)

        out = x

        for cur_stage in self.stages:
            out = cur_stage(out, att)

        return out



class st_SNLUnit(nn.Module):
    def __init__(self, inplanes, planes, use_scale=False, relu=False, aff_kernel='dot'):
        self.use_scale = use_scale

        super(st_SNLUnit, self).__init__()

        self.g = nn.Conv3d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm3d(inplanes)
        self.w_1 = nn.Conv3d(planes, inplanes, kernel_size=1, stride=1, bias=False)
        self.w_2 = nn.Conv3d(planes, inplanes, kernel_size=1, stride=1, bias=False)

        self.is_relu = relu
        if self.is_relu:
            self.relu = nn.ReLU(inplace=True)

        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, att):
        residual = x

        g = self.g(x)

        b, c, n, h, w = g.size()

        g = g.view(b, c, -1).permute(0, 2, 1)

        x_1 = g.permute(0, 2, 1).contiguous().view(b,c,n,h,w)
        x_1 = self.w_1(x_1)

        out = x_1

        x_2 = torch.bmm(att, g)
        x_2 = x_2.permute(0, 2, 1)
        x_2 = x_2.contiguous()
        x_2 = x_2.view(b, c, n, h, w)
        x_2 = self.w_2(x_2)
        out = out + x_2

        out = self.bn(out)

        if self.is_relu:
            out = self.relu(out)

        out = out + residual

        return out

###############################################################################################################################





















