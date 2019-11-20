import torch.nn as nn
import torch
from torch.autograd import Variable

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

      
class Encode_Vis_Att(nn.Module):
    def __init__(self, opt):
        super(Encode_Vis_Att, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, opt.ngh)
        self.fc4 = nn.Linear(opt.ngh, opt.attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, feat):      
        h = self.lrelu(self.fc1(feat))
        mix_feature = self.lrelu(self.fc2(h))
        h = self.lrelu(self.fc3(mix_feature))
        h = self.lrelu(self.fc4(h))
        return h, mix_feature

class Decode_Att_Vis(nn.Module):
    def __init__(self, opt):
        super(Decode_Att_Vis, self).__init__()
        self.fc1 = nn.Linear(opt.attSize, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, opt.ngh)
        self.fc4 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, feat):      
        h = self.lrelu(self.fc1(feat))
        mix_feature = self.lrelu(self.fc2(h))
        h = self.lrelu(self.fc3(mix_feature))
        h = self.lrelu(self.fc4(h))
        return h, mix_feature
    
class ODDetector(nn.Module):
    def __init__(self, input_dim, h_size, num_classes):
        super(ODDetector, self).__init__()
        self.relu = nn.ReLU(True)
        self.fc1 = nn.Linear(input_dim, h_size)
        self.fc2 = nn.Linear(h_size,h_size)
        self.classifier = nn.Linear(h_size, num_classes)
    
    def forward(self,x,center_loss=False):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        pred = self.classifier(h)
        return pred
