from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import math
import util 
import classifier2 as classifier
import classifier_entropy
import my_model as model
import numpy as np

def generate_register(netAV, classes, attribute, opt):
    # generate num synthetic samples for each class in classes
    nclass = classes.size(0)
    feature_reg = torch.FloatTensor(nclass, opt.ngh)
    label_reg = torch.LongTensor(nclass) 

    if opt.cuda:
        feature_reg = feature_reg.cuda()
        label_reg = label_reg.cuda()

    for i in range(nclass):
        iclass = classes[i].cuda()
        iclass_att = attribute[iclass].unsqueeze(0)
        with torch.no_grad():
            _, output = netAV(Variable(iclass_att).cuda())
        feature_reg[i, :] = output
        label_reg[i] = iclass
        
    return feature_reg, label_reg

def generate_register_v2m(netVA, classes, attribute, data, opt, seen=True):
    # generate num synthetic samples for each class in classes
    nclass = classes.size(0)
    feature_reg = torch.FloatTensor(nclass, opt.ngh)
    label_reg = torch.LongTensor(nclass) 
    train_feature = data.train_feature
    train_label = data.train_label
    num_lab = torch.zeros(nclass)
    map_class = {}
    if seen:
        start = 0
        end = opt.num_seen
    else:
        start = opt.num_seen + 1
        end = len(train_label)
    for i in range(nclass):
        map_class[classes[i].item()] = i
    if opt.cuda:
        feature_reg = feature_reg.cuda()
        label_reg = label_reg.cuda()
    for i in range(start, end):
        cur_feature = train_feature[i].cuda()
        cur_label = map_class[train_label[i].item()]
        with torch.no_grad():
            _, output = netVA(Variable(cur_feature))
        feature_reg[cur_label, :] += output
        num_lab[cur_label] += 1
    for i in range(nclass):
        label_reg[i] = classes[i].cuda()
        feature_reg[i, :] = feature_reg[i, :] / num_lab[i]

    '''    
    for i in range(nclass):
        iclass = classes[i].cuda()
        iclass_att = attribute[iclass].unsqueeze(0)
        with torch.no_grad():
            _, output = netAV(Variable(iclass_att).cuda())
        feature_reg[i, :] = output
        label_reg[i] = iclass
    '''
        
    return feature_reg, label_reg



def compute_per_class_acc_zsl(test_label, predicted_label, nclass):
    acc_per_class = torch.FloatTensor(nclass).fill_(0)
    sum_acc = 0
    for i in range(nclass):
        idx = (test_label == i)
        if torch.sum(idx) == 0: #no examples for this class
            continue
        acc_per_class[i] = float(torch.sum(test_label[idx]==predicted_label[idx])) / float( torch.sum(idx))
        sum_acc = sum_acc + float(torch.sum(test_label[idx]==predicted_label[idx]))
    return float(sum_acc) / float(predicted_label.size(0)), sum_acc, predicted_label.size(0)


def eval_unseen(netAV, netVA, data, opt):
    if opt.use_train and opt.use_gan:
        feature_register, regster_label = generate_register_v2m(netVA, data.unseenclasses, data.attribute, data, opt, seen=False)
    else:
        feature_register, regster_label = generate_register(netAV, data.unseenclasses, data.attribute, opt)
    feature_register = feature_register.unsqueeze(0).expand(opt.batch_size, -1, -1)
    test_feature = data.test_unseen_feature
    test_label = data.test_unseen_label
    regster_label = data.unseenclasses    

    mean_acc, _, _ = run_eval(netAV, netVA, test_feature, test_label, regster_label, feature_register)
    return mean_acc


def run_eval(netAV, netVA, test_feature, test_label, regster_label, feature_register, opt):
    n_test = test_feature.size(0)
    predicted_label = torch.LongTensor(test_label.size())
    start = 0
    for i in range(0, n_test, opt.batch_size):
        end = min(n_test, start + opt.batch_size)
        with torch.no_grad():
            _, mix_output = netVA(Variable(test_feature[start:end].cuda()))
        if feature_register.size(0) > mix_output.size(0):
            feature_register = feature_register[0:mix_output.size(0), :, :]
        mix_output = mix_output.unsqueeze(1).expand(feature_register.size())
        dis = mix_output - feature_register
        dis = torch.pow(dis, 2)
        dis = torch.sum(dis, dim=2)
        dis = torch.sqrt(dis)
        #print(dis.size())
        _, predicted_label[start:end] = torch.min(dis, dim=1)
        start = end
    mean_acc, sum_acc, num = compute_per_class_acc_zsl(util.map_label(test_label, regster_label), predicted_label, regster_label.size(0))

    return mean_acc, sum_acc, num



def eval_gzsl(netAV, netVA, data, opt, online_train=False):
    if opt.use_train:
        if opt.use_gan:
            feature_register_unseen, regster_label_unseen = generate_register_v2m(netVA, data.unseenclasses, data.attribute, data, opt, seen=False)
        else:
            feature_register_unseen, regster_label_unseen = generate_register(netAV, data.unseenclasses, data.attribute, opt)
        feature_register_seen, regster_label_seen = generate_register_v2m(netVA, data.seenclasses, data.attribute, data, opt, seen=True)
    else:
        feature_register_unseen, regster_label_unseen = generate_register(netAV, data.unseenclasses, data.attribute, opt)
        feature_register_seen, regster_label_seen = generate_register(netAV, data.seenclasses, data.attribute, opt)
    #feature_register_unseen, regster_label_unseen = generate_register_v2m(netVA, data.unseenclasses, data.attribute, data, opt)
    #feature_register_seen, regster_label_seen = generate_register(netAV, data.seenclasses, data.attribute, opt)
    #feature_register_seen, regster_label_seen = generate_register_v2m(netVA, data.seenclasses, data.attribute, data, opt)
    regster_label = torch.cat((data.unseenclasses, data.seenclasses), dim=0)
    feature_register = torch.cat((feature_register_unseen, feature_register_seen), dim = 0)
    feature_register = feature_register.unsqueeze(0).expand(opt.batch_size, -1, -1)

    test_feature = data.test_unseen_feature
    test_label = data.test_unseen_label
    #unseenclasses = data.unseenclasses  

    if online_train:
        mean_acc_unseen, sum_acc_unseen, num_unseen = run_eval_ol(netAV, netVA, test_feature, test_label, regster_label, feature_register, opt)
    else:
        mean_acc_unseen, sum_acc_unseen, num_unseen = run_eval(netAV, netVA, test_feature, test_label, regster_label, feature_register, opt)        
    test_feature_seen = data.test_seen_feature
    test_label_seen = data.test_seen_label
    seenclasses= data.seenclasses  
    if online_train:
        mean_acc_seen, sum_acc_seen, num_seen = run_eval_ol(netAV, netVA, test_feature_seen, test_label_seen, regster_label, feature_register, opt)
    else:
        mean_acc_seen, sum_acc_seen, num_seen = run_eval(netAV, netVA, test_feature_seen, test_label_seen, regster_label, feature_register, opt)
    mean_acc = float(sum_acc_unseen + sum_acc_seen) / float(num_unseen + num_seen)
    
    return mean_acc, mean_acc_seen, mean_acc_unseen


def eval_seen(netAV, netVA, data, opt):
    if opt.use_train:
        feature_register, regster_label = generate_register_v2m(netVA, data.seenclasses, data.attribute, data, opt, seen=True)
    else:
        feature_register, regster_label = generate_register(netAV, data.seenclasses, data.attribute, opt)
    feature_register = feature_register.unsqueeze(0).expand(opt.batch_size, -1, -1)
    test_feature = data.test_seen_feature
    test_label = data.test_seen_label
    regster_label= data.seenclasses
 
    mean_acc, _, _ = run_eval(netAV, netVA, test_feature, test_label, regster_label, feature_register, opt)
    return mean_acc

############################################
def run_eval_ol(netAV, netVA, test_feature, test_label, regster_label, feature_register, opt):
    n_test = test_feature.size(0)
    predicted_label = torch.LongTensor(test_label.size())
    start = 0
    for i in range(0, n_test, opt.batch_size):
        end = min(n_test, start + opt.batch_size)
        with torch.no_grad():
            _, mix_output = netVA(Variable(test_feature[start:end].cuda()))
        if feature_register.size(0) > mix_output.size(0):
            feature_register = feature_register[0:mix_output.size(0), :, :]
        mix_output_org = mix_output.clone()
        mix_output = mix_output.unsqueeze(1).expand(feature_register.size())
        dis = mix_output - feature_register
        dis = torch.pow(dis, 2)
        dis = torch.sum(dis, dim=2)
        dis = torch.sqrt(dis)
        #print(dis.size())
        _ , predicted_label[start:end] = torch.min(dis, dim=1)
        ######
        _ , register_nb_idx = torch.min(dis, dim=0)
        update_f = torch.index_select(mix_output_org, 0, register_nb_idx).unsqueeze(0)
        #print(update_f.size())
        feature_register = (feature_register + update_f) / 2
        ######
        start = end
    mean_acc, sum_acc, num = compute_per_class_acc_zsl(util.map_label(test_label, regster_label), predicted_label, regster_label.size(0))

    return mean_acc, sum_acc, num


def eval_gzsl_with_st(netAV, netVA, data, opt, online_train=False, k=20):
    feature_register_unseen, regster_label_unseen = generate_register(netAV, data.unseenclasses, data.attribute, opt)
    feature_register_seen, regster_label_seen = generate_register(netAV, data.seenclasses, data.attribute, opt)
    regster_label = torch.cat((data.unseenclasses, data.seenclasses), dim=0)
    feature_register = torch.cat((feature_register_unseen, feature_register_seen), dim = 0)
    feature_register = feature_register.unsqueeze(0).expand(opt.batch_size, -1, -1)
    ####################################
    st_feature = torch.cat((data.test_unseen_feature, data.test_seen_feature), dim=0)
    st_label = torch.cat((data.test_unseen_label, data.test_seen_label), dim=0)
    n_test = st_feature.size(0)
    predicted_label = torch.LongTensor(st_label.size())
    start = 0
    for i in range(0, n_test, opt.batch_size):
        end = min(n_test, start + opt.batch_size)
        with torch.no_grad():
            _, mix_output = netVA(Variable(st_feature[start:end].cuda()))
        if feature_register.size(0) > mix_output.size(0):
            feature_register = feature_register[0:mix_output.size(0), :, :]
        mix_output_org = mix_output.clone()
        mix_output = mix_output.unsqueeze(1).expand(feature_register.size())
        dis = mix_output - feature_register
        dis = torch.pow(dis, 2)
        dis = torch.sum(dis, dim=2)
        dis = torch.sqrt(dis)
        if start == 0:
            all_dis = dis.cpu()
            all_feature = mix_output_org.cpu()
        else:
            all_dis = torch.cat((all_dis, dis.cpu()), dim=0)
            all_feature = torch.cat((all_feature, mix_output_org.cpu()), dim=0)
        start = end
    _ , register_nb_idx = all_dis.topk(k, dim=0, largest=False)
    for i in range(k):
        cur_idx = register_nb_idx[i, :]
        update_f = torch.index_select(all_feature, 0, cur_idx).unsqueeze(0)
        feature_register = feature_register + update_f.cuda() / float(k)
    feature_register = feature_register / 2
    feature_register = feature_register[0, :, :]
    feature_register = feature_register.unsqueeze(0).expand(opt.batch_size, -1, -1)
    ####################################


    test_feature = data.test_unseen_feature
    test_label = data.test_unseen_label
    #unseenclasses = data.unseenclasses  

    if online_train:
        mean_acc_unseen, sum_acc_unseen, num_unseen = run_eval_ol(netAV, netVA, test_feature, test_label, regster_label, feature_register, opt)
    else:
        mean_acc_unseen, sum_acc_unseen, num_unseen = run_eval(netAV, netVA, test_feature, test_label, regster_label, feature_register, opt)        
    test_feature_seen = data.test_seen_feature
    test_label_seen = data.test_seen_label
    seenclasses= data.seenclasses  
    if online_train:
        mean_acc_seen, sum_acc_seen, num_seen = run_eval_ol(netAV, netVA, test_feature_seen, test_label_seen, regster_label, feature_register, opt)
    else:
        mean_acc_seen, sum_acc_seen, num_seen = run_eval(netAV, netVA, test_feature_seen, test_label_seen, regster_label, feature_register, opt)
    mean_acc = float(sum_acc_unseen + sum_acc_seen) / float(num_unseen + num_seen)
    
    return mean_acc, mean_acc_seen, mean_acc_unseen


##########generate mask#############

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()
        self.logsoft = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, neg=True, batch=False):
        b = self.softmax(x) * self.logsoft(x)
        if batch:
            return -1.0 * b.sum(1)
        if neg:
            return -1.0 * b.sum()/x.size(0)
        else:
            return  b.sum()/x.size(0)

def compute_per_class_acc_od(test_label, predicted_label, nclass, mask):
    acc_per_class = torch.FloatTensor(nclass).fill_(0)
    sum_acc = 0
    for i in range(nclass):
        idx = (test_label == i)
        if torch.sum(idx) == 0: #no examples for this class
            continue
        
        acc_per_class[i] = float(torch.sum( (test_label[idx]==predicted_label[idx]) * mask[idx])) / float( torch.sum(idx))
        sum_acc = sum_acc + float(torch.sum( (test_label[idx]==predicted_label[idx]) * mask[idx]))
    return float(sum_acc) / float(predicted_label.size(0)), sum_acc, predicted_label.size(0)


def run_eval_odmask(netAV, netVA, netOD, thresh, test_feature, test_label, regster_label, feature_register, opt, seen_classes=True):
    n_test = test_feature.size(0)
    predicted_label = torch.LongTensor(test_label.size())
    start = 0
    criterion = HLoss().cuda()
    entropy = []
    for i in range(0, n_test, opt.batch_size):
        end = min(n_test, start + opt.batch_size)
        with torch.no_grad():
            _, mix_output = netVA(Variable(test_feature[start:end].cuda()))
        if feature_register.size(0) > mix_output.size(0):
            feature_register = feature_register[0:mix_output.size(0), :, :]
        output_od = netOD(Variable(test_feature[start:end].cuda()))
        entropy_batch = criterion(output_od, batch=True)
        entropy.extend(entropy_batch.data.view(-1).cpu().numpy())
        mix_output = mix_output.unsqueeze(1).expand(feature_register.size())
        dis = mix_output - feature_register
        dis = torch.pow(dis, 2)
        dis = torch.sum(dis, dim=2)
        dis = torch.sqrt(dis)
        #print(dis.size())
        _, predicted_label[start:end] = torch.min(dis, dim=1)
        start = end
    seen_mask = torch.Tensor(np.array(entropy)) < thresh
    if not seen_classes:
        seen_mask = 1 - seen_mask

    od_sum_acc = torch.sum(seen_mask)
    mean_acc, sum_acc, num = compute_per_class_acc_od(util.map_label(test_label, regster_label), predicted_label, regster_label.size(0), seen_mask)

    return mean_acc, sum_acc, num, od_sum_acc

def eval_seen_od(netAV, netVA, netOD, thresh, data, opt):
    if opt.use_train:
        feature_register, regster_label = generate_register_v2m(netVA, data.seenclasses, data.attribute, data, opt, seen=True)
    else:
        feature_register, regster_label = generate_register(netAV, data.seenclasses, data.attribute, opt)
    feature_register = feature_register.unsqueeze(0).expand(opt.batch_size, -1, -1)
    test_feature = data.test_seen_feature
    test_label = data.test_seen_label
    regster_label= data.seenclasses
 
    mean_acc, sum_acc, num, od_sum_acc = run_eval_odmask(netAV, netVA, netOD, thresh, test_feature, test_label, regster_label, feature_register, opt, True)
    return mean_acc, sum_acc, num, od_sum_acc

def eval_unseen_od(netAV, netVA, netOD, thresh, data, opt):
    if opt.use_train and opt.use_gan:
        feature_register, regster_label = generate_register_v2m(netVA, data.unseenclasses, data.attribute, data, opt, seen=False)
    else:
        feature_register, regster_label = generate_register(netAV, data.unseenclasses, data.attribute, opt)
    feature_register = feature_register.unsqueeze(0).expand(opt.batch_size, -1, -1)
    test_feature = data.test_unseen_feature
    test_label = data.test_unseen_label
    regster_label = data.unseenclasses    

    mean_acc, sum_acc, num, od_sum_acc = run_eval_odmask(netAV, netVA, netOD, thresh, test_feature, test_label, regster_label, feature_register, opt, False)
    return mean_acc, sum_acc, num, od_sum_acc


def eval_od(netAV, netVA, netOD, thresh, data, opt):
    seen_acc, seen_sum_acc, seen_num, seen_od_sum_acc = eval_seen_od(netAV, netVA, netOD, thresh, data, opt)
    unseen_acc, unseen_sum_acc, unseen_num, unseen_od_sum_acc = eval_unseen_od(netAV, netVA, netOD, thresh, data, opt)
    eps = 1e-12
    H = 2*seen_acc*unseen_acc / (seen_acc + unseen_acc + eps)
    total_acc = float(seen_sum_acc + unseen_sum_acc) / float(seen_num + unseen_num)
    seen_od = float(seen_od_sum_acc) / float(seen_num)
    unseen_od = float(unseen_od_sum_acc) / float(unseen_num) 
    total_od_acc = float(unseen_od_sum_acc + seen_od_sum_acc) / float(seen_num + unseen_num)

    return seen_acc, unseen_acc, H, total_acc, seen_od, unseen_od, total_od_acc


