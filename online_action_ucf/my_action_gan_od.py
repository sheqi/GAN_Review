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
from model import MLP_G
from eval import *
import selector

parser = argparse.ArgumentParser("GZSL Action")
parser.add_argument('--dataset', default='hmdb51', help='Dataset name')
parser.add_argument('--dataroot', default='data_action/', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--action_embedding', default='i3d')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--split', type=int, default=1)
parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')
parser.add_argument('--gzsl_od', action='store_true', default=False, help='enable out-of-distribution based generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true', default=False, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--resSize', type=int, default=4096, help='size of visual features')
parser.add_argument('--attSize', type=int, default=300, help='size of semantic features')
parser.add_argument('--nz', type=int, default=256, help='size of the latent z vector')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=4096, help='size of the hidden units in discriminator')
parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--cosem_weight', type=float, default=0.1, help='weight of the cos embed loss')
parser.add_argument('--recons_weight', type=float, default=0.01, help='recons_weight for decoder')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netDec', default='', help="path to netDec (to continue training)")
parser.add_argument('--netG_name', default='')
parser.add_argument('--netD_name', default='')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')

parser.add_argument('--use_gan', type=bool, default=False, help='use gan to generate')
parser.add_argument('--use_od', type=bool, default=False, help='use gan to generate')
parser.add_argument('--use_train', type=bool, default=False, help='use gan to generate')

opt = parser.parse_args()
print(opt)

def generate_syn_feature(netG, classes, attribute, num):
    # generate num synthetic samples for each class in classes
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass*num) 
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)

    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()


    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        with torch.no_grad():
            output = netG(Variable(syn_noise), Variable(syn_att))
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass)
                
    return syn_feature, syn_label


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# load data
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

# initialize generator, discriminator and decoder
netVA = model.Encode_Vis_Att(opt)
netAV = model.Decode_Att_Vis(opt)
# print nets
print(netVA)
print(netAV)

##############################
netG = MLP_G(opt).cuda()
netG.load_state_dict(torch.load("netG.tar")['state_dict'].state_dict())
syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
seen_class = data.seenclasses.size(0)
if opt.use_od:
    netOD = selector.ODDetector(data.train_feature.size(1), 512, seen_class).cuda()
    netOD.load_state_dict(torch.load("netOD.tar")['state_dict'])
    od_seen_thresh = torch.load("netOD.tar")['seen_thresh']
    od_unseen_thresh = torch.load("netOD.tar")['unseen_thresh']
    od_thresh = 0.1
    #od_thresh = (od_unseen_thresh - od_seen_thresh) / 2
    #print(od_thresh)

    #netOD = selector.CLASSIFIER(data.train_feature, util.map_label(data.train_label, data.seenclasses), data, seen_class, syn_feature, syn_label, opt.cuda, _batch_size=128)
    #od_seen_thresh = netOD.thresh
    #od_unseen_thresh = netOD.unseen_thresh
    #netOD = netOD.model
    #torch.save({'state_dict':netOD.state_dict(), 'seen_thresh':od_seen_thresh, 'unseen_thresh':od_unseen_thresh}, 'netOD.tar')

opt.num_seen = len(data.train_feature)
if opt.use_gan:
    data.train_feature = torch.cat((data.train_feature, syn_feature), dim = 0)
    data.train_label = torch.cat((data.train_label, syn_label), dim = 0)

##############################

emb_criterion = nn.CosineEmbeddingLoss(margin=0)
recons_criterion = nn.MSELoss()
# recons_criterion = nn.L1Loss()  # L1 loss 

input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
input_label = torch.LongTensor(opt.batch_size)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
one = torch.FloatTensor([1])
mone = one * -1

if opt.cuda:
    netVA.cuda()
    netAV.cuda()
    input_res, input_label = input_res.cuda(), input_label.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    one = one.cuda()
    mone = mone.cuda()
    emb_criterion.cuda()
    recons_criterion.cuda()
    
def sample():
    # Sample a batch
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    if batch_feature.size(1) > opt.resSize:
        batch_feature = batch_feature[:, 0:opt.resSize]
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))


def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    # Gradient penalty of WGAN
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, Variable(input_att))

    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty


# setup optimizer
optimizerVA = optim.Adam(netVA.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerAV = optim.Adam(netAV.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        
# Start training
for epoch in range(opt.nepoch):
    # set to training mode
    netVA.train()
    netAV.train()

    mean_lossAV, mean_lossVA = 0, 0
    mean_lossM = 0 
    for i in range(0, data.ntrain, opt.batch_size):
        ############################
        # (1) Update Vis to Att
        ###########################
        for p in netVA.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update
        for p in netAV.parameters(): # reset requires_grad
            p.requires_grad = False # they are set to False below in netG update

        for iter_d in range(opt.critic_iter):
            sample()
            netVA.zero_grad()
            # train with realG, sample a mini-batch
            input_resv = Variable(input_res) #I3D feature
            input_attv = Variable(input_att) #semantic attribute
            
            recons_a, mix_f_a = netVA(input_resv) #reconstruct semantic attribute
            VA_cost = recons_criterion(recons_a, input_attv)

            VA_cost.backward()
            optimizerVA.step()
            mean_lossVA += VA_cost.item()

        
        ###########################
        # (2) Update Att to Vis
        ###########################
        for p in netVA.parameters(): # reset requires_grad
            p.requires_grad = False # avoid computation
        for p in netAV.parameters(): # reset requires_grad
            p.requires_grad = True # avoid computation
        for iter_d in range(opt.critic_iter):
            sample()
            netAV.zero_grad()
            # train with realG, sample a mini-batch
            input_resv = Variable(input_res) #I3D feature
            input_attv = Variable(input_att) #semantic attribute
            
            # Decoder training
            recons_v, mix_f_v = netAV(input_attv) #reconstruct semantic attribute
            AV_cost = recons_criterion(recons_v, input_resv)
            AV_cost.backward()
            optimizerAV.step()
            mean_lossAV += AV_cost.item()
        
        ###########################
        # (2) Update Mix Domain
        ###########################

        for p in netVA.parameters(): # reset requires_grad
            p.requires_grad = True # avoid computation
        for p in netAV.parameters(): # reset requires_grad
            p.requires_grad = True # avoid computation

        for iter_d in range(opt.critic_iter):
            sample()
            netAV.zero_grad()
            netVA.zero_grad()
            # train with realG, sample a mini-batch
            input_resv = Variable(input_res) #I3D feature
            input_attv = Variable(input_att) #semantic attribute
            
            # Decoder training
            recons_v, mix_f_v = netAV(input_attv) #reconstruct semantic attribute
            recons_a, mix_f_a = netVA(input_resv) #reconstruct semantic attribute
            M_cost = recons_criterion(mix_f_v, mix_f_a)

            M_cost.backward()
            optimizerAV.step()
            optimizerVA.step()
            mean_lossM += M_cost.item()


        
    mean_lossVA /=  opt.critic_iter * data.ntrain / opt.batch_size 
    mean_lossAV /=  opt.critic_iter * data.ntrain / opt.batch_size
    mean_lossM /=  data.ntrain / opt.batch_size
    
    print('[%d/%d] Loss_VA: %.4f Loss_AV: %.4f, Loss_M: %.4f' % (epoch, opt.nepoch, mean_lossVA, mean_lossAV, mean_lossM))


    ##########eval_unseen#########

    if opt.use_od:
        seen_acc, unseen_acc, H, total_acc, seen_od, unseen_od, total_od_acc = eval_od(netAV, netVA, netOD, od_thresh, data, opt)
        print('Test with OD [%d/%d]: Unseen: %.4f, Seen: %4f, H: %4f, Total: %4f' % (epoch, opt.nepoch, unseen_acc, seen_acc, H, total_acc))
        print('Test with OD [%d/%d]: Unseen_od: %.4f, Seen_od: %4f, Total_od: %4f' % (epoch, opt.nepoch, unseen_od, seen_od, total_od_acc))
    else:
        total_acc, seen_acc, unseen_acc = eval_gzsl(netAV, netVA, data, opt)
        eps = 1e-12
        H = 2*seen_acc*unseen_acc / (seen_acc + unseen_acc + eps)
        print('Test [%d/%d]: Unseen: %.4f, Seen: %4f, H: %4f, Mix: %4f' % (epoch, opt.nepoch, unseen_acc, seen_acc, H, total_acc))


           
