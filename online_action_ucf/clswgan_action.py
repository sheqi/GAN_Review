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
import model
import numpy as np

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

opt = parser.parse_args()
print(opt)

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
netD = model.MLP_CRITIC(opt)
netG = model.MLP_G(opt)
netDec = model.Dec(opt)

# Load nets if paths present
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
if opt.netDec != '':
    netDec.load_state_dict(torch.load(opt.netDec))

# print nets
print(netG)
print(netD)
print(netDec)

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
    netD.cuda()
    netG.cuda()
    netDec.cuda()
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
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerDec = optim.Adam(netDec.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        
# Start training
for epoch in range(opt.nepoch):
    # set to training mode
    netD.train()
    netG.train()
    netDec.train()

    mean_lossD, mean_lossG = 0, 0
    mean_lossR, mean_lossC = 0, 0 
    for i in range(0, data.ntrain, opt.batch_size):
        ############################
        # (1) Update D network: optimize WGAN-GP objective
        ###########################
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        for iter_d in range(opt.critic_iter):
            sample()
            netD.zero_grad()
            # train with realG, sample a mini-batch
            sparse_real = opt.resSize - input_res[1].gt(0).sum()
            input_resv = Variable(input_res) #I3D feature
            input_attv = Variable(input_att) #semantic attribute
            
            # Decoder training
            netDec.zero_grad()
            recons = netDec(input_resv) #reconstruct semantic attribute
            R_cost = recons_criterion(recons, input_attv)
            R_cost.backward()
            optimizerDec.step()
            
            # Discriminator training with real
            criticD_real = netD(input_resv, input_attv) #judge I3D for real feature
            criticD_real = criticD_real.mean()
            criticD_real.backward(mone)

            # train Discriminator with fakeG
            noise.normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev, input_attv) #generate I3D feature by G and semantic attribute
            fake_norm = fake.data[0].norm() 
            sparse_fake = fake.data[0].eq(0).sum()
            criticD_fake = netD(fake.detach(), input_attv) #judge the fake feature by D
            criticD_fake = criticD_fake.mean()
            criticD_fake.backward(one)

            # WGAN gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data, input_att)
            gradient_penalty.backward()

            Wasserstein_D = criticD_real - criticD_fake
            D_cost = criticD_fake - criticD_real + gradient_penalty
            optimizerD.step()
            mean_lossD += D_cost.item()

        ############################
        # (2) Update G network: optimize WGAN-GP objective
        ###########################
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = False # avoid computation

        netG.zero_grad()
        input_attv = Variable(input_att)
        noise.normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev, input_attv)
        criticG_fake = netD(fake, input_attv)
        criticG_fake = criticG_fake.mean()
        G_cost = -criticG_fake
        mean_lossG += G_cost.item()
        errG = G_cost
        
        ### cosine embedding loss for matching pairs
        temp_label = torch.ones(fake.shape[0])
        if opt.cuda:
            temp_label = temp_label.cuda()
        temp_label = Variable(temp_label)
        # fake and input_resv are matched already
        embed_match = emb_criterion(fake, input_resv, temp_label)

        ### cosine embedding loss for non-matching pairs
        # Randomly permute the labels and real input data
        if opt.cuda:
            rand_index = torch.randperm(input_label.shape[0]).cuda()
        else:
            rand_index = torch.randperm(input_label.shape[0])
        
        new_label = input_label[rand_index]
        new_feat = input_res[rand_index, :]
        z1 = input_label.cpu().numpy()
        z2 = new_label.cpu().numpy()
        temp_label = -1 * torch.ones(fake.shape[0])
        # Label correction for pairs that remain matched after random permutation
        if len(np.where(z1==z2)[0])>0:
            temp_label[torch.from_numpy(np.where(z1==z2)[0])] = 1
        if opt.cuda:
            temp_label = temp_label.cuda()
        embed_nonmatch = emb_criterion(fake, Variable(new_feat), Variable(temp_label))
        embed_err = embed_match + embed_nonmatch
        mean_lossC += embed_err.item()
        errG += opt.cosem_weight*embed_err

        ### Attribute reconstruction loss
        netDec.zero_grad()
        recons = netDec(fake)    
        R_cost = recons_criterion(recons, input_attv)
        mean_lossR += R_cost.item()
        errG += opt.recons_weight*R_cost
            
        errG.backward()
        optimizerG.step()
        optimizerDec.step()
        
    mean_lossG /=  data.ntrain / opt.batch_size 
    mean_lossD /=  opt.critic_iter * data.ntrain / opt.batch_size
    mean_lossC /=  data.ntrain / opt.batch_size
    mean_lossR /=  data.ntrain / opt.batch_size
    
    print('[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f' % (epoch, opt.nepoch, mean_lossD, mean_lossG, Wasserstein_D.item()))
       
    # set to evaluation mode
    netG.eval()
    netDec.eval()
    # Synthesize unseen class samples
    syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
    if opt.gzsl_od:
        # OD based GZSL
        seen_class = data.seenclasses.size(0)     
        clsu = classifier.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), data, data.unseenclasses.size(0), opt.cuda, _nepoch=25, _batch_size=opt.syn_num)
        clss = classifier.CLASSIFIER(data.train_feature, util.map_label(data.train_label,data.seenclasses), data, seen_class, opt.cuda, _nepoch=25, _batch_size=opt.syn_num, test_on_seen=True) 
        clsg = classifier_entropy.CLASSIFIER(data.train_feature, util.map_label(data.train_label,data.seenclasses), data, seen_class, syn_feature, syn_label, opt.cuda, clss, clsu, _batch_size=128)
        print('GZSL-OD: Acc seen=%.4f, Acc unseen=%.4f, h=%.4f' % (clsg.acc_seen, clsg.acc_unseen, clsg.H))
    elif opt.gzsl:
        # Generalized zero-shot learning    
        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)
        nclass = opt.nclass_all   
        clsg = classifier.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, _nepoch=25, _batch_size=opt.syn_num, generalized=True)
        print('GZSL: Acc seen=%.4f, Acc unseen=%.4f, h=%.4f' % (clsg.acc_seen, clsg.acc_unseen, clsg.H))
    else:
        # Zero-shot learning
        clsz = classifier.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), data, data.unseenclasses.size(0), opt.cuda, _nepoch=25, _batch_size=opt.syn_num)
        print('ZSL: Acc unseen=%.4f' % (clsz.acc))
     
