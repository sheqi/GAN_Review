import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
from torch.autograd import Variable
from collections import defaultdict

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i    

    return mapped_label

class Logger(object):
    def __init__(self, filename):
        self.filename = filename
        f = open(self.filename+'.log', "a")
        f.close()

    def write(self, message):
        f = open(self.filename+'.log', "a")
        f.write(message)  
        f.close()
        
class DATA_LOADER(object):
    def __init__(self, opt):
        assert opt.matdataset, 'Can load dataset in MATLAB format only'
        self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0

    def read_matdataset(self, opt):
        # load features and labels
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.action_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        # Load split details
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/split_" + str(opt.split) + "/" + opt.class_embedding + "_splits.mat")
        
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
        self.attribute = torch.from_numpy(matcontent['att'].T).float()
        
        if not opt.validation:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()
                
                _train_feature = scaler.fit_transform(feature[trainval_loc])
                _test_seen_feature = scaler.transform(feature[test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1/mx)
                self.train_label = torch.from_numpy(label[trainval_loc]).long() 
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1/mx)
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long() 
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float() 
                self.test_seen_feature.mul_(1/mx)
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()


            else:
                self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
                self.train_label = torch.from_numpy(label[trainval_loc]).long() 
                self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long() 
                self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float() 
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
            ########################################################
            if self.train_feature.size(1) > opt.resSize:
                self.train_feature= self.train_feature[:, 0:opt.resSize]
            if self.test_unseen_feature.size(1) > opt.resSize:
                self.test_unseen_feature= self.test_unseen_feature[:, 0:opt.resSize]
            if self.test_seen_feature.size(1) > opt.resSize:
                self.test_seen_feature = self.test_seen_feature[:, 0:opt.resSize]
            ####triplet sampler####
            if opt.triplet:
                self.index_dic = {}
                self.label_dic = []
                for index, lab in enumerate(self.train_label):
                    lab = np.int(lab.numpy())
                    if lab not in self.label_dic:
                        self.label_dic.append(lab)
                        self.index_dic[lab] = []
                    self.index_dic[lab].append(index)
            #########################################################



        else:
            self.train_feature = torch.from_numpy(feature[train_loc]).float()
            self.train_label = torch.from_numpy(label[train_loc]).long()
            self.test_unseen_feature = torch.from_numpy(feature[val_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(label[val_unseen_loc]).long() 
            ########################################################
            if self.train_feature.size(1) > opt.resSize:
                self.train_feature= self.train_feature[:, 0:opt.resSize]
            if self.test_unseen_feature.size(1) > opt.resSize:
                self.test_unseen_feature= self.test_unseen_feature[:, 0:opt.resSize]
    
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))

        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()
        self.train_mapped_label = map_label(self.train_label, self.seenclasses) 

    # Random batch sampling
    def next_batch_triplet(self, num_class, num_instance):
        #print(self.train_feature.size())
        #print(self.train_label.size())
        #print(self.attribute.size())
        cls_idx = torch.randperm(len(self.label_dic))[:num_class]
        batch_feature = torch.zeros((num_class * num_instance, self.train_feature.size(1))).float()
        batch_label = torch.zeros((num_class * num_instance)).long()
        batch_att = torch.zeros((num_class * num_instance, self.attribute.size(1))).float()
        count = 0
        for lab in cls_idx:
            lab = self.label_dic[lab]
            r_idx = torch.randperm(len(self.index_dic[lab]))[:num_instance]
            for j in r_idx:
                idx = self.index_dic[lab][j]
                batch_feature[count] = self.train_feature[idx]
                batch_label[count] = self.train_label[idx]
                batch_att[count] = self.attribute[self.train_label[idx]]
                count = count + 1
        return batch_feature, batch_label, batch_att

    def next_batch(self, batch_size):
        idx = torch.randperm(self.ntrain)[:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att

    def next_batch_one_class(self, batch_size):
        if self.index_in_epoch == self.ntrain_class:
            self.index_in_epoch = 0 
            perm = torch.randperm(self.ntrain_class)
            self.train_class[perm] = self.train_class[perm]

        iclass = self.train_class[self.index_in_epoch]
        idx = self.train_label.eq(iclass).nonzero().squeeze()
        perm = torch.randperm(idx.size(0))
        idx = idx[perm]
        iclass_feature = self.train_feature[idx]
        iclass_label = self.train_label[idx]
        self.index_in_epoch += 1
        return iclass_feature[:batch_size], iclass_label[:batch_size], self.attribute[iclass_label[:batch_size]] 
    
            