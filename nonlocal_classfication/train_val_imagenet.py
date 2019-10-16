import os
import argparse
import time
import os.path as osp
import sys

import shutil
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torchvision

from tensorboardX import SummaryWriter
from torchvision import transforms
from termcolor import cprint
from lib import dataloader
from model import resnet_snl, preresnet_snl


from utils.loggers import Logger

# torch version
cprint('=> Torch Vresion: ' + torch.__version__, 'green')

# args
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--debug', '-d', dest='debug', action='store_true',
        help='enable debug mode')

parser.add_argument('--warmup', '-w', dest='warmup', action='store_true',
        help='using warmup strategy')

parser.add_argument('--print-freq', '-p', default=200, type=int, metavar='N',
        help='print frequency (default: 10)')

parser.add_argument('--data_dir', default='', type=str,
        help='the root dir of the dataset')

parser.add_argument('--dataset', default='imagenet', type=str,
        help='cub | cifar10 | cifar100 (default: cub)')

parser.add_argument('--valid', default=False, type=bool,
        help='just run validation')


parser.add_argument('--checkpoints', default='snl', type=str,
        help='the dir of checkpoints')

parser.add_argument('--num_gpu', default=1, type=int,
        help='number of gpu')


#######################################################################################
parser.add_argument('--nl-nums', default=1, type=int, metavar='N',
        help='number of the SNL block (default: 1)')

parser.add_argument('--nl-type', default='snl', type=str,
        help='choose snl | gsnl')

parser.add_argument('--stage-nums', default=1, type=int,
        help='the stage number of the nonlocal stage')

parser.add_argument('--div', default=2, type=float, metavar='LR',
        help='div for subplane of the nonlocal')

parser.add_argument('--nl_layer', default=['1'], type=list,
        help='add the nonlocal block in which layer of preresnet')

parser.add_argument('--relu', default=False, type=bool,
        help='whether uses the noliner transfer')

parser.add_argument('--pretrained', default=False, type=bool,
        help = 'using pretrained model')


#######################################################################################

parser.add_argument('--backbone', default ='resnet', type = str,
        help = 'the backbone model: resnet for cub, preresnet for cifar')

parser.add_argument('--arch', default='50', type=str,
        help='the depth of resnet or preresnet || 50, 101 for resnet || 20, 56 for preresnet')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR',
        help='initial learning rate (default: 0.01)')


########################################################################################


parser.add_argument('--isresume', default = False, type = bool,
        help = 'resume training')

parser.add_argument('--resumename', default ='', type = str,
        help = 'the resume path')

parser.add_argument('--check_path', default ='', type = str,
        help = 'the resume path')




best_prec1 = 0
best_prec5 = 0

args = parser.parse_args()

checkpoint_fold = os.path.join('save_model', args.dataset, args.arch, args.checkpoints)

if args.isresume:
    resume_path = os.path.join(checkpoint_fold, args.resumename)

if args.num_gpu == 1:
    torch.cuda.set_device(3)

np.random.seed(67)
torch.manual_seed(67)



def main():
    global args
    global best_prec1, best_prec5
    global checkpoint_fold, checkpoint_best


    sys.stdout = Logger(osp.join(checkpoint_fold, 'log_train.txt'))

    writer = SummaryWriter('save_model/{}/{}/{}'.format(args.dataset, args.arch, args.checkpoints))
    # simple args
    debug = args.debug
    if debug: cprint('=> WARN: Debug Mode', 'yellow')

    dataset = args.dataset


    if dataset == 'cub':
        num_classes = 200
        base_size = 512
        #batch_size = 60
        batch_size = 48
        crop_size = 448
        pool_size = 14
        args.warmup = True
        pretrain = args.pretrained
        args.backbone = 'resnet'
        args.arch = '50'

        epochs = 100
        eval_freq = 5
        args.lr = 0.01
        lr_drop_epoch_list = [31, 61, 81]
    elif dataset == 'cifar10':
        num_classes = 10
        base_size = 32
        batch_size = 128
        crop_size = 32
        args.warmup = True
        pretrain = args.pretrained
        #args.backbone = 'resnet'
        epochs = 300
        eval_freq = 5
        args.lr = 0.1
        lr_drop_epoch_list = [150, 250]

        if args.backbone == 'preresnet':
            pool_size = 8
        else:
            pool_size = 4

    elif dataset == 'cifar100':
        num_classes = 100
        base_size = 32
        batch_size = 128
        crop_size = 32
        args.warmup = True
        pretrain = args.pretrained
        epochs = 300
        eval_freq = 5
        args.lr = 0.1
        lr_drop_epoch_list = [150, 250]

        if args.backbone == 'preresnet':
            pool_size = 8
        else:
            pool_size = 4

    
    else: ##imagenet
        num_classes = 1000
        base_size = 256
        batch_size = 100
        crop_size = 224
        pool_size = 7
        args.warmup = True
        pretrain = args.pretrained
        args.backbone = 'resnet'

        epochs = 100
        eval_freq = 5
        args.lr = 0.01
        lr_drop_epoch_list = [31, 61, 81]
    
    workers = 4

    if debug:
        batch_size = 2
        workers = 0


    if base_size == 512 and \
        args.arch == '152':
        batch_size = 128
    drop_ratio = 0.1
    gpu_ids = [0,1]

    # args for the nl and cgnl block
    arch = args.arch
    nl_type  = args.nl_type # 'cgnl' | 'cgnlx' | 'nl'
    nl_nums  = args.nl_nums # 1: stage res4

    # warmup setting
    WARMUP_LRS = [args.lr * (drop_ratio**len(lr_drop_epoch_list)), args.lr]
    WARMUP_EPOCHS = 10

    # data loader
    if dataset == 'cub':
        data_root = os.path.join(args.data_dir, 'cub')
        imgs_fold = os.path.join(data_root, 'images')
        train_ann_file = os.path.join(data_root, 'cub_train.list')
        valid_ann_file = os.path.join(data_root, 'cub_val.list')
    elif dataset == 'imagenet':
        data_root = '/home/sheqi/lei/dataset/imagenet'
        imgs_fold = os.path.join(data_root)
        train_ann_file = os.path.join(data_root, 'imagenet_train.list')
        valid_ann_file = os.path.join(data_root, 'imagenet_val.list')
    elif dataset == 'cifar10':
        print("cifar10")
    elif dataset == 'cifar100':
        print("cifar100")
    else:
        raise NameError("WARN: The dataset '{}' is not supported yet.")

    if dataset == 'cub' or dataset == 'imagenet':
        train_dataset = dataloader.ImgLoader(
                root = imgs_fold,
                ann_file = train_ann_file,
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(
                        size=crop_size, scale=(0.08, 1.25)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
                    ]))

        val_dataset = dataloader.ImgLoader(
                root = imgs_fold,
                ann_file = valid_ann_file,
                transform = transforms.Compose([
                    transforms.Resize(base_size),
                    transforms.CenterCrop(crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
                    ]))

        train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size = batch_size,
                shuffle = True,
                num_workers = workers,
                pin_memory = True)

        val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size = batch_size,
                shuffle = False,
                num_workers = workers,
                pin_memory = True)

    elif dataset == 'cifar10':
        train_transform = transforms.Compose([
                    transforms.RandomCrop(crop_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465],
                        [0.2023, 0.1994, 0.2010])
                    ])
        val_transform = transforms.Compose([
                    #transforms.Resize(base_size),
                    #transforms.CenterCrop(crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465],
                        [0.2023, 0.1994, 0.2010])
                    ])
        trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=False , transform = train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
        testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=False , transform = val_transform)
        val_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    elif dataset == 'cifar100':
        train_transform = transforms.Compose([
                    transforms.RandomCrop(crop_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465],
                        [0.2023, 0.1994, 0.2010])
                    ])
        val_transform = transforms.Compose([
                    #transforms.Resize(base_size),
                    #transforms.CenterCrop(crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465],
                        [0.2023, 0.1994, 0.2010])
                    ])
        trainset = torchvision.datasets.CIFAR100(root=args.data_dir, train=True, download=False , transform = train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
        testset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=False , transform = val_transform)
        val_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    # build model

#####################################################
    if args.backbone == 'resnet':
        model = resnet_snl.model_hub(arch,
                                 pretrained=pretrain,
                                 nl_type=nl_type,
                                 nl_nums=nl_nums,
                                 stage_num=args.stage_nums,
                                 pool_size=pool_size, div=args.div, isrelu=args.relu)
    elif args.backbone == 'preresnet':
        model = preresnet_snl.model_hub(arch,
                                 pretrained=pretrain,
                                 nl_type=nl_type,
                                 nl_nums=nl_nums,
                                 stage_num=args.stage_nums,
                                 pool_size=pool_size, 
                                 div=args.div,
                                 nl_layer = args.nl_layer,
                                 relu = args.relu)
    else:
        raise KeyError("Unsupported nonlocal type: {}".format(nl_type))
####################################################


    # change the first conv for CIFAR
    if dataset == 'cifar10' or dataset == 'cifar100':
         model._modules['conv1'] = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
         model._modules['maxpool'] = torch.nn.Sequential()

    # change the fc layer
    if dataset != 'imagenet':
        model._modules['fc'] = torch.nn.Linear(in_features=2048,
                                           out_features=num_classes)
        torch.nn.init.kaiming_normal_(model._modules['fc'].weight,
                                  mode='fan_out', nonlinearity='relu')
    print(model)

    # parallel
    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids).cuda()
    else:
        model = model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # optimizer
    optimizer = torch.optim.SGD(
            model.parameters(),
            args.lr,
            momentum=0.9,
            weight_decay=1e-4)

    # cudnn
    cudnn.benchmark = True

    # warmup
    if args.warmup:
        epochs += WARMUP_EPOCHS
        lr_drop_epoch_list = list(
                np.array(lr_drop_epoch_list) + WARMUP_EPOCHS)
        cprint('=> WARN: warmup is used in the first {} epochs'.format(
            WARMUP_EPOCHS), 'yellow')


    start_epoch = 0
    if args.isresume:
        print('loading checkpoint {}'.format(resume_path))
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        print("epoches: {}, best_prec1: {}". format(start_epoch, best_prec1 ))


    # valid
    if args.valid:
        cprint('=> WARN: Validation Mode', 'yellow')
        print('start validation ...')
        print('=> loading state_dict from {}'.format(args.check_path))
        model.load_state_dict(
                torch.load(args.check_path)['state_dict'], strict=True)
        prec1, prec5 = validate(val_loader, model, criterion)
        print(' * Final Accuracy: Prec@1 {:.3f}, Prec@5 {:.3f}'.format(prec1, prec5))
        exit(0)

    # train
    print('start training ...')
    for epoch in range(start_epoch, epochs):
        current_lr = adjust_learning_rate(optimizer, drop_ratio, epoch, lr_drop_epoch_list,
                                          WARMUP_EPOCHS, WARMUP_LRS)
        # train one epoch
        cur_loss = train(train_loader, model, criterion, optimizer, epoch, epochs, current_lr)
        writer.add_scalar("Train Loss", cur_loss, epoch + 1)

        if nl_nums > 0:
            checkpoint_name = '{}-{}-r-{}-w-{}{}-block.pth.tar'.format(epoch, dataset, arch, nl_nums, nl_type)
        else:
            checkpoint_name = '{}-r-{}-{}-base.pth.tar'.format(dataset, arch, epoch)

        checkpoint_name = os.path.join(checkpoint_fold, checkpoint_name)

        if (epoch + 1) % eval_freq == 0:
            prec1, prec5 = validate(val_loader, model, criterion)
##########################################################
            writer.add_scalar("Top1", prec1, epoch + 1)
            writer.add_scalar("Top5", prec5, epoch + 1)
##########################################################
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            best_prec5 = max(prec5, best_prec5)
            print(' * Best accuracy: Prec@1 {:.3f}, Prec@5 {:.3f}'.format(best_prec1, best_prec5))
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, filename=checkpoint_name)


def train(train_loader, model, criterion, optimizer, epoch, epochs, current_lr):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        input = input.cuda(non_blocking=True)
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        # compute output
        if args.nl_type == 'nl_lim':
            att, output = model(input)
        else:
            output = model(input)

        loss = criterion(output, target)

        if args.nl_type == 'nl_lim':
            limit_loss = torch.norm(torch.matmul(att, att) - att)
            #print(limit_loss)
            loss = loss + 0.01 * limit_loss
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0:3d}/{1:3d}][{2:3d}/{3:3d}]\t'
                  'LR: {lr:.7f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, epochs, i, len(train_loader), 
                   lr=current_lr, batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    return loss


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            if args.nl_type == 'nl_lim':
                att, output = model(input)
            else:
                output = model(input)


            loss = criterion(output, target)
            if args.nl_type == 'nl_lim':
                limit_loss = torch.norm(torch.matmul(att, att) - att)
                loss = loss + 0.01 * limit_loss

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def adjust_learning_rate(optimizer, drop_ratio, epoch, lr_drop_epoch_list,
                         WARMUP_EPOCHS, WARMUP_LRS):
    if args.warmup and epoch < WARMUP_EPOCHS:
        # achieve the warmup lr
        lrs = np.linspace(WARMUP_LRS[0], WARMUP_LRS[1], num=WARMUP_EPOCHS)
        cprint('=> warmup lrs {}'.format(lrs), 'green')
        for param_group in optimizer.param_groups:
            param_group['lr'] = lrs[epoch]
        current_lr = lrs[epoch]
    else:
        decay = drop_ratio if epoch in lr_drop_epoch_list else 1.0
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * decay
        args.lr *= decay
        current_lr = args.lr
    return current_lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_fold, 'model_best.pth.tar'))

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
