from __future__ import print_function
import sys

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

#optional apex or distributed
from apex import amp
from apex.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp

import models
from models import MixNet
from flops_counter import get_model_complexity_info
from PIL import ImageFile
import tensorboardX
ImageFile.LOAD_TRUNCATED_IMAGES = True

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p

# for servers to immediately record the logs
def flush_print(func):
    def new_print(*args, **kwargs):
        func(*args, **kwargs)
        sys.stdout.flush()
    return new_print
print = flush_print(print)


# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=200, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--lr-decay', type=float, default=0.1, help='LR is multiplied by gamma on step schedule.')
parser.add_argument('--lr-mode', type=str, default='multistep', help='LR Schedule Mode.')
parser.add_argument('--lr-decay-period', type=int, default=0, help='Interval for periodic learning rate decays.')
parser.add_argument('--lr-decay-epoch', type=str, default="30,60,90", help='Epoches at which learning rate decays..')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--logdir', default='./logs/mixnet', type=str,
                    help='path to save log.')
# Architecture
parser.add_argument('--modelsize', '-ms', metavar='l', default='l', \
                    choices=['l', 'm', 's'], \
                    help = 'model_size affects the data augmentation, please choose:' + \
                           ' large or small ')

# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

#Device options
parser.add_argument('--local_rank', default=-1, type=int, 
                    help='node rank for distributed training')

args = parser.parse_args()


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target

# Use CUDA
use_cuda = torch.cuda.is_available()
nprocs = torch.cuda.device_count()

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
    cudnn.deterministic = True

best_acc = 0  # best test accuracy
num_itertions = 0 #total iterations

def main():
    dist.init_process_group(backend='nccl')
    
    train_batch = int(args.train_batch / nprocs)
    test_batch = int(args.test_batch / nprocs)
    
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_aug_scale = (0.08, 1.0) if args.modelsize == 'l' else (0.2, 1.0)

    train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224, scale = data_aug_scale),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=train_batch,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               shuffle=True,
                                               sampler=train_sampler)

    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=test_batch,
                                             num_workers=args.workers,
                                             pin_memory=True,
                                             sampler=val_sampler)


    # create model
   
    
    print("=> creating model MixNet.")
    model = MixNet(args.modelsize)
    flops, params = get_model_complexity_info(model, (224, 224), as_strings=False, print_per_layer_stat=False)
    print('Flops:  %.3fG' % (flops / 1e9))
    print('Params: %.2fM' % (params / 1e6))
    torch.cuda.set_device(args.local_rank)
    model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    model, optimizer = amp.initialize(model, optimizer)
    model = DistributedDataParallel(model)
    cudnn.benchmark = True

    lr_mode = args.lr_mode
    lr_decay_period = args.lr_decay_period
    lr_decay_epoch = args.lr_decay_epoch
    lr_decay = args.lr_decay
    if lr_decay_period > 0:
        lr_decay_epoch = list(range(lr_decay_period, num_epochs, lr_decay_period))
    else:
        lr_decay_epoch = [int(i) for i in lr_decay_epoch.split(",")]
    if (lr_mode == "step") and (lr_decay_period != 0):
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=lr_decay_period,
            gamma=lr_decay,
            last_epoch=-1)
    elif (lr_mode == "multistep") or ((lr_mode == "step") and (lr_decay_period == 0)):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=lr_decay_epoch,
            gamma=lr_decay,
            last_epoch=-1)
    elif lr_mode == "cosine":
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=args.epochs,
            last_epoch=(args.epochs - 1))

    
    # Resume
    title = 'ImageNet-MixNet'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..', args.resume)
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        # model may have more keys
        t = model.state_dict()
        c = checkpoint['state_dict']
        flag = True 
        for k in t:
            if k not in c:
                print('not in loading dict! fill it', k, t[k])
                c[k] = t[k]
                flag = False
        model.load_state_dict(c)
        if flag:
            print('optimizer load old state')
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('new optimizer !')
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Epoch', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(val_loader, model, criterion, start_epoch, use_cuda, local_rank)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return
    # TensorBoardX Logs
    train_writer = tensorboardX.SummaryWriter(args.logdir)
    # Train and val
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        lr_scheduler.step()

        print('\nEpoch: [%d | %d]' % (epoch + 1, args.epochs))

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda, local_rank)
        test_loss, test_acc = test(val_loader, model, criterion, epoch, use_cuda, local_rank)

        #add scalars
        train_writer.add_scalar('train_epoch_loss', train_loss, epoch)
        train_writer.add_scalar('train_epoch_acc', train_acc, epoch)
        train_writer.add_scalar('test_epoch_acc', test_acc, epoch)

        
        # append logger file
        logger.append([epoch, train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

    logger.close()
    train_writer.close()
    print('Best acc:')
    print(best_acc)

def train(train_loader, model, criterion, optimizer, epoch, use_cuda, local_rank):
    # switch to train mode
    model.train()
    torch.set_grad_enabled(True)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_loader))
    show_step = len(train_loader) // 10
    prefetcher = data_prefetcher(train_loader)
    inputs, targets = prefetcher.next()
    batch_idx = 0
    while inputs is not None:
        
        batch_size = inputs.size(0)
        if batch_size < args.train_batch:
            continue
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))

        torch.distributed.barrier()

        reduced_loss = reduce_mean(loss, nprocs)
        reduced_prec1 = reduce_mean(prec1, nprocs)
        reduced_prec5 = reduce_mean(prec5, nprocs)

        losses.update(reduced_loss.item(), inputs.size(0))
        top1.update(reduced_prec1.item(), inputs.size(0))
        top5.update(reduced_prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # add scalars
        train_writer.add_scalar('train_batch_loss', loss.avg, num_itertions)
        train_writer.add_scalar('train_batch_acc', top1.avg, num_itertions)
        num_itertions += 1

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        batch_idx += 1
        inputs, targets = prefetcher.next()
        if (batch_idx) % show_step == 0:
            print(bar.suffix)
        bar.next()
    bar.finish()
    
    return (losses.avg, top1.avg)

def test(val_loader, model, criterion, epoch, use_cuda, local_rank):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    torch.set_grad_enabled(False)

    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    prefetcher = data_prefetcher(val_loader)
    inputs, targets = prefetcher.next()
    batch_idx = 0
    while inputs is not None:
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))

        torch.distributed.barrier()
        
        reduced_loss = reduce_mean(loss, nprocs)
        reduced_prec1 = reduce_mean(prec1, nprocs)
        reduced_prec5 = reduce_mean(prec5, nprocs)

        losses.update(reduced_loss.item(), inputs.size(0))
        top1.update(reduced_prec1.item(), inputs.size(0))
        top5.update(reduced_prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(val_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
        batch_idx += 1
        inputs, targets = prefetcher.next()
    print(bar.suffix)
    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

if __name__ == '__main__':
    main()
