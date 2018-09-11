import torch
import torch.nn as nn
import torch.optim as optim
from dataset import create_train_mnist_dataset, create_test_mnist_dataset

from utils import TrainClock, save_args, MultiStageLearningRatePolicy

from train import train_one_epoch, val_one_epoch
from test import test_on_benchmark
from model import MnistModel_v1

import argparse
import os
parser = argparse.ArgumentParser()


parser.add_argument('--weight_decay', default=1e-4, type = float, help='weight decay (default: 5e-4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (if has resume, this is not needed')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--exp', default='baseline', type = str, help = 'the name of this experiment')


parser.add_argument('--tv', default=1.0, type = float, help = 'The weight of TV loss term')

args = parser.parse_args()


log_dir = os.path.join('../logs', args.exp)
exp_dir = os.path.join('../exps', args.exp)
benchmark_result_dir = os.path.join(exp_dir, 'benchmark_results')
if not os.path.exists(exp_dir):
    os.mkdir(exp_dir)
if not os.path.exists(benchmark_result_dir):
    os.mkdir(benchmark_result_dir)

save_args(args, exp_dir)


clock = TrainClock()

learning_rate_policy = [[10, 0.01],
                        [5, 0.001],
                        [5, 0.0001]
                        ]

get_learing_rate = MultiStageLearningRatePolicy(learning_rate_policy)

def adjust_learning_rate(optimzier, epoch):
    #global get_lea
    lr = get_learing_rate(epoch)
    for param_group in optimizer.param_groups:

        param_group['lr'] = lr


torch.backends.cudnn.benchmark = True
ds_train = create_train_mnist_dataset()

ds_val = create_test_mnist_dataset()


net = MnistModel_v1()
net.cuda()

CrossEntropyCriterion = nn.CrossEntropyLoss().cuda()

optimizer = optim.SGD(net.parameters(), lr = get_learing_rate(0), momentum=0.9, weight_decay=args.weight_decay)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        check_point = torch.load(args.resume)
        args.start_epoch = check_point['epoch']
        net.load_state_dict(check_point['state_dict'])

        print('Modeled loaded from {} with metrics:'.format(args.resume))
        print(results)
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

clock.epoch = args.start_epoch

while True:
    if clock.epoch > args.epochs:
        break

    adjust_learning_rate(optimizer, clock.epoch)


    acc, crossLoss, TVLoss = train_one_epoch(net, optimizer, ds_train, CrossEntropyCriterion, clock, args.tv)

    valacc, valcrossLoss, valTVLoss = val_one_epoch(net, ds_val, CrossEntropyCriterion, clock, args.tv)

    result_dir = os.path.join(benchmark_result_dir, '{}'.format(clock.epoch))

    test_on_benchmark(net, None, CrossEntropyCriterion, result_dir)