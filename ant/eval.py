import torch

from train import val_one_epoch

from test import test_on_benchmark, benchmark_smooth_grad, evalRob, generate_large_eps_adversarial_examples

import argparse
import os
from model import ant_model
from dataset import create_test_dataset, create_benchmark_dataset
from utils import TrainClock
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--val', action='store_true', help='weither to run validation')
    parser.add_argument('--bench', action='store_true',
                        help = 'weither to generate results on benchmark dataset')
    parser.add_argument('--smooth', action='store_true',
                        help = 'weither to generator smootGrad results on benchmark dataset')
    parser.add_argument('--gen', action = 'store_true',
                        help = 'weither to generate Large eps adversarial examples on benchmark data')

    parser.add_argument('--resume', type=str, default=None,
                        help='checkpoint path')

    args = parser.parse_args()

    clock = TrainClock()
    clock.epoch = 21
    net = ant_model()
    net.cuda()
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            check_point = torch.load(args.resume)
            net.load_state_dict(check_point['state_dict'])

            print('Modeled loaded from {} with metrics:'.format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

        base_path = os.path.split(args.resume)[0]
    else:
        base_path = './'

    net.eval()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    ds_val = create_test_dataset()
    ds_b = create_benchmark_dataset()

    if args.val:
        valacc, valcros, valtv, l1 = val_one_epoch(net, ds_val, criterion, clock)
        val_file = os.path.join(base_path, 'val.txt')
        with open(val_file, 'w') as f:
            print([valacc, valcros, valtv, l1], file=f)
    if args.bench:

        results_path = os.path.join(base_path, 'benchmark_results', str(clock.epoch))
        test_on_benchmark(net, ds_b, criterion, results_path)

    if args.smooth:

        results_path = os.path.join(base_path, 'smooth')
        benchmark_smooth_grad(net, ds_b, results_path)

    if args.gen:
        results_path = os.path.join(base_path, 'synimg40')
        generate_large_eps_adversarial_examples(net, ds_b, 40.0 / 255, 20, results_path)

    eval_res = evalRob(net, ds_val)
    print(eval_res)
    rob_file = os.path.join(base_path, 'rob.txt')
    with open(rob_file, 'w') as f:
        print(eval_res, file=f)

