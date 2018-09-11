import torch
from typing import Tuple, List, Dict
import os
import json
import math
def TvLoss(img):

    tv_loss = torch.sum(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])) \
              + torch.sum(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    tv_loss = tv_loss / img.size(0)

    return tv_loss



class TrainClock(object):
    def __init__(self):
        self.epoch = 0
        self.minibatch = 0
        self.step = 0

    def tick(self):
        self.minibatch += 1
        self.step += 1

    def tock(self):
        self.epoch += 1

        self.minibatch = 0

class AvgMeter(object):
    name = 'No name'
    def __init__(self, name = 'No name'):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = 0
        self.mean = 0
        self.num = 0
        self.now = 0
    def update(self, mean_var, count = 1):
        if math.isnan(mean_var):
            mean_var = 1e6
            print('Avgmeter getting Nan!')
        self.now = mean_var
        self.num += count

        self.sum += mean_var * count
        self.mean = float(self.sum) / self.num

def torch_accuracy(output, target, topk = (1, )):
    '''
    param output, target: should be torch Variable
    '''
    #assert isinstance(output, torch.cuda.Tensor), 'expecting Torch Tensor'
    #assert isinstance(target, torch.Tensor), 'expecting Torch Tensor'
    #print(type(output))

    topn = max(topk)
    batch_size = output.size(0)

    _, pred = output.topk(topn, 1, True, True)
    pred = pred.t()

    is_correct = pred.eq(target.view(1, -1).expand_as(pred))

    ans = []
    for i in topk:
        is_correct_i = is_correct[:i].view(-1).float().sum(0, keepdim = True)
        ans.append(is_correct_i.mul_(100.0 / batch_size))

    return ans


def save_args(args, save_dir = None):
    if save_dir == None:
        param_path = os.path.join(args.resume, "params.json")
    else:
        param_path = os.path.join(save_dir, 'params.json')

    #logger.info("[*] MODEL dir: %s" % args.resume)
    #logger.info("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(args.__dict__, fp, indent=4, sort_keys=True)



def dump_dict(json_file_path, dic):
    with open(json_file_path, 'a') as f:
        f.write('\n')
        json.dump(dic, f)

class MultiStageLearningRatePolicy(object):
    '''
    '''

    _stages = None
    def __init__(self, stages:List[Tuple[int, float]]):

        assert(len(stages) >= 1)
        self._stages = stages


    def __call__(self, cur_ep:int) -> float:
        e = 0
        for pair in self._stages:
            e += pair[0]
            if cur_ep < e:
                return pair[1]
      #  return pair[-1][1]
        return pair[-1]




