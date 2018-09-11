import torch
from  tqdm import tqdm
import time

from utils import TvLoss, AvgMeter, torch_accuracy
def train_one_epoch(net, optimizer, batch_generator, CrossEntropyCriterion, clock, tv_loss_weight = 1.0):
    '''

    :param net:   network
    :param optimizer:
    :param batch_generator:   pytorch dataloader or other generator
    :param CrossEntropyCriterion:  Used for calculating CrossEntropy loss
    :param clock: TrainClock from utils
    :return:
    '''

    Acc = AvgMeter()
    CrossLoss = AvgMeter()
    GradTvLoss = AvgMeter()

    net.train()
    clock.tock()

    pbar = tqdm(batch_generator)


    for (data, label) in pbar:

        clock.tick()

        data = data.cuda()
        label = label.cuda()

        #print(data.requires_grad)
        data.requires_grad = True

        pred = net(data)

        cross_entropy_loss = CrossEntropyCriterion(pred, label)

        grad_map = torch.autograd.grad(cross_entropy_loss, data, create_graph = True, only_inputs = False)[0]


        #print(grad_map.size())
        grad_tv_loss = TvLoss(grad_map)

        loss = torch.add(cross_entropy_loss, grad_tv_loss * tv_loss_weight)

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        acc = torch_accuracy(pred, label, topk = (1, ))[0].item()

        Acc.update(acc)
        CrossLoss.update(cross_entropy_loss.item())
        GradTvLoss.update(grad_tv_loss.item())

        pbar.set_description("Training Epoch: {}".format(clock.epoch))

        pbar.set_postfix({'Acc': '{:.3f}'.format(Acc.mean),
                          'CrossLoss': "{:.2f}".format(CrossLoss.mean),
                         'GradTvLoss': '{:.3f}'.format(GradTvLoss.mean)})

    return Acc.mean, CrossLoss.mean, GradTvLoss.mean



def val_one_epoch(net, batch_generator, CrossEntropyCriterion, clock):
    '''

    :param net:   network
    :param optimizer:
    :param batch_generator:   pytorch dataloader or other generator
    :param CrossEntropyCriterion:  Used for calculating CrossEntropy loss
    :param clock: TrainClock from utils
    :return:
    '''

    Acc = AvgMeter()
    CrossLoss = AvgMeter()
    GradTvLoss = AvgMeter()
    L1 = AvgMeter()

    net.eval()
    pbar = tqdm(batch_generator)

    for (data, label) in pbar:

        data = data.cuda()
        label = label.cuda()

        data.requires_grad = True

        pred = net(data)

        cross_entropy_loss = CrossEntropyCriterion(pred, label)

        grad_map = torch.autograd.grad(cross_entropy_loss, data, create_graph = True, only_inputs = False)[0]

        l1_loss = torch.sum(torch.mean(torch.abs(grad_map), dim = 0))
        grad_tv_loss = TvLoss(grad_map)

        #loss = torch.add(cross_entropy_loss, grad_tv_loss * tv_loss_weight)

        #optimizer.zero_grad()

        acc = torch_accuracy(pred, label, topk = (1, ))[0].item()

        Acc.update(acc)
        CrossLoss.update(cross_entropy_loss.item())
        GradTvLoss.update(grad_tv_loss.item())
        L1.update(l1_loss.item())

        pbar.set_description("Validation Epoch: {}".format(clock.epoch))

        pbar.set_postfix({'Acc': '{:.3f}'.format(Acc.mean),
                          'CrossLoss': "{:.2f}".format(CrossLoss.mean),
                         'GradTvLoss': '{:.3f}'.format(GradTvLoss.mean),
                          'L1': '{:.3f}'.format(L1.mean)})

    return Acc.mean, CrossLoss.mean, GradTvLoss.mean, L1.mean


