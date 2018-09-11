import torch
import torch.nn as nn

def GetSmoothGrad(net, img, label, stdev_spread = 0.15, num = 32, square = False):
    '''

    :param net: pytorch network
    :param img: a single image
    :stdev_spread: Amount of noise to add to the input, as fraction of the
                    total spread (x_max - x_min). Defaults to 15%.
    :num:  Number of samples used
    :square: If True, computes the sum of squares of gradients instead of
                 just the sum. Defaults to False.
    :return: smooth grad
    '''

    size = list(img.size())
    size = [num, ] + size
    img = img.expand((size))
    #print('daw', label)
    label = label.expand((num))
    #print(img.size())
    #print(label.size())
    net.eval()
    img = img.cuda()
    label = label.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    img.requires_grad = True

    stdev = (torch.max(img) - torch.min(img)) * stdev_spread
    stdev = stdev.expand_as(img)
    mean = torch.zeros_like(img)
    noises = torch.normal(mean, stdev)
    img = img + noises

    pred = net(img)
    loss = criterion(pred, label)
    grad_maps = torch.autograd.grad(loss, img, create_graph=True, only_inputs=False)[0]

    grad_map = torch.sum(grad_maps, dim = 0)

    return grad_map

