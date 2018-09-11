import torch
from dataset import create_benchmark_dataset
from tqdm import tqdm
import copy
import os
import numpy as np
from utils import AvgMeter
from PIL import  Image
from cv2 import imwrite
from attack import IPGD
from smoothgrad import GetSmoothGrad
import argparse
def clip_gradmap(gradmap, transpose = False, togray = True):
    '''

    :param gradmap: should be a single image!! shape of (c, h, w)
    :return: a image of the same shape.   values of pixels: (0, 255) unint8
    '''
    if isinstance(gradmap, torch.Tensor):
        gradmap = deepcopy(gradmap.numpy())

    assert len(gradmap.shape) == 3 or len(gradmap.shape == 2),  gradmap.shape

    gradmap = np.abs(gradmap)

    if gradmap.shape[0] == 1:
        gradmap = gradmap[0]
    if togray and gradmap.shape[0] == 3:
        gradmap = np.sum(gradmap, axis = 0)

    max_value = np.percentile(gradmap, 99)
    img = np.clip( (gradmap) / (  max_value), 0, 1) * 255

    img = img.astype(np.uint8)
    if transpose:
        img = np.transpose(img, (1, 2, 0))

    return img

def clip_and_save_single_img(grad_map, i = 0, save_dir = './benchmark_smooth'):
    grad_map = grad_map.detach().cpu()
    grad_map = grad_map.numpy()

    image_name = os.path.join(save_dir, '{}-smooth.png'.format(i))

    img = clip_gradmap(grad_map)
    imwrite(image_name, img)


def clip_and_save_batched_imgs(grad_maps, start_i = 0, save_dir = './benchmark_results'):
    grad_maps = grad_maps.detach().cpu()
    grad_maps = grad_maps.numpy()

    start_i = start_i * grad_maps.shape[0]
    for i, grad_map in enumerate(grad_maps):
        image_name = os.path.join(save_dir, '{}-result.png'.format(i + start_i))

        img = clip_gradmap(grad_map)
        imwrite(image_name, img)



def test_on_benchmark(net, batch_generator = None, CrossEntropyCriterion = None, save_dir = './benchmark_results'):

    if batch_generator is None:
        batch_generator = create_benchmark_dataset()
    if CrossEntropyCriterion is None:
        CrossEntropyCriterion = torch.nn.CrossEntropyLoss().cuda()
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    pbar = tqdm(enumerate(batch_generator))
    net.eval()

    for i, sample in pbar:

        data = sample['data'].cuda()
        label = sample['label'].cuda()
        data.requires_grad = True
        pred = net(data)

        cross_entropy_loss = CrossEntropyCriterion(pred, label)

        grad_map = torch.autograd.grad(cross_entropy_loss, data, create_graph=True, only_inputs=False)[0]

        clip_and_save_batched_imgs(grad_map, i, save_dir)


def benchmark_smooth_grad(net, batch_generator, save_dir = './benchmark_smooth',
                          stdev_spread = 0.15, num = 32, square = False):

    if batch_generator is None:
        batch_generator = create_benchmark_dataset()

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    pbar = tqdm(enumerate(batch_generator))

    net.eval()
    for i, sample in pbar:

        data = sample['data'].cuda()
        label = sample['label'].cuda()

        for j in range(data.size(0)):
            img = data[j]
            l = label[j]
            grad_map = GetSmoothGrad(net, img, l, stdev_spread, num, square)
            clip_and_save_single_img(grad_map, i * data.size(0) + j, save_dir)


def evalRob(net, batch_generator):

    net.eval()
    #epss = [0,  4.0/255.0, 6.0/255.0, 8.0/255.0, 10.0/255.0, 12.0/255.0, 16.0/255.0]
    epss = [0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0]
    iters = [1, 20, 20, 20, 20, 20, 20]
    acc = []

    for e, i in zip(epss, iters):
        acc.append(evalGivenEps(net, batch_generator, e, i))
    return acc

def evalGivenEps(net, batch_generator, eps, nb_iter):
    defense_accs = AvgMeter()
    net.eval()
    attack = IPGD(eps, eps / 2.0, nb_iter)

    pbar = tqdm(batch_generator)

    for data, label in pbar:
        data = data.cuda()
        label = label.cuda()

        defense_accs.update(attack.get_batch_accuracy(net, data, label))
        pbar.set_description('Evulating Roboustness')

    return defense_accs.mean


def generate_large_eps_adversarial_examples(net, batch_generator, eps, nb_iter, save_dir):
    attack = IPGD(eps, eps/4.0, nb_iter)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    pbar = tqdm(enumerate(batch_generator))
    net.eval()

    for i, sample in pbar:
        data = sample['data'].cuda()
        label = sample['label'].cuda()

        imgs = attack.attack(net, data, label)

        imgs = imgs.detach().cpu().numpy() * 255

        imgs = imgs.astype(np.uint8)

        for j, img in enumerate(imgs):

            index = j + i * data.size(0)

            save_path = os.path.join(save_dir, '{}.png'.format(index))

            img = np.transpose(img, (1, 2, 0))

            imwrite(save_path, img)

if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    pass


