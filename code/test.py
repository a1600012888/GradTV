import torch
from dataset import create_benchmark_dataset
from tqdm import tqdm
import copy
import os
import numpy as np

from PIL import  Image
from cv2 import imwrite

def clip_gradmap(gradmap):
    '''

    :param gradmap: should be a single image!! shape of (c, h, w)
    :return: a image of the same shape.   values of pixels: (0, 255) unint8
    '''
    if isinstance(gradmap, torch.Tensor):
        gradmap = deepcopy(gradmap.numpy())

    assert len(gradmap.shape) == 3 or len(gradmap.shape == 2),  gradmap.shape

    if gradmap.shape[0] == 1:
        gradmap = gradmap[0]

    max_value = np.percentile(gradmap, 99)

    img = np.clip( (gradmap + max_value) / (2 * max_value), 0, 1) * 255

    img = img.astype(np.uint8)

    return img

def clip_and_saveimgs(grad_maps, start_i = 0, save_dir = './benchmark_results'):
    grad_maps = grad_maps.detach().cpu()
    grad_maps = grad_maps.numpy()

    start_i = start_i * grad_maps.shape[0]
    for i, grad_map in enumerate(grad_maps):
        image_name = os.path.join(save_dir, '{}-result.png'.format(i + start_i))

        img = clip_gradmap(grad_map)
        #img = Image.fromarray(img)

        #img.save(image_name)
        imwrite(image_name, img)
        #print('saving image {}'.format(image_name))

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

        clip_and_saveimgs(grad_map, i, save_dir)