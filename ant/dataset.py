import torch

import os
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from PIL import Image
import json
def create_train_dataset(batch_size = 16, root = '../data'):

    transform_train = transforms.Compose(
        [transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()],
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
    )
    data_dir = '../data/hymenoptera_data/'

    trainset = ImageFolder(os.path.join(data_dir, 'train'), transform_train)
    #trainset = MNIST(root = root, train = True, download = True, transform = transform_train)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    return trainloader


def create_test_dataset(batch_size = 20, root = '../data'):
    transform_train = transforms.Compose(
        [transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()],
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
    )
    data_dir = '../data/hymenoptera_data/'

    testset = ImageFolder(os.path.join(data_dir, 'val'), transform_train)
    testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = True, num_workers = 4)

    return testloader


def produce_benchmark_datatsets(root = '../data', num = 30):

    transform_b = transforms.Compose(
        [transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip()],)
    data_dir = '../data/hymenoptera_data/'

    dataset = ImageFolder(os.path.join(data_dir, 'val'), transform_b)
    benchmark_datasets_dir = os.path.join(root, 'ant_benchmark_data')

    label_dic = {}
    if os.path.exists(benchmark_datasets_dir) is False:
        os.mkdir(benchmark_datasets_dir)
    for i in range(num):
        print('saving image {}'.format(i))
        img = transform_b(dataset[i+60][0])
        label = dataset[i+60][1]
        print(label)
        label_dic[i] = label
        img.save(os.path.join(benchmark_datasets_dir, '{}.png'.format(i)))

    label_file = os.path.join(benchmark_datasets_dir, 'label.json')
    with open(label_file, 'a') as f:
        json.dump(label_dic, f)
    '''
    for i, sample in enumerate(loader):

        imgs = sampel['image']

        print(imgs.size())
        img = imgs[0]


        print(torch.mean(img))

        print(img)

        break
    '''

class benchmark_dataset(Dataset):

    def __init__(self, benchmark_dir = '../data/ant_benchmark_data', num = 100, transform = None):

        self.benchmark_dir = benchmark_dir
        self.num  = 30
        self.transform = transform
        label_file = os.path.join(benchmark_dir, 'label.json')
        with open(label_file, 'r') as f:
            self.label_dic = json.load(f)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):

        label = self.label_dic[str(idx)]
        #print(label)
        image_name = os.path.join(self.benchmark_dir, '{}.png'.format(idx))

        img = Image.open(image_name)
        if callable(self.transform):
            img = self.transform(img)
            label = torch.tensor(label, dtype = torch.long)
        sample = {'data': img, 'label': label}
        return sample
def create_benchmark_dataset(batchsize = 100, bechmark_dir = '../data/ant_benchmark_data'):

    bechmark_transform =  transforms.Compose(
        [transforms.ToTensor()],
    )
    #print(callable(bechmark_transform))
    ds_bench = benchmark_dataset(benchmark_dir=bechmark_dir, num = 5, transform = bechmark_transform)
    dl_bench = torch.utils.data.DataLoader(ds_bench, batch_size=batchsize, shuffle=False, num_workers=4)

    return dl_bench


def test_bechmark_dataset():
    dl = create_benchmark_dataset()

    for i, data in enumerate(dl):
        print(len(data))


if __name__ == '__main__':
    #produce_benchmark_datatsets()
    test_bechmark_dataset()
