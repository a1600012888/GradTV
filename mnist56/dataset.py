import torch

import os
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from PIL import Image
import json
def create_train_mnist_dataset(batch_size = 128, root = '../data'):

    transform_train = transforms.Compose(
        [transforms.Resize(56), transforms.ToTensor()],
    )
    trainset = MNIST(root = root, train = True, download = True, transform = transform_train)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    return trainloader


def create_test_mnist_dataset(batch_size = 128, root = '../data'):

    transform_test = transforms.Compose(
        [transforms.Resize(56), transforms.ToTensor()],
    )
    testset = MNIST(root = root, train = False, download = True, transform = transform_test)

    testloader = torch.utils.data.DataLoader(testset, batch_size = 100, shuffle = False, num_workers = 2)

    return testloader


def produce_benchmark_datatsets(root = '../data', num = 100):

    dataset = MNIST(root=root , train = False, download=True)

    #loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=2)

    benchmark_datasets_dir = os.path.join(root, '2benchmark_data')

    label_dic = {}
    if os.path.exists(benchmark_datasets_dir) is False:
        os.mkdir(benchmark_datasets_dir)
    for i in range(num):
        print('saving image {}'.format(i))
        img = dataset[i][0].resize((56, 56))
        label = dataset[i][1].item()
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

    def __init__(self, benchmark_dir = '../data/2benchmark_data', num = 100, transform = None):

        self.benchmark_dir = benchmark_dir
        self.num  = 100
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
def create_benchmark_dataset(batchsize = 100, bechmark_dir = '../data/2benchmark_data'):

    bechmark_transform =  transforms.Compose(
        [transforms.ToTensor()],
    )
    #print(callable(bechmark_transform))
    ds_bench = benchmark_dataset(benchmark_dir=bechmark_dir, num = 100, transform = bechmark_transform)
    dl_bench = torch.utils.data.DataLoader(ds_bench, batch_size=batchsize, shuffle=False, num_workers=2)

    return dl_bench


def test_bechmark_dataset():
    dl = create_benchmark_dataset()

    for i, data in enumerate(dl):
        print(len(data))


if __name__ == '__main__':
    produce_benchmark_datatsets()
    #test_bechmark_dataset()
