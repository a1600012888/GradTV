import torch


from torchvision.datasets import MNIST



def create_train_mnist_dataset(batch_size = 128, root = '../data'):

    trainset = MNIST(root = root, train = True, download = True)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    return trainloader


def create_test_mnist_dataset(batch_size = 128, root = '../data'):

    testset = torchvision.datasets.CIFAR10(root = root, train = False, download = True)

    testloader = torch.utils.data.DataLoader(testset, batch_size = 100, shuffle = False, num_workers = 2)

    return testloader