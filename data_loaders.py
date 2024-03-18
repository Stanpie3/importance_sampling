

import torch
from torchvision.datasets import CIFAR10, MNIST, CIFAR100, VisionDataset
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler
from torch.utils.data import DataLoader
from PIL import Image

normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.247, 0.2435, 0.2616],
)



def addIndexes(data_set: type[VisionDataset]) :
    class IndexWrrapper(data_set):
        def __getitem__(self, index: int) :
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)
            return index, img, target
    
    return IndexWrrapper



def samplers(n, split_shuffle=True, val_size=0.1):
    if split_shuffle:
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    else:
        idx = torch.arange(n)
    split_idx = int((1.0 - val_size) * n)
    train_sampler = SubsetRandomSampler(idx[:split_idx])
    val_sampler = SubsetRandomSampler(idx[split_idx:])
    return train_sampler, val_sampler

def train_val_dataloader(root_dir='./cifar10', split_shuffle=True, val_size=0.1, batch_size=120, index = False):
    DataSet = addIndexes(CIFAR10) if index else CIFAR10

    train_dataset = DataSet(root=root_dir, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), normalize]))
    val_dataset = CIFAR10(root=root_dir, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), normalize]))
    
    train_sampler, val_sampler = samplers(len(train_dataset), split_shuffle, val_size)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=int(batch_size / 3))
    return train_dataloader, val_dataloader


def train_dataloader(root_dir='./cifar10', batch_size=120, index = False, seed = None , subset =None ):
    DataSet = addIndexes(CIFAR10) if index else CIFAR10
    train_dataset = DataSet(root=root_dir, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), normalize]))
    if subset :
        n = len(train_dataset)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(0))
        split_idx = int(subset * n)
        train_sampler = SubsetRandomSampler(idx[:split_idx]) 
    else:
        train_sampler = RandomSampler(train_dataset)
    
    return DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)


def test_dataloader(root_dir='./cifar10', batch_size=120):
    test_dataset = CIFAR10(root=root_dir, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), normalize]))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    return test_dataloader

