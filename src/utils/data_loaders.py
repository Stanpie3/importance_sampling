

from typing import Iterable, Iterator, List, Sized, Union
import torch
from torchvision.datasets import CIFAR10, MNIST, CIFAR100, VisionDataset
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, Sampler
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



class RnadomBatchSampler(Sampler[List[int]]):
    r"""Sample real random batches.

    Args:

        data_source (Dataset): dataset to sample from
        batch_size (int): Size of mini-batch.
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        generator (Generator): Generator used in sampling.

    """

    def __init__(self, dataset: Sized, batch_size: int, replacement = False, generator = None) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")

        self._replacement = replacement
        self._dataset_size = len(dataset)
        self._batch_size = batch_size
        self._generator = generator
        self._unif = torch.ones(self.dataset_size)


    def __iter__(self) -> Iterator[List[int]]:
        n = self.__len__()

        for _ in range(n):
            batch = self._unif.multinomial(self._batch_size, 
                                           replacement=self._replacement, 
                                           generator= self._generator)
            yield batch
        #except StopIteration:

    def __len__(self) -> int:
        n = self._dataset_size // self._batch_size
        n += (self._dataset_size % self._batch_size) > 0
        return n




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


def test_dataloader(root_dir='./cifar10', batch_size=120, index = False ):
    DataSet = addIndexes(CIFAR10) if index else CIFAR10
    test_dataset = DataSet(root=root_dir, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), normalize]))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_dataloader



if __name__ == '__main__':
    import numpy as np

    train = train_dataloader(batch_size=120,index=True,subset=0.6)
    print(train._auto_collation)
    print(train.batch_sampler)
    print(train.sampler)
    print(train._index_sampler)

    l = []
    for i,x,y in train:
        i :torch.Tensor = i
        #print(i)
        l.extend(i)
        #print(i)
    
    values, counts = np.unique(l, return_counts=True)
    print(max(counts),len(l), len(train))