from torch_svrg_tr import SVRG_k ,SVRG_Snapshot, test_epoch_SVRG, train_epoch_SVRG


import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from tqdm import tqdm

mean, std = [0.5, 0.5, 0.5] , [0.5, 0.5, 0.5]
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(), 
    transforms.Normalize(mean, std) 
])
test_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean, std)
])

train_set = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

test_set = CIFAR10(root='./data', train=False, download=True, transform=test_transform) 
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)


### END Solution (do not delete this comment)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print('Train size', len(train_set))
print('Test size', len(test_set))



n_epochs = 20

device = "cuda"

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            # BEGIN Solution (do not delete this comment!)

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)   

            # END Solution (do not delete this comment!)
        )
        self.classifier = nn.Sequential(
            # BEGIN Solution (do not delete this comment!)
            nn.Linear(64 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
            # END Solution (do not delete this comment!)
        )

    def forward(self, x):
        # BEGIN Solution (do not delete this comment!)
        
        return self.classifier(self.features(x).view(x.size(0), -1) )
        # END Solution (do not delete this comment!)



model = CNN().to(device)
model_snapshot = CNN().to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = SVRG_k(model.parameters(), lr=0.001, weight_decay = 0.0001)
optimizer_snapshot = SVRG_Snapshot(model_snapshot.parameters())

for epoch in tqdm(range(n_epochs)):
    train_loss, train_acc = train_epoch_SVRG(model, model_snapshot, optimizer , optimizer_snapshot, train_loader, loss_fn )
    test_loss, test_acc = test_epoch_SVRG(model, test_loader,  loss_fn)
    print(f'[Epoch {epoch + 1}] train loss: {train_loss:.3f}; train acc: {train_acc:.2f}; ' + 
          f'test loss: {test_loss:.3f}; test acc: {test_acc:.2f}')