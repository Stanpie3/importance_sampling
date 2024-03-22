import time
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.common import Accumulator
from torch.optim import Optimizer
from tqdm import tqdm 



def train_on_batch(clf, x_batch, y_batch, opt, loss_fn, device = "cpu"):
    clf.train(True)

    opt.zero_grad()
    output = clf.forward(x_batch.to(device))

    loss = loss_fn(output, y_batch.to(device)).mean()
    loss.backward()

    accuracy = np.mean((y_batch.cpu().numpy() == output.argmax(1).cpu().numpy()))

    opt.step()
    return loss.cpu().item(), accuracy.item()

def epoch_train(loader, clf, loss_fn, opt, device = "cpu"):

    epoch_loss = np.array([])
    epoch_accuracy = np.array([])

    for it, (batch_of_x, batch_of_y) in enumerate(loader):
        batch_loss, batch_accuracy = train_on_batch(
            clf, 
            batch_of_x.to(device), 
            batch_of_y.to(device), 
            opt, 
            loss_fn, 
            device = device )

        epoch_loss = np.append(epoch_loss, batch_loss)
        epoch_accuracy = np.append(epoch_accuracy, batch_accuracy)

    return np.mean(epoch_loss), np.mean(epoch_accuracy)



def test_on_batch(clf, x_batch, y_batch, loss_fn, device = "cpu"):
    clf.eval()

    with torch.no_grad():

        output = clf.forward(x_batch.to(device))
        loss = loss_fn(output, y_batch.to(device)).mean()
        accuracy = np.mean((y_batch.cpu().numpy() == output.argmax(1).cpu().numpy()))

    return loss.cpu().item(), accuracy.item()


def epoch_test(loader, clf, loss_fn, device = "cpu"):

    epoch_loss = np.array([])
    epoch_accuracy = np.array([])

    for it, (batch_of_x, batch_of_y) in enumerate(loader):
        batch_loss, batch_accuracy = test_on_batch(
            clf, 
            batch_of_x.to(device), 
            batch_of_y.to(device),  
            loss_fn, 
            device = device )

        epoch_loss = np.append(epoch_loss, batch_loss)
        epoch_accuracy = np.append(epoch_accuracy, batch_accuracy)

    return np.mean(epoch_loss), np.mean(epoch_accuracy)




def train_full_apricot(clf, train_loader,test_loader, loss_fn, opt, n_epochs,   callback=None,  device = "cpu"):

    #dict = {'time':np.array([]), 'train_loss':np.array([]), 'train_acc':np.array([]), 'test_loss':np.array([]), 'test_acc':np.array([])}
    
    start = time.time()
    
    for epoch in tqdm(range(n_epochs)):
        train_loss, train_acc = epoch_train(train_loader, clf, loss_fn, opt, device = device)
        test_loss, test_acc = epoch_test(test_loader, clf, loss_fn, device = device)

        if callback:
            cb_dict = callback(
                time = (time.time() - start),
                train_loss = train_loss,
                train_acc = train_acc,
                test_loss = test_loss,
                test_acc = test_acc)

            epoch.set_postfix(cb_dict)
        #print(f'[time {epoch + 1}] train loss: {train_loss:.3f}; train acc: {train_acc:.2f}; ' +
        #      f'test loss: {test_loss:.3f}; test acc: {test_acc:.2f}')

    return dict
