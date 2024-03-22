import numpy as np
import torch
import torch.nn.functional as F
from src.utils.common import Accumulator
from torch.optim import Optimizer
from tqdm import tqdm 

def create_batches(p, N, k, b):
    all_points = np.arange(N)
    batches = []
    for j in range(k):
        batch = np.random.choice(all_points, size=b, replace=False, p=p)
        batches.append(batch)
        tmp = p[batch].sum()
        p[batch] = 0
        if j != k-1:
            p[p>0] += tmp / p[p>0].shape[0]
    return batches

def get_batches(loss, se, b):
    sorted_arg_loss = (-loss).argsort()
    N = loss.shape[0]
    p = np.zeros(N)
    for i in range(N):
        p[sorted_arg_loss[i]] = 1.0 / np.exp(np.log(se) / N)**(i+1)
    p /= p.sum()
    #print(f'Minimal probability {p.min()}, maximal probability {p.max()}')
    return create_batches(p, N, N//b, b)



def train_batch_loshchilov(model,
                x_batch,
                y_batch,
                loss_fn,
                optimizer: Optimizer,
                accumulator ):

    model.train()
    optimizer.zero_grad()

    output = model(x_batch)
    loss = loss_fn(output, y_batch)
    loss = loss.mean()

    loss.backward()
    optimizer.step()


    n = len(output)
    with torch.no_grad():
        batch_loss = loss.mean().cpu().item()
        batch_acc_sum = (output.argmax(dim=1) == y_batch).sum().cpu().item()/n

    accumulator.average(
        train_loss = ( batch_loss, n),
        train_acc = ( batch_acc_sum, n))
    



def train_full_loshchilov(model, train_dataloader, loss_fn, optimizer, n_epochs, eval = None, callback=None, device = "cpu"):

    epochs = tqdm(range(n_epochs), desc='Epochs', leave=True)
    X = torch.tensor(train_dataloader.dataset.data, dtype=torch.float32).transpose(1, -1)
    y = torch.tensor(train_dataloader.dataset.targets).long()
    batch_size = int(train_dataloader.batch_size)
    # n_batches = len(train_dataloader)

    if callback :
        callback.setMeta(
            large_batch = batch_size,
            n_epochs = n_epochs)
        
    n_batches = y.shape[0] // batch_size
    losses = torch.zeros(n_batches * batch_size)
    se_0, se_end = 10.0**2, 1.0
    se = se_0
    for i_epoch in epochs:
        accum = Accumulator()
        batch_indices = get_batches(np.abs(losses), se, batch_size)
        #print(len(batch_indices), len(batch_indices[0]))

        for batch_idx in range(len(batch_indices)):
            train_batch_loshchilov( model,
                            X[batch_indices[batch_idx]].to(device),
                            y[batch_indices[batch_idx]].to(device),
                            loss_fn,
                            optimizer,
                            accum)
        model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            losses = torch.zeros(n_batches * batch_size)
            for batch_idx in range(len(batch_indices)):
                output = model(X[batch_indices[batch_idx]].to(device))
                losses[batch_indices[batch_idx]] = loss_fn(output, y[batch_indices[batch_idx]].to(device)).cpu()
        se = se_0 * np.exp(np.log(se_end/se_0)/n_epochs) ** i_epoch

        if callback :
            val_scores = eval(model) if eval else {}
            cb_dict = callback(**accum.getAll(), **val_scores)
            epochs.set_postfix(cb_dict)