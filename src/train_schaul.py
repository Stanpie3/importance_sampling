import numpy as np
import torch
import torch.nn.functional as F

from src.utils.common import Accumulator , Eval_ty
from torch.optim import Optimizer
from tqdm import tqdm 


def train_batch_schaul(
            model,
            x_batch,
            y_batch,
            loss_fn,
            optimizer: Optimizer,
            accumulator: Accumulator,
            weights ):


    model.train()
    optimizer.zero_grad()
    
    output = model(x_batch)
    loss = loss_fn(output, y_batch) * weights
    loss = loss.mean()

    loss.backward()
    optimizer.step()

    n = len(output)
    with torch.no_grad():
        batch_loss = loss.mean().cpu().item()
        batch_acc_sum = (output.argmax(dim=1) == y_batch).sum().cpu().item()/n

        accumulator.average(
            train_loss = ( batch_loss, n) ,
            train_acc = ( batch_acc_sum, n) )



def train_full_schaul( model, 
                train_loader, 
                loss_fn, 
                optimizer, 
                n_epochs : int, 
                eval : Eval_ty = None,
                callback=None, 
                presample = 3, 
                beta = 1,
                alpha = 0.7,
                device = "cpu"):
    
    epochs = tqdm(range(n_epochs), desc='Epochs', leave=True)
    
    X = torch.tensor(train_loader.dataset.data, dtype=torch.float32).transpose(1, -1)
    y = torch.tensor(train_loader.dataset.targets).long()
    
    batch_size = int( train_loader.batch_size)
    mini_batch_size = int(batch_size/ presample)
    n_batches = len(train_loader)
    size = n_batches * batch_size

    if callback :
        callback.setMeta(
            large_batch = batch_size,
            n_epochs = n_epochs)

    
    sample_weight_logit = torch.ones(len(X))

    for _ in epochs:
        accum = Accumulator()

        probs = F.softmax(sample_weight_logit, dim=-1)

        batch_indices = torch.multinomial(probs, 
                                          n_batches*mini_batch_size, 
                                          replacement=True).view(n_batches, mini_batch_size)
        
        weights = 1. / (probs[batch_indices] * len(X))**beta
        weights /= weights.sum(dim=1, keepdim=True)
        #weights /= weights.max()

        for batch_idx in range(n_batches):
            #print(torch.unique(y[batch_indices[batch_idx]],return_counts = True))
            train_batch_schaul( model,
                            X[batch_indices[batch_idx]].to(device),
                            y[batch_indices[batch_idx]].to(device),
                            loss_fn,
                            optimizer,
                            accum,
                            weights[batch_idx].to(device).detach())
        model.eval()

        with torch.no_grad():
            losses = torch.zeros(len(X)).to(device)
            #print(len(X), size)
            for batch_idx in range(n_batches):
                
                pos = batch_idx * batch_size 
                end_pos = (batch_idx + 1) * batch_size 
                
                output = model(X[pos : end_pos].to(device))
                tmp_loss =F.nll_loss(output, y[pos : end_pos].to(device), reduction='none')
                losses[pos : end_pos] = tmp_loss
            
            sample_weight_logit = (losses * alpha).cpu()
            # clip logits for stability
            percentiles = np.percentile(sample_weight_logit, [10, 90])
            sample_weight_logit = torch.clamp(sample_weight_logit, percentiles[0], percentiles[1])


        if callback :
            val_scores = eval(model) if eval else {}
            cb_dict = callback( **accum.getAll(), **val_scores)
            epochs.set_postfix(cb_dict)


def train_full_schaul2(model, train_dataloader, loss_fn, optimizer, n_epochs, eval = None, callback=None, device = "cpu"):
    epochs = tqdm(range(n_epochs), desc='Epochs', leave=True)
    X = torch.tensor(train_dataloader.dataset.data, dtype=torch.float32).transpose(1, -1)
    y = torch.tensor(train_dataloader.dataset.targets).long()
    batch_size = int( train_dataloader.batch_size)
    n_batches = len(train_dataloader)

    if callback :
        callback.setMeta(
            large_batch = batch_size,
            n_epochs = n_epochs)



    sample_weight_logit = torch.ones(len(train_dataloader))

    for i_epoch in epochs:
        accum = Accumulator()

        probs = F.softmax(sample_weight_logit, dim=-1)
        batch_indices = torch.multinomial(probs, n_batches * batch_size, replacement=True).view(n_batches, batch_size)
        importance_sampling_weights = 1/(probs[batch_indices]*n_batches*batch_size)
        importance_sampling_weights /= importance_sampling_weights.sum(dim=1, keepdim=True)
        importance_sampling_weights = importance_sampling_weights.to(device)



        for batch_idx in range(len(train_dataloader)):
            train_batch_schaul( model,
                            X[batch_indices[batch_idx]].to(device),
                            y[batch_indices[batch_idx]].to(device),
                            loss_fn,
                            optimizer,
                            accum,
                            importance_sampling_weights[batch_idx])
        model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            losses = torch.zeros(n_batches * batch_size)
            for batch_idx in range(len(train_dataloader)):
                output = model(X[batch_idx * batch_size : (batch_idx + 1) * batch_size].to(device))
                losses[batch_idx * batch_size : (batch_idx + 1) * batch_size] = F.nll_loss(output, y[batch_idx * batch_size : (batch_idx + 1) * batch_size].to(device), reduction='none').cpu()


            alpha = 1
            sample_weight_logit = losses*alpha
            # clip logits for stability
            percentiles = np.percentile(sample_weight_logit, [10, 90])
            sample_weight_logit = torch.clamp(sample_weight_logit, percentiles[0], percentiles[1])


        if callback :
            val_scores = eval(model) if eval else {}
            #print(condition.string + f" n_un={callback.n_un[-1]}")
            cb_dict = callback(  **accum.getAll(), **val_scores)
            epochs.set_postfix(cb_dict)



def train_full_schaul3(model, train_dataloader, loss_fn, optimizer, n_epochs, eval = None, callback=None, device = "cpu"):
    epochs = tqdm(range(n_epochs), desc='Epochs', leave=True)
    X = torch.tensor(train_dataloader.dataset.data, dtype=torch.float32).transpose(1, -1)
    y = torch.tensor(train_dataloader.dataset.targets).long()

    batch_size = int( train_dataloader.batch_size)
    n_batches = len(train_dataloader)

    if callback :
        callback.setMeta(
            large_batch = batch_size,
            n_epochs = n_epochs)



    sample_weight_logit = torch.ones(len(train_dataloader))

    for i_epoch in epochs:
        accum = Accumulator()

        probs = F.softmax(sample_weight_logit, dim=-1)
        batch_indices = torch.multinomial(probs, n_batches * batch_size, replacement=True).view(n_batches, batch_size)
        weights = 1/(probs[batch_indices]*n_batches*batch_size)
        weights /= weights.sum(dim=1, keepdim=True)
        weights =  weights.to(device)



        for batch_idx in range(len(train_dataloader)):
            train_batch_schaul( model,
                            X[batch_indices[batch_idx]].to(device),
                            y[batch_indices[batch_idx]].to(device),
                            loss_fn,
                            optimizer,
                            accum,
                            weights[batch_idx])
        model.eval()


        with torch.no_grad():
            losses = torch.zeros(n_batches * batch_size)
            for batch_idx in range(len(train_dataloader)):
                pos = batch_idx * batch_size 
                end_pos = (batch_idx + 1) * batch_size 
                output = model(X[pos: end_pos].to(device))
                losses[pos: end_pos] = F.nll_loss(output, y[pos: end_pos].to(device), reduction='none').cpu()


            alpha = 1
            sample_weight_logit = losses*alpha
            # clip logits for stability
            percentiles = np.percentile(sample_weight_logit, [10, 90])
            sample_weight_logit = torch.clamp(sample_weight_logit, percentiles[0], percentiles[1])


        if callback :
            val_scores = eval(model) if eval else {}
            #print(condition.string + f" n_un={callback.n_un[-1]}")
            cb_dict = callback( **accum.getAll(), **val_scores)
            epochs.set_postfix(cb_dict)