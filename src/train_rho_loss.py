import numpy as np
import torch
import torch.nn.functional as F

from src.utils.common import Accumulator
from torch.optim import Optimizer
from tqdm import tqdm 


def topIndices(vec, x, largest):
    sorted_idx = torch.argsort(vec, descending=largest)
    top_x_idxs, other_idxs = (sorted_idx[:x], sorted_idx[x:])
    return top_x_idxs, other_idxs

def irreducibleLoss(data=None, target=None, global_index=None, 
                              irreducible_loss_model=None, target_device=None):
    
    if type(irreducible_loss_model) is torch.Tensor:
        if (target_device is not None and irreducible_loss_model.device != target_device ):
            irreducible_loss_model = irreducible_loss_model.to(device=target_device)
        irreducible_loss = irreducible_loss_model[global_index]
    else:
        irreducible_loss = F.cross_entropy(irreducible_loss_model(data), target, reduction="none")
    return irreducible_loss

class ReducibleLossSelection:
    bald = False

    def __call__(self, selected_batch_size, data=None, target=None, global_index=None, large_model=None,
        irreducible_loss_model=None):

        with torch.no_grad():

            model_loss = F.cross_entropy( large_model(data), target, reduction="none")

            irreducible_loss = irreducibleLoss(data, target, global_index, irreducible_loss_model, model_loss.device)

            reducible_loss = model_loss - irreducible_loss

            top_x_idxs, _ = topIndices( reducible_loss, selected_batch_size, largest=True )
            selected_irreducible_loss = irreducible_loss[top_x_idxs]

        return top_x_idxs, selected_irreducible_loss
    
selection_method = ReducibleLossSelection()
update_irreducible = True

def train_irr_loss(
        irreducible_loss_clf, 
        data_loader,
        loss_fn, 
        opt, 
        accumulator : Accumulator = None , 
        selection__mode = True, presample = 3.0 ):
    
    # BEGIN Solution (do not delete this comment!)
    ret_loss, correct = (0.0, 0)
    irreducible_loss_clf.train(True)  # Set model to training mode
    
    for x, labels in data_loader:
        x, labels = x.cuda(), labels.cuda()
        opt.zero_grad()
        all_pred = irreducible_loss_clf(x)
        loss = loss_fn(all_pred, labels).mean()

        loss.backward()
        opt.step()
        
        ret_loss += loss.item()
        #pred = all_pred.max(1)[1]

        with torch.no_grad():
            #correct += (pred==labels).sum().item()
            correct += (all_pred.argmax(dim=1) == labels).sum().cpu().item()

    ret_loss = ret_loss / len(data_loader)
    accuracy = correct / len(data_loader.dataset)


    return {"ret_loss":ret_loss, "accuracy": accuracy}


def train_batch_rho_loss(
        large_model,
        irreducible_loss_model,
        batch,
        loss_fn, 
        optimizer, 
        accumulator : Accumulator, 
        selection__mode = True, presample = 3.0 ):


    global_index, data, target = batch
    batch_size = len(data)
    selected_batch_size = max(1, int(batch_size /presample))


    large_model.eval() 

    selected_indices,  irreducible_loss = selection_method.__call__(
        selected_batch_size=selected_batch_size,
        data=data,
        target=target,
        global_index=global_index,
        large_model=large_model,
        irreducible_loss_model=irreducible_loss_model,
    ) 

    large_model.train()  # switch to eval mode to compute selection

    data, target = data[selected_indices], target[selected_indices]

    optimizer.zero_grad()
    logits = large_model(data)
    loss = loss_fn(logits, target) 

    mean_loss = loss.mean()
    mean_loss.backward()
    optimizer.step()

    # training metrics
    #preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)

    n = len(logits)
    acc = (logits.argmax(dim=1) == target).sum().cpu().item()/n

    accumulator.average( 
        train_loss = ( mean_loss.cpu().item(), n) ,
        train_acc = ( acc, n) )




def train_full_rho_loss(model, 
                        model_irr, 
                        train_dataloader,
                        train_irr_loader, 
                        loss_fn, 
                        optimizer, 
                        optimizer_irr,
                        n_epochs, 
                        eval = None, 
                        callback = None, 
                        presample = 3, 
                        tau_th = None):


    large_batch = int( train_dataloader.batch_size)
    
    if callback :
        callback.setMeta(
            large_batch = large_batch,
            n_epochs = n_epochs, 
            presample = presample, 
            tau_th = tau_th)
        

    try:
        model.load_state_dict(torch.load("irr_model"))
        model.eval()
        print("irr model loades")
    except:
        epochs = tqdm(range(n_epochs), desc='Irr epochs', leave=True)
        for i_epoch in epochs:
            dict_ = train_irr_loss(model_irr, train_irr_loader, loss_fn, optimizer_irr)
            epochs.set_postfix(dict_)

    torch.save(model_irr.state_dict(), "irr_model")
    
    d = model.device
    epochs = tqdm(range(n_epochs), desc='Epochs', leave=True)
    for i_epoch in epochs:
        accum = Accumulator()
        
        
        for idxs, X_batch, y_batch in train_dataloader:
            train_batch_rho_loss(model, 
                                model_irr,
                                (idxs, X_batch.to(d), y_batch.to(d)),
                                loss_fn,
                                optimizer, 
                                accum,
                                presample)

        if callback :
            val_scores = eval(model) if eval else {}
            cb_dict = callback( **accum.getAll(), **val_scores)
            epochs.set_postfix(cb_dict)