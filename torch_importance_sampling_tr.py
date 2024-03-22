import numpy as np
import torch
import torch.nn.functional as F
from common_utils import Accumulator
from torch.optim import Optimizer


class VarReductionCondition:

    def __init__(self, vr_th=1.2, momentum=0.9):
        self._vr_th = vr_th
        self._vr = 0.0
        self.previously_satisfied = False
        self._momentum = momentum

    @property
    def variance_reduction(self):
        return self._vr

    @property
    def satisfied(self):
        #self._previous_vr = self._vr
        self.previously_satisfied = (self._vr > self._vr_th)
        return self.previously_satisfied

    #@property
    #def previously_satisfied(self):
    #    return self._previous_vr > self._vr_th
    
    @property
    def string(self):
        return f"vr={self._vr:.3f},  vr_th={self._vr_th:.3f}"

    def update(self, scores):
        u = 1.0/len(scores)
        S = scores.sum()
        #if S == 0:
        #    g = torch.array(u)
        #else:

        g = scores/S
        new_vr = 1.0 / torch.sqrt(1 - ((g-u)**2).sum()/(g**2).sum())
        self._vr = (
            self._momentum * self._vr +
            (1-self._momentum) * new_vr
        )


def get_g(output, y_batch , loss_fn,  use_loss = False ):
    num_classes = output.shape[1]
    with torch.no_grad():
        if use_loss:
            g_i_norm = loss_fn(output, y_batch).detach()#.cpu().numpy()
        else:
            assert(type(loss_fn)==torch.nn.CrossEntropyLoss)
            # this is particular implementation for crossentropy loss
            probs = F.softmax(output, dim=1)
            one_hot_targets = F.one_hot(y_batch, num_classes=num_classes)
            g_i_norm = torch.norm(probs - one_hot_targets, dim=-1).detach()#.cpu().numpy()
    
    #print(g_i_norm.shape)
    return g_i_norm



def train_batch_is_tr(model,
                batch,  
                loss_fn, 
                optimizer : Optimizer, 
                accumulator : Accumulator,
                condition : VarReductionCondition, 
                presample = 3.0, 
                use_loss_estimation = False,
                second_approach = False):

    satisfied = False
    model.train()
    optimizer.zero_grad( set_to_none=True )

    (X_batch, y_batch) = batch
    batch_size = X_batch.shape[0]
    no_grad_y_batch = y_batch
    selected_batch_size = int(batch_size / presample)

    with torch.no_grad(): no_grad_output = model(X_batch)

    if condition.satisfied :
        if second_approach:
            
            g_i_norm = get_g(no_grad_output, y_batch, loss_fn, use_loss_estimation)
        else :
            output = model(X_batch)
            g_i_norm = get_g(output, y_batch, loss_fn, use_loss_estimation)

        condition.update(g_i_norm)
        satisfied = True
    else:
        #print("condition not satisfied")
        #g_i_norm = np.ones(batch_size)
        g_i_norm = torch.ones(batch_size).to(X_batch.device).detach()


    #p_i = g_i_norm / np.sum(g_i_norm)
    #batch_indices = np.random.choice(np.arange(batch_size), size = selected_batch_size, replace=True, p=p_i)
    #

    p_i = g_i_norm / torch.sum(g_i_norm)
    batch_indices = torch.multinomial(p_i, selected_batch_size, replacement = True )
    


    selected_p_i = p_i[batch_indices]

    if satisfied:
        if second_approach:
            output = model(X_batch[batch_indices])
            y_batch = y_batch[batch_indices]
            loss = loss_fn(output, y_batch)
            selected_loss = loss
        else:
            loss = loss_fn(output, y_batch)
            selected_loss = loss[batch_indices]
    else :
        output = model(X_batch[batch_indices])
        y_batch = y_batch[batch_indices]
        condition.update( get_g(output, y_batch, loss_fn, use_loss_estimation) )
        loss = loss_fn(output, y_batch)
        selected_loss = loss
        
    w_i = 1.0 / (batch_size * selected_p_i)

    #weighted_loss = (torch.tensor(w_i).to(selected_loss.device).detach() * selected_loss).mean()
    weighted_loss = (w_i.detach() * selected_loss).mean()

    weighted_loss.backward()

    optimizer.step()
    
    #max_p_i = np.max(p_i)
    #num_unique_points = np.unique(batch_indices).size
    
    with torch.no_grad():
        if second_approach:
            output = no_grad_output
            y_batch = no_grad_y_batch
            loss = loss_fn(output, y_batch)
            
        n = len(output)
        batch_loss = loss.mean()#.cpu().item()
        weighted_batch_loss = weighted_loss.mean()#.cpu().item()
        batch_acc_sum = (output.argmax(dim=1) == y_batch).sum()/n#.cpu().item()/n
        
        accumulator.average( 
            train_loss = ( batch_loss, n) ,
            train_acc = ( batch_acc_sum, n) ,
            train_w_loss = ( weighted_batch_loss, selected_batch_size) ,
            train_uniform_cnt = satisfied)
        
        #accumulator.store( 
        #    max_p_i = max_p_i ,
        #    num_unique_points = num_unique_points)