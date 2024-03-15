import numpy as np
import torch
import torch.nn.functional as F
from common_utils import Accumulator

class VarReductionCondition:

    def __init__(self, vr_th=1.2, momentum=0.9):
        self._vr_th = vr_th
        self._vr = 0.0
        self._previous_vr = 0.0
        self._momentum = momentum

    @property
    def variance_reduction(self):
        return self._vr

    @property
    def satisfied(self):
        self._previous_vr = self._vr
        return self._vr > self._vr_th

    @property
    def previously_satisfied(self):
        return self._previous_vr > self._vr_th
    
    @property
    def string(self):
        return f"vr={self._vr:.3f},  vr_th={self._vr_th:.3f}"

    def update(self, scores):
        u = 1.0/len(scores)
        S = scores.sum()
        if S == 0:
            g = np.array(u)
        else:
            g = scores/S
        new_vr = 1.0 / np.sqrt(1 - ((g-u)**2).sum()/(g**2).sum())
        self._vr = (
            self._momentum * self._vr +
            (1-self._momentum) * new_vr
        )


def get_g(output, y_batch  ):
    num_classes = output.shape[1]
    with torch.no_grad():
        probs = F.softmax(output, dim=1)
        one_hot_targets = F.one_hot(y_batch, num_classes=num_classes)
        g_i_norm = torch.norm(probs - one_hot_targets, dim=-1).detach().cpu().numpy()
    return g_i_norm



class CallBack:
    def __init__(self, eval_fn, name=None):
        self.eval_fn = eval_fn
        self.train_losses = []
        self.train_accs = []
        self.train_w_losses = []
        self.train_max_p_i = []
        self.train_num_unique_points = []
        self.val_losses = []
        self.val_accs = []
        self.n_un = []


        

    def last_info(self):
        return {'loss_train': f'{self.train_losses[-1]:.3f}',
                'acc_train': f'{self.train_accs[-1]:.3f}',
                'w_loss_train': f'{self.train_w_losses[-1]:.3f}',
                'loss_val': f'{self.val_losses[-1]:.3f}',
                'acc_val': f'{self.val_accs[-1]:.3f}',
                'n_un': f'{self.n_un[-1]:.3f}',
        }
    


    def __call__(self, model, val_dataloader, loss_fn,
                 epoch_loss=None, epoch_acc=None, epoch_weighted_loss=None, epoch_max_p_i_s=None, epoch_num_unique_points_s=None, n_un=None):
        self.train_losses.append(epoch_loss)
        self.train_accs.append(epoch_acc)
        self.train_w_losses.append(epoch_weighted_loss)
        self.train_max_p_i.append(epoch_max_p_i_s)
        self.train_num_unique_points.append(epoch_num_unique_points_s)
        loss_val, acc_val = self.eval_fn(model, val_dataloader, loss_fn)
        self.val_losses.append(loss_val)
        self.val_accs.append(acc_val)
        self.n_un.append(n_un)
        return self.last_info()


def train_batch_is(model,
                x_batch, 
                y_batch, 
                loss_fn, 
                optimizer,
                accumulator : Accumulator,  
                condition :VarReductionCondition, 
                presample = 3.0 ):

    flag = False
    model.train()
    model.zero_grad()

    batch_size = x_batch.shape[0]

    selected_batch_size = int(batch_size / presample)
    
    if condition.satisfied :
        #print("condition satisfied")
        output = model(x_batch)
        g_i_norm = get_g(output, y_batch  )
        condition.update(g_i_norm)

    else:
        #print("condition not satisfied")
        g_i_norm = np.ones(batch_size)


    p_i = g_i_norm / np.sum(g_i_norm)
    batch_indices = np.random.choice(np.arange(batch_size), size = selected_batch_size, replace=True, p=p_i)

    selected_p_i = p_i[batch_indices]

    if condition.previously_satisfied:
        loss = loss_fn(output, y_batch)
        selected_loss = loss[batch_indices]
    else :
        output = model(x_batch[batch_indices])
        y_batch = y_batch[batch_indices]
        condition.update( get_g(output, y_batch ) )
        loss = loss_fn(output, y_batch)
        selected_loss = loss
        flag = True
        
    w_i = 1.0 / (batch_size * selected_p_i)

    weighted_loss = (torch.tensor(w_i).to(selected_loss.device).detach() * selected_loss).mean()

    weighted_loss.backward()

    optimizer.step()

    max_p_i = np.max(p_i)

    num_unique_points = np.unique(batch_indices).size

    with torch.no_grad():
        batch_loss = loss.mean().cpu().item()
        weighted_batch_loss = weighted_loss.mean().cpu().item()
        batch_acc_sum = (output.argmax(dim=1) == y_batch).mean().cpu().item()

    n = len(output)


    accumulator( train_losses = ( batch_loss, n) ,
                 train_accs = ( batch_acc_sum, n) ,
                 train_w_losses = ( weighted_batch_loss, selected_batch_size) ,
                 train_flag = flag
                )

    return  max_p_i, num_unique_points