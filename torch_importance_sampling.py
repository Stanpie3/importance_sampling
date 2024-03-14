import numpy as np
import torch
import torch.nn.functional as F



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
        print(f"vr={self._vr:.3f},  vr_th={self._vr_th:.3f}")



def get_g(output, y_batch  ):
    num_classes = output.shape[1]
    with torch.no_grad():
        probs = F.softmax(output, dim=1)
        one_hot_targets = F.one_hot(y_batch, num_classes=num_classes)
        g_i_norm = torch.norm(probs - one_hot_targets, dim=-1).detach().cpu().numpy()
    return g_i_norm


def train_batch_is(model,
                x_batch, 
                y_batch, 
                loss_fn, 
                optimizer, 
                whole_dataset_size, 
                condition :VarReductionCondition, 
                presample = 3.0 ):


    model.train()
    model.zero_grad()

    batch_size = x_batch.shape[0]

    selected_batch_size = int(batch_size / presample)
    
    if condition.satisfied :
        print("in condition satisfied")
        output = model(x_batch)


        g_i_norm = get_g(output, y_batch  )

        condition.update(g_i_norm)
        
        N = batch_size
        print("condition satisfied")
    else:
        g_i_norm = np.ones(batch_size)
        print("condition not satisfied")
        N = whole_dataset_size

    p_i = g_i_norm / np.sum(g_i_norm)
    batch_indices = np.random.choice(np.arange(batch_size), size=selected_batch_size, replace=True, p=p_i)

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

        
    w_i = 1.0 / (N * selected_p_i)

    weighted_loss = (torch.tensor(w_i).to(selected_loss.device).detach() * selected_loss).mean()

    weighted_loss.backward()

    optimizer.step()

    batch_loss = loss.mean().cpu().item()
    weighted_batch_loss = weighted_loss.mean().cpu().item()
    max_p_i = np.max(p_i)

    num_unique_points = np.unique(batch_indices).size

    with torch.no_grad():
        batch_acc_sum = (output.argmax(dim=1) == y_batch).sum().cpu().item()
    
    return batch_loss, batch_acc_sum, weighted_batch_loss, max_p_i, num_unique_points, selected_batch_size