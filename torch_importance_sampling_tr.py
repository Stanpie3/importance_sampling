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


def get_g(output, y_batch , loss_fn,  use_loss = False ):
    num_classes = output.shape[1]
    with torch.no_grad():
        if use_loss:
            g_i_norm = loss_fn(output, y_batch).detach().cpu().numpy()
        else:
            assert(type(loss_fn)==torch.nn.CrossEntropyLoss)
            # this is particular implementation for crossentropy loss
            probs = F.softmax(output, dim=1)
            one_hot_targets = F.one_hot(y_batch, num_classes=num_classes)
            g_i_norm = torch.norm(probs - one_hot_targets, dim=-1).detach().cpu().numpy()
    
    #print(g_i_norm.shape)
    return g_i_norm

