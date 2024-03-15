import torch
from torch.optim import Optimizer
import copy

def accuracy(yhat, labels):
    _, indices = yhat.max(1)
    return (indices == labels).sum().data.item() / float(len(labels))

class AverageCalculator():
    def __init__(self):
        self.reset() 
    
    def reset(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
    
    def update(self, val, n=1):
        assert(n > 0)
        self.sum += val * n 
        self.count += n
        self.avg = self.sum / float(self.count)



class SVRG_k(Optimizer):
    r"""Optimization class for calculating the gradient of one iteration.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
    """
    def __init__(self, params, lr, weight_decay=0):
        print("Using optimizer: SVRG")
        self.u = None
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(SVRG_k, self).__init__(params, defaults)
    
    def get_param_groups(self):
            return self.param_groups

    def set_u(self, new_u):
        """Set the mean gradient for the current epoch. 
        """
        if self.u is None:
            self.u = copy.deepcopy(new_u)
        for u_group, new_group in zip(self.u, new_u):  
            for u, new_u in zip(u_group['params'], new_group['params']):
                u.grad = new_u.grad.clone()

    def step(self, params):
        """Performs a single optimization step.
        """
        for group, new_group, u_group in zip(self.param_groups, params, self.u):
            weight_decay = group['weight_decay']

            for p, q, u in zip(group['params'], new_group['params'], u_group['params']):
                if p.grad is None:
                    continue
                if q.grad is None:
                    continue
                # core SVRG gradient update 
                new_d = p.grad.data - q.grad.data + u.grad.data
                if weight_decay != 0:
                    new_d.add_(weight_decay, p.data)
                p.data.add_(-group['lr'], new_d)


class SVRG_Snapshot(Optimizer):
    r"""Optimization class for calculating the mean gradient (snapshot) of all samples.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
    """
    def __init__(self, params):
        defaults = dict()
        super(SVRG_Snapshot, self).__init__(params, defaults)
      
    def get_param_groups(self):
            return self.param_groups
    
    def set_param_groups(self, new_params):
        """Copies the parameters from another optimizer. 
        """
        for group, new_group in zip(self.param_groups, new_params):
            for p, q in zip(group['params'], new_group['params']):
                  p.data[:] = q.data[:]




def train_epoch_SVRG(model_k, 
                     model_snapshot, 
                     optimizer_k, 
                     optimizer_snapshot, 
                     train_loader, 
                     loss_fn, 
                     metric = accuracy,
                     device = "cuda",
                     data_transform = lambda x:x):
    model_k.train()
    model_snapshot.train()
    loss = AverageCalculator()
    av_metric = AverageCalculator()

    # calculate the mean gradient
    optimizer_snapshot.zero_grad()  # zero_grad outside for loop, accumulate gradient inside
    for points, labels in train_loader:
        points = points.to(device)
        points = data_transform(points)
            
        pred = model_snapshot(points)
        labels = labels.to(device)
        snapshot_loss = loss_fn(pred, labels) / len(train_loader)
        snapshot_loss.backward()

    # pass the current paramesters of optimizer_0 to optimizer_k 
    u = optimizer_snapshot.get_param_groups()
    optimizer_k.set_u(u)
    
    for points, labels in train_loader:
        points = points.to(device)
        points = data_transform(points)

        pred_1 = model_k(points)
        labels = labels.to(device)
        loss_iter = loss_fn(pred_1, labels)

        # optimization 
        optimizer_k.zero_grad()
        loss_iter.backward()    

        pred_2 = model_snapshot(points)
        loss_2 = loss_fn(pred_2, labels)

        optimizer_snapshot.zero_grad()
        loss_2.backward()

        optimizer_k.step(optimizer_snapshot.get_param_groups())

        loss.update(loss_iter.data.item())
        av_metric.update(metric(pred_1, labels))

    # update the snapshot 
    optimizer_snapshot.set_param_groups(optimizer_k.get_param_groups())
    
    return loss.avg, av_metric.avg



def test_epoch_SVRG(model, test_loader, loss_fn, metric = accuracy, device = "cuda", data_transform = lambda x:x):
    """One epoch of validation
    """
    model.eval()
    loss = AverageCalculator()
    av_metric = AverageCalculator()
    with torch.no_grad():
        for points, labels in test_loader:
            points = points.to(device)
            points = data_transform(points)
            pred = model(points)
            labels = labels.to(device)

            loss_iter = loss_fn(pred, labels)
            av_metric.update(metric(pred, labels))
            loss.update(loss_iter.data.item())
            
    
    return loss.avg, av_metric.avg