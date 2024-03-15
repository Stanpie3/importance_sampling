

class Average():
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = 0
        
    def avg(self):
        self.avg = self.sum / float(self.count)

    def update(self, val, n=1):
        assert(n > 0)
        self.sum += val * n
        self.count += n


class Accumulator():
    def __init__(self):
        items = dict()

    def __call__(self,**kwargs):
        for i in kwargs:
            if len(i)==2: 
                pass





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