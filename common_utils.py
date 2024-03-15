

def _has_len(obj):
    return hasattr(obj, '__len__')

class Average():
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = 0
        
    def avg(self):
        return  self.sum / float(self.count)

    def update(self, val, n=1):
        assert(n > 0)
        print(self.count)
        self.sum += val * n
        self.count += n


class Accumulator():
    def __init__(self):
        pass

    def __call__(self,**kwargs):
        for key, i in kwargs.items():
            if not (key in self.__dict__):
                self.__dict__[key] = Average()
            d , n = i if _has_len(i) and len(i)>1 else (i,1)
            self.__dict__[key].update(d , n)

    def __str__(self):
        return "["+ ", ".join([f"{key}: {i.avg()}" for key, i in self.__dict__.items()]) + "]"

    def get_av_dict(self):
        return {key: i.avg() for key, i in self.__dict__.items()}
    
class UnCallBack():
    def __init__(self, info_list = [] ):
        # for example ['loss_train','acc_train',"w_loss_train','loss_val','acc_val','n_un']
        self.info_list = info_list
        for key in self.info_lis:
           self.__dict__[key] = []

    def __call__(self, **kwargs):
        for key, i in kwargs.items():
           if not (key in self.__dict__):
               self.__dict__[key] = []
           self.__dict__[key].append()       

    def last_info(self):
       return {key: f'{self.__dict__[key] [-1]:.3f}' for key in self.info_list}
    

class CallBack_old:
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
    


acc = Accumulator()



acc(accuracy= (10,20), data=10,newo =6.6)

acc(accuracy= (10,10), data=5,newo =6.6)
acc(accuracy= (10,10), data=5,newo =6.6)


CallBack()

print(acc)