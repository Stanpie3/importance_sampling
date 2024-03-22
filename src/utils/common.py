

import os
import pickle
from pathlib import Path, PurePath
from typing import Any, Callable
from torch import is_tensor, nn


def _has_len(obj):
    return hasattr(obj, '__len__')

class Average():
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = 0
        
    def avg(self):
        if (is_tensor(self.sum)):
            return self.sum.cpu().item()/float(self.count)
        return  self.sum / float(self.count)

    def update(self, val, n=1):
        assert(n > 0)
        self.sum += val * n
        self.count += n


class Accumulator():
    def __init__(self):
        self._av = {}
        self._storage = {}

    def average(self,**kwargs):
        for key, i in kwargs.items():
            if not (key in self._av):
                self._av[key] = Average()
            d , n = i if _has_len(i) and len(i)>1 else (i,1)
            self._av[key].update(d , n)

    def store(self,**kwargs):
        for key, i in kwargs.items():
            if not (key in self._storage):
                self._storage[key] = []
            self._storage[key].append(i)

    def __str__(self):
        return "["+ ", ".join([f"{key}: {i.avg()}" for key, i in self.getAll.items()]) + "]"

    def getStored(self):
        return self._storage

    def getAverage(self):
        return {key: i.avg() for key, i in self._av.items()}
    
    def getAll(self):
        return {**self.getAverage(), **self._storage}
    

class UnCallBack():
    __exclude = ["info_list", "name", "meta"]

    def __init__(self, name="callback", info_list = [], meta = None ):
        # for example ['loss_train','acc_train',"w_loss_train','loss_val','acc_val','n_un']
        self.info_list = info_list
        self.name = name
        self.meta = None

    def setMeta(self,**kwargs):
        self.meta = kwargs

    def __call__(self, **kwargs):
        for key, i in kwargs.items():
           if not (key in self.__dict__):
               self.__dict__[key] = []
           self.__dict__[key].append(i)       


        return self.last_info()

    def last_info(self):
       ret = {}
       for key in self.info_list:
           val = self.__dict__.get(key, None)
           if val :
               ret[key] = f'{val[-1]:.3f}'   
           else :
               ret[key] = 'undefined'
       return ret
       #return {key: f'{self.__dict__[key][-1]:.3f}' for key in self.info_list if  self.__dict__.get(key, None) }
    
    def save(self, name = None):
        if name is None :
            name = self.name
        
        p = PurePath(name)
        np= str(p.parent)+"/"+p.stem 
        suffix = p.suffix if  p.suffix else ".pickle"
        
        isExist = os.path.exists(p.parent)
        if not isExist:
            os.makedirs(p.parent)

        name = f"{np}{suffix}"
        if os.path.exists(name):
            i = 0
            name = f"{np}_{i}{suffix}"
            while os.path.exists(name):
                i += 1
                name = f"{np}_{i}{suffix}"

      

        with open(name, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(name):
        with open(name, 'rb') as f:
            callback = pickle.load(f)
        return callback

    def __str__(self):
        return "\n".join(
            [ f"{key} ({type(i[0])})" for key, i in self.__dict__.items() if not key in self.__exclude ])


    def __repr__(self):
        return "meta:\n"+str(self.meta) + "\n\ncolumns:\n" + self.__str__()
    


Eval_ty = Callable[[nn.Module], dict[str, Any]] 