# importance_sampling


# About common_utils
How to use Accumulator from the utils 

```python
>>> from common_utils import Accumulator
>>> acc = Accumulator( )
>>> for i in range(1,10):
...     acc.average(some_name_1 = (i,10),
...                 some_name_2 = (i,i),
...                 blabla = i,
...                 blabla_2 = 1)
...
...     acc.store(any_name_youwant = 9, 
...               or_even_this= [i,i+1])
... 
>>> acc.getAll()
>>>

```
