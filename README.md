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
...     acc.store(any_name_youwhant = 9, 
...               or_even_this= [i,i+1])
... 
>>> acc.getAll()
{'some_name_1': 5.0, 'some_name_2': 6.333333333333333, 'blabla': 5.0, 'blabla_2': 1.0, 'any_name_youwhant': [9, 9, 9, 9, 9, 9, 9, 9, 9], 'or_even_this': [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10]]}
>>>

```
