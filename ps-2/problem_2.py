#!/usr/bin/env python
# coding: utf-8

#run with ipython to make this work

import numpy as np
import timeit

size = 100
#for loops simple calculation
def madelung_forloop(size):
    madelung = 0
    for i in range(-size,size+1): # +1 to make center zero
        for j in range(-size,size+1):
            for k in range(-size,size+1):
                if i == j == k == 0:
                    continue
                madelung += (-1)**(i+j+k)/((i*i + j*j + k*k)**.5)
    return madelung

def madelung_where(size):
    one_dim = np.arange(-size,size+1)
    i, j, k = np.meshgrid(one_dim,one_dim,one_dim)
    return np.where((i!=0) | (j!=0) | (k!=0),(-1)**(abs(i+j+k))/((i*i + j*j +k*k)**.5),0).sum()


Mwhere = madelung_where(size)
Mloop = madelung_forloop(size)

print('Madelung with where function: ', Mwhere)
print('Madelung with for loop: ',Mloop)


get_ipython().run_line_magic('timeit', 'madelung_where(size)')
get_ipython().run_line_magic('timeit', 'madelung_forloop(size)')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




