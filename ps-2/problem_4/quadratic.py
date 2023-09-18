#!/usr/bin/env python
# coding: utf-8

# In[1]:


def quadratic(a,b,c):
    import numpy as np
    pos = (-b + np.sqrt(b**2-4*a*c))/(2*a)
    neg = (-b - np.sqrt(b**2-4*a*c))/(2*a)
    return pos, neg


# In[ ]:




