#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
def calc_quad1(a,b,c):
    pos = (-b + np.sqrt(b**2-4*a*c))/(2*a)
    neg = (-b - np.sqrt(b**2-4*a*c))/(2*a)
    return pos, neg
def calc_quad2(a,b,c):
    pos = 2*c/(-b-np.sqrt(b**2-4*a*c))
    neg = 2*c/(-b+np.sqrt(b**2-4*a*c))
    return pos, neg

ans1 = calc_quad1(.001, 1000, .001)
print('The normal quadratic equation gives', ans1)
ans2 = calc_quad2(.001, 1000, .001)
print('The alternative version gives', ans2)





