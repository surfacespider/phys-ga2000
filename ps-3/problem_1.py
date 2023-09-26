#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
def our_function(x_value):
    return (x_value*(x_value-1)) #our function is x*(x-1)
def derivative(x_value, delta):
    numerator = our_function((x_value+delta)) - our_function(x_value) # the derivative of x*(x-1) is 2x - 1)
    return numerator/delta

delta = 10**-2
print('the derivative of x(x-1) at x = 1 is 1, with a delta of 10^-2 we get', derivative(1,delta), 'a difference of 10^-2')
print(' ') #this differs because we have to take the limit as delta goes to zero so we need a smaller delta

deltas = [10**-2, 10**-4,10**-6,10**-8,10**-10,10**-12,10**-14]
for i in deltas:
    print('numerical derivative for delta =', i, ':', derivative(1,i))
    print('error from actual:', abs(derivative(1,i)-1))
    print(' ') #the error initially starts dropping but then starts rising again. computers dont like
    # really small numbers. as we get smaller we start getting rounding errors building up


