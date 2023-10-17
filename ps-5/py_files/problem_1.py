#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_legendre
import scipy.integrate as integrate


def function_original(x,a):
    # integral of this function from 0 to infinity is the definition of gamma(a)
    return x**(a-1)*np.exp(-x)

def function_rewrite(x,a):
    #rewrite the function into an equivalent form that is better behaved for computers
    return np.exp((a-1)*np.log(x)-x)
def x_transform(x,a):
    #we want to integrate from 0 to 1 not 0 to inf. this transform does 
    #that while centering the transformation at the position
    #x = a-1, the maximum of the function so that the integral is better
    return (a-1)*x/(1-x)
    
def gamma(a):
    f = lambda x: (a-1)*function_rewrite(x_transform(x,a),a)/(1-x)**2 #function after bounds transform
    s = integrate.fixed_quad(f, 0, 1, n=100) # integrate with fixed quad so i dont have to think about bounds of quadrature
    
    return s[0]
# plot of integrand    
# a,b,N = 0,5,250
# x = np.linspace(a,b,N)
# for i in [2,3,4]:
#     integrand_value = np.zeros(len(x))
#     integrand_value = function_original(x,i)
#     plt.plot(x,integrand_value,label = 'a = {}'.format(i))
#     plt.scatter(i-1,function_original(i-1,i))
# plt.xlabel('X')
# plt.ylabel('Intagrand Value')
# plt.title("Integrand Value between 0 and 5")
# plt.legend(loc="upper left");
# plt.savefig('integrand_values.png')

# gamma calculations
#gamma(3/2)-np.sqrt(np.pi)/2
#gamma(3)
#gamma(6)
#gamma(10)


# 

# 
# 
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




