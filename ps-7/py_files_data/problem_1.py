#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import time

def parabolic_step(a, b, c):
    """returns the minimum of the function as approximated by a parabola"""
    fa = f(a)
    fb = f(b)
    fc = f(c)
    #denom = (b - a) * (fb - fc) - (b -c) * (fb - fa)
    denom = (b - a) * (fb - fc) - (b -c) * (fb - fa)
    numer = (b - a)**2 * (fb - fc) - (b -c)**2 * (fb - fa)
    # If singular, just return b 
    if(np.abs(denom) < 1.e-15):
        x = b
    else:
        x = b - 0.5 * numer / denom
    return(x)

def golden(a, b, c):
    gsection = (3. - np.sqrt(5)) / 2
        # Split the larger interval
    if((b - a) > (c - b)):
        x = b
        b = b - gsection * (b - a)
    else:
        x = b + gsection * (c - b)
    fb = f(b)
    fx = f(x)
    if(fb < fx):
        c = x
    else:
        a = b
        b = x 
    return(a,b,c)

def brent(a,b):
    #xgrid
    tol = 1e-6
    niter = 10_000
    i = 0
    if abs(f(a)) < abs(f(b)): # is function taller on left or right
        a, b = b, a #swap bounds
    c = b+a
    d = 100
    flag = True
    err = abs(b-a) #distance between b and a
    err_list, b_list = [err], [b] #b_list is location of b  
    
    while (err > tol and i < niter): #while a != b
        s = parabolic_step(a, b, c)
        if ((s >= b))\
            or ((flag == False) and (abs(s-b) >= abs(c-d)))\
            or ((flag == True) and (abs(s-b) >= abs(b-c))):
            a,b,c = golden(a,b,c)
            flag = True # did golden
        else:
            flag = False #did parabolic
            c, d = b, c # d is c from previous step
            a = s
            
            
        if abs(f(a)) < abs(f(b)):
            a, b = b, a #swap if needed
        err = abs(b-a) #update error to check for convergence
        err_list.append(err)
        b_list.append(b)
        #print(flag)
        i+= 1
    return err_list, b_list
def plot(b_list, err_list):
    log_err = [np.log10(err) for err in err_list]
    fig, axs = plt.subplots(2,1, sharex=True)
    ax0, ax1 = axs[0], axs[1]
    #plot root
    ax0.scatter(range(len(b_list)), b_list, marker = 'o', facecolor = 'red', edgecolor = 'k')
    ax0.plot(range(len(b_list)), b_list, 'r-', alpha = .5)
    ax1.plot(range(len(err_list)), log_err,'.-')
    ax1.set_xlabel('number of iterations')
    ax0.set_ylabel(r'$x_{min}$')
    ax1.set_ylabel(r'$\log{\delta}$')
    plt.savefig('convergence.png')
    

f = lambda x: (x-.3)**2*np.exp(x)
errors, bs = brent(.1,2)
plot(bs,errors)
optimize.brent(f, brack=(.1, 2),tol = 10**-6 ,full_output=1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




