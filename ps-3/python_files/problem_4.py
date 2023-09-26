#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import random
import matplotlib.pyplot as plt
random_nums = np.random.uniform(0,1,1000)
tau = 3.053*60 # decay rate
mu = np.log(2)/(tau) #change log 2 to natural log

exponential_decay = -np.log(1-random_nums)/mu #exponential weighting
exponential_decay_sorted = np.sort(exponential_decay)

#reverse so it shows decay
reversed = exponential_decay_sorted[::-1]

#plots
plt.plot(reversed,np.arange(1000));
plt.plot(tau,500,'x')
plt.xlabel('Time (seconds)')
plt.ylabel('Number of atoms')
plt.legend(['Exponential decay of Tl','Location of expected halved amount'])
plt.title('Simulated decay of 1000 Thallium atoms')
plt.savefig('sim_decay_thallium_p4')

