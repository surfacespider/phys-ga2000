#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import time
import matplotlib.pyplot as plt

def matrix_mult(size):
    for i in range(size):
        for j in range(size):
            for k in range(size):
                c[i,j] += A[i,k]*B[k,j]
def matrix_dot(size,A,B):
    np.dot(A,B)

sizes = np.array([5,10,20,40,60,80,100,120,140,160,180])

#timing size increase with for loops
time_loop = np.zeros(len(sizes))
for i, N in enumerate(sizes):
    c = np.zeros([N,N])
    A = np.ones([N,N])
    B = np.ones([N,N])
    
    start_time = time.time()
    matrix_mult(N)
    end_time = time.time()
    time_loop[i] = end_time-start_time

#timing size increase with np.dot
time_dot = np.zeros(len(sizes))
for i, N in enumerate(sizes):
    c = np.zeros([N,N])
    A = np.ones([N,N])
    B = np.ones([N,N])
    
    start_time = time.time()
    matrix_dot(N,A,B)
    end_time = time.time()
    
    time_dot[i] = end_time-start_time
    
plt.plot(sizes,time_loop, 'x');
plt.plot(sizes,time_dot, '^')
plt.plot(sizes,sizes*sizes*sizes/(100**3)*.5)
plt.xlabel('Size of one side of array')
plt.ylabel('Time (seconds)')
plt.legend(['For loop times','np.dot times', 'Normalized N^3 dependency']);
plt.savefig('runtime.png')




