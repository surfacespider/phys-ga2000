#!/usr/bin/env python
# coding: utf-8

# In[125]:


import numpy as np
import matplotlib.pyplot as plt

def mandelbrot(x, y,num_iterations): #checks if this is in the set
    c = complex(x, y)
    z = 0
    for i in range(1, num_iterations):
        if abs(z) > 2:
            return False
        z = z * z + c
    return True

size = 1000 # number of points wide and high we are looking in
size_squared = size**2

x = np.linspace(-2, 2, size+1)
y = np.linspace(-2, 2, size+1)

num_iterations = 100
in_set = False

xvals = np.zeros(size_squared)
yvals = np.zeros(size_squared)

i = 0
#i tried to make this work not with for loops for so long
for cx in x: 
    for cy in y:
        in_set = mandelbrot(cx,cy,num_iterations)
        if in_set: #if in set add to this array
            xvals[i] = cx
            yvals[i] = cy
            i = i+1
    
plt.scatter(xvals[0:i], yvals[0:i],color = 'black',s=.05)
plt.title("Mandelbrot Set")
plt.xlabel("Real part of c")
plt.ylabel("Imaginary part of c")
plt.savefig('mandelbrot_adam.png')
