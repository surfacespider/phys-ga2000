#!/usr/bin/env python
# coding: utf-8

# In[45]:


import numpy as np
import matplotlib.pyplot as plt
def gaussian(x,stdv):
    value = (1/(stdv*np.sqrt(2*np.pi))*np.exp(-(x*x)/(2*stdv*stdv)))
    return value

xvals = np.linspace(-10,10,1000)
stdv = 3
plt.plot(xvals,gaussian(xvals,stdv));
plt.xlabel('X');
plt.ylabel('Y');
plt.title("Gaussian from -10 to 10");
plt.savefig('gaussian.png')


# In[ ]:





# In[27]:





# In[ ]:





# In[ ]:





# In[ ]:




