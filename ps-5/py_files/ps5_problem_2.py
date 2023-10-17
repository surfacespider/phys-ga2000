#!/usr/bin/env python
# coding: utf-8

# In[126]:





# In[180]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as linalg
from scipy.fft import fft, fftfreq

df = pd.read_csv('signal.dat', delimiter='|')
#df.dtypes
time_pds = df["               time "]
signal_pds = df["                signal "]
signal = signal_pds.values
time = time_pds.values

# plot signal
# plt.scatter(time,signal,s=2);
# plt.xlabel('time')
# plt.ylabel('signal')
# plt.title('signal vs time')
# plt.savefig('signal.png')

# third order fit
time_scaled = time/10**9 #scale time down bc dont work with such large numbers
A = np.zeros((len(test), 4))
A[:, 0] = 1.
A[:, 1] = time_scaled 
A[:, 2] = time_scaled**2
A[:, 3] = time_scaled**3

(u, w, vt) = np.linalg.svd(A, full_matrices=False)
ainv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())
c = ainv.dot(signal)
ym = A.dot(c) 
# plt.scatter(time_scaled, signal, s =2 )
# plt.scatter(time_scaled, ym)
# plt.xlabel('time/t_max')
# plt.ylabel('signal')
# plt.title('3rd Order Polynomial Fit')
#plt.savefig('3rd_polynomail.png')
#residuals_cubic = abs(signal-ym)

#21st order polynomial
A = np.zeros((len(test), 22))
A[:, 0] = 1.
A[:, 1] = time_scaled 
A[:, 2] = time_scaled**2
A[:, 3] = time_scaled**3
A[:, 4] = time_scaled**4
A[:, 5] = time_scaled**5
A[:, 6] = time_scaled**6
A[:, 7] = time_scaled**7
A[:, 8] = time_scaled**8
A[:, 9] = time_scaled**9
A[:, 10] = time_scaled**10
A[:, 11] = time_scaled**11
A[:, 12] = time_scaled**12
A[:, 13] = time_scaled**13
A[:, 14] = time_scaled**14
A[:, 15] = time_scaled**15
A[:, 16] = time_scaled**16
A[:, 17] = time_scaled**17
A[:, 18] = time_scaled**18
A[:, 19] = time_scaled**19
A[:, 20] = time_scaled**20
A[:, 21] = time_scaled**21


(u, w, vt) = np.linalg.svd(A, full_matrices=False)
ainv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())
c = ainv.dot(signal)
ym = A.dot(c) 
# plt.scatter(time_scaled, signal, s =2 )
# plt.scatter(time_scaled, ym)
# plt.xlabel('time/t_max')
# plt.ylabel('signal')
# plt.title('21st Order Polynomail Fit')
#plt.savefig('21stpolynomial.png')
#max(w)/min(w) #condition number

#remove linear trend if you want
# A = np.zeros((len(time), 2))
# A[:, 0] = 1.
# A[:, 1] = time
# (u, w, vt) = np.linalg.svd(A, full_matrices=False)
# ainv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())
# c = ainv.dot(signal)
# ym = A.dot(c) 
# signal_flat = signal - ym
# plt.scatter(time_scaled,signal_flat,s=2)

#Lomb-Scargle calculation
# this fourier transforms your data and gives you the actual primary frequency but 
#if yuo want you can just look at the data and guess that omega ~ 7*2pi 
#bc peak to peak is ~ .15 t
N = len(time_scaled)
T = time_scaled[N-1]-time_scaled[0]
yf = fft(signal_flat)
xf = fftfreq(N, T)[:N//2]
omega = 2*np.pi*xf[np.argsort(2.0/N * np.abs(yf[0:N//2]))[::-1]][0] 

A = np.zeros((len(time), 4))
A[:, 0] = 1.
A[:, 1] = time_scaled
A[:, 2] = np.cos(omega*2*np.pi*time_scaled)
A[:, 3] = np.sin(omega*2*np.pi*time_scaled)


(u, w, vt) = np.linalg.svd(A, full_matrices=False)
ainv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())
c = ainv.dot(signal)
ym = A.dot(c) 

plt.plot(time_scaled, signal, '.', label='data')
plt.plot(time_scaled, ym, '.', label='model')
plt.xlabel('t/t_max')
plt.ylabel('y')
plt.title('Signal Fitted with Sin and Cos')
plt.legend();

#plt.savefig('cossinPlot.png')
#w[0]/w[-1] # condition number


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




