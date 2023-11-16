#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
def log_likelihood(beta, xs, ys):
    beta_0 = beta[0]
    beta_1 = beta[1]
    epsilon = 1e-16
    l_list = [ys[i]*np.log(p(xs[i], beta_0, beta_1)/(1-p(xs[i], beta_0, beta_1)+epsilon)) 
              + np.log(1-p(xs[i], beta_0, beta_1)+epsilon) for i in range(len(xs))]
    ll = np.sum(np.array(l_list), axis = -1)
    return -ll # return log likelihood
def p(x, beta_0, beta_1):
    return 1/(1+np.exp(-(beta_0+beta_1*x)))
data = pd.read_csv('survey.csv')  
xs = data['age'].to_numpy()
ys = data['recognized_it'].to_numpy()
x_sort = np.argsort(xs)
xdata = xs[x_sort]
ydata = ys[x_sort]



#Fits
pstart = [0,0]

errFunc = lambda p, x, y: log_likelihood(p,x,y) - y

# Covariance matrix of parameters
def Covariance(hess_inv, resVariance):
    return hess_inv * resVariance

#Error of parameters
def error(hess_inv, resVariance):
    covariance = Covariance(hess_inv, resVariance)
    return np.sqrt( np.diag( covariance ))

result = optimize.minimize(lambda p,x,y: log_likelihood(p,x,y), pstart,  args=(xdata, ydata))
hess_inv = result.hess_inv # inverse of hessian matrix
var = result.fun/(len(ydata)-len(pstart)) 
dFit = error( hess_inv,  var)
print('Optimal parameters and error:\n\tp: ' , result.x, '\n\tdp: ', dFit)
print('Covariance matrix of optimal parameters:\n\tC: ' , Covariance( hess_inv,  var))


plt.scatter(xs,p(xs,result.x[0],result.x[1]),s = 4,label='Logistic')
plt.scatter(-result.x[0]/result.x[1],.5)
plt.scatter(xs,ys,s=4,label='Data')
plt.xlabel('age')
plt.legend()
plt.title('Data and fit')
plt.ylabel('yes/no?');
#plt.savefig('logistic.png')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




