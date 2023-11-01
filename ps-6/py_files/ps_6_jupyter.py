#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import astropy.io
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy.linalg as linalg
from time import time
#import data and initialize
hdu_list = astropy.io.fits.open('specgrid.fits')
logwave = hdu_list['LOGWAVE'].data # 4001 wavelengths of light
flux = hdu_list['FLUX'].data # 9713 galaxies
wave = 10**logwave/10
N_gal = 9713
N_lambda = 4001

#i tried doing this without for loops but it was giving me wrong result
# flux_sum = np.sum(flux, axis = 1)
# norm_flux = flux/np.tile(flux_sum, (np.shape(flux)[1], 1)).T
# mean_spectra = np.mean(norm_flux, axis=1)
# residuals = flux_normalized-np.tile(mean_spectra, (np.shape(flux)[1], 1)).T

# normalize data and find residuals
mean_spectra = np.zeros((N_gal,N_lambda))
norm_flux = np.zeros((N_gal,N_lambda))
residuals = np.zeros((N_gal,N_lambda))
normalization = np.zeros(N_gal)
for i in np.arange(N_gal):
    normalization[i] = np.sum(flux[i][:])
    norm_flux[i,:] = flux[i,:]/normalization[i]
    #norm_flux[i][:] = flux[i][:]/np.sum(flux[i][:])
    
    mean_spectra[i] = np.mean(norm_flux[i,:])

    residuals[i,:] = norm_flux[i,:]-mean_spectra[i]
    
# plot normalized or original flux
# for i in np.arange(5):
#     plt.plot(wave,flux[i][:],linewidth = '.4')
#     plt.plot(wave,residuals[i][:],linewidth = '.4')
#     plt.xlabel('wavelength (nm)')
#     plt.ylabel('flux')
#     plt.title('flux vs wavelength ')
    #plt.savefig('unnormalizedflux.png')

#using RR.T = C to find covariance matrix and eigenvectors using np.linalg
R = np.matrix(residuals)
#dt = time()
C = R.T*R
eigenvalues_linalg, eigenvectors_linalg = linalg.eig(C)
#dt = time() - dt
#print(dt) #took my computer 20 seconds

# using svd on R to find eigenvectors
#dt = time()
(U, w, VT) = np.linalg.svd(R) 
#dt = time() - dt
#print(dt) # took my computer 43 seconds
eigenvectors_svd = VT.T
eigenvalues_svd = w**2

#plot some lin alg covariance eigenvectors
# for i in np.arange(10):
#     plt.plot(wave, eigenvectors_linalg[:, i])
# plt.xlabel('wave')
# plt.ylabel('Eigenvector')
# plt.title('Eigenvectors via np.linalg')
# plt.savefig('linalgeigenvectors.png')

#plot some svd eigenvectors
# for i in np.arange(10):
#     plt.plot(wave, eigenvectors_svd[:, i])
# plt.xlabel('wave')
# plt.ylabel('Eigenvector')
# plt.title('Eigenvectors via SVD')
# plt.savefig('svdeigenvectors.png')

# (e)compare data
# for i in np.arange(10):
#     plt.plot(eigenvectors_linalg[:,i],eigenvectors_svd[:,i])
# plt.xlabel('linalg eigenvectors')
# plt.ylabel('svd eigenvectors')
# plt.title('svd vs lingalg eigenvectors')
#plt.savefig('linalgVSsvd.png')

# (f) condition numbers
# linalg_cond = max(eigenvalues_linalg)/min(eigenvalues_linalg)
# svd_cond = max(eigenvalues_svd)/min(eigenvalues_svd) 
# ratio = svd_cond/linalg_cond
# print('condition num C', linalg_cond)
# print('condition num SVD', svd_cond)
# print('ratio', ratio)


# (g) fitting to actual data

def pca_weights(num_spectra,data,eigenvecs):
    vecs = eigenvecs[:,:num_spectra] #how many eigenvectors do we want to use
    return np.dot(vecs.T,data.T) #return matrix of eigenvector weights 

def pca_reduced_data(num_spectra,data,eigenvecs):
    vecs = eigenvecs[:,:num_spectra]
    #return weights dotted into eigenvectors get back reduced data
    return np.dot(vecs,pca_weights(num_spectra,data,eigenvecs)).T 

num_weights = 100
pca_cs = pca_weights(num_weights,residuals,eigenvectors_linalg)
pca_vectors = pca_reduced_data(num_weights,residuals,eigenvectors_linalg)

unnormalized_pca = np.zeros((N_gal,N_lambda))
#i was doing this all without for loops but it was giving weird results so back to loops
for i in np.arange(N_gal):
    unnormalized_pca[i,:] = np.asarray(pca_vectors)[i,:] + mean_spectra[i,:]
    unnormalized_pca[i,:] = unnormalized_pca[i,:]*normalization[i]



# plt.plot(logwave, flux[1,:], label = 'original data')
# plt.plot(logwave, unnormalized_pca[1,:], label = 'l = 5')
# plt.xlabel('log wave')
# plt.ylabel('flux')
# plt.title('pca fitting vs original data')
# plt.legend()
#plt.savefig('5eigenvectorpca.png')

# (i) squared fractional residuals 

least_squares = np.zeros(20)
for i in np.arange(20):
    pca_vectors_least = np.asarray(pca_reduced_data(i,residuals,eigenvectors_linalg))
    least_squares[i] = np.sum((pca_vectors_least-residuals)**2)
fractional_error = np.sqrt(least_squares[19]) # ~ 10 percent normalized to 1 dont have to divide


# plt.plot(np.arange(20),least_squares);
# plt.xlabel('Number of eigenvectors used')
# plt.ylabel('sum of least squares')
# plt.title('Decrease in error with increasing number Principal Components')
# plt.savefig('errorreduce.png')


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





# In[37]:





# In[ ]:





# In[ ]:





# In[41]:


# (h) coefficient plots
# cs_array = np.asarray(pca_cs)
# plt.scatter(cs_array[0,:],cs_array[7,:], s = .7)
# plt.xlabel('C_0')
# plt.ylabel('C_1')
# plt.title('C_1 vs C_0')
# # plt.savefig('c0vsc1.png')

plt.scatter(cs_array[0,:],cs_array[2,:], s = .7)
plt.xlabel('C_0')
plt.ylabel('C_2')
plt.title('C_0 vs C_2')

plt.savefig('c0vsc2.png')


# In[ ]:





# In[44]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




