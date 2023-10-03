#!/usr/bin/env python
# coding: utf-8


import numpy as np
import scipy
import matplotlib.pyplot as plt
# aluminum
rho = 6.022*10**28 #number density (inverse meters cubed)
debye = 428 # debye temp in kelvin
volume = 1000/ (100**3) # m^3
#constants
boltzmann = 1.38*10**-23
cv_constant = 9*volume*rho*boltzmann/(debye**3) #constant out front after making integral unitless


#functions
def funct(x = None):
    # our function to integrate to get the heat capacity of a solid
    return (x**4*np.exp(x))/((np.exp(x)-1)*(np.exp(x)-1))
def Cv(T,quad_num):
    #uses scipy to integrated these using gaussian quadrature
    integral = scipy.integrate.fixed_quad(funct,0,debye/T,n=quad_num)
    return integral[0] 

#calculation of cv at different temps
temp_array = np.linspace(5,500,100)
cv_array = np.zeros(len(temp_array))
for i,temp in enumerate(temp_array):
    cv_array[i] = Cv(temp,50)*cv_constant*(temp**3)


#plotting section comment or uncomment to plot these
plt.plot(temp_array,cv_array);
plt.xlabel('Temperature (Kelvin)');
plt.ylabel('Heat Capacity (Joules per Kelvin)');
plt.title('Heat Capacity of 1000 cm^3 of Aluminum Vs. Temperature');
#plt.savefig('heat_capacity.png') 

#convergence of the integral plots

#convergence = np.zeros(7)
#for i,n in enumerate(np.linspace(10,70,7)):
#    convergence[i] = Cv(1.5,n)
#plt.plot(np.linspace(10,70,7),convergence);
#plt.xlabel('Number of points')
#plt.ylabel('Integration Value')
#plt.title('Convergence of integration with increasing number of points')
#plt.savefig('convergence_plot.png')




