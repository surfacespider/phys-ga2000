

import numpy as np
import matplotlib.pyplot as plt
import scipy
from numpy import copy
from scipy import linalg

#initialize constants and grid
h = 10**-6 #seconds
hbar = 1
L = 1 #nano meters
N = 10001
a = L/(N-1)
x_0 = L/2
m = 1 #9.109 * 10**-31 # kg
sigma = L/10
k = 50/L

#initialize wave function
xs = np.linspace(0,L,N)
psi = np.exp(-1*((xs-x_0)**2)/(2*sigma**2))*np.exp(1j*k*xs)
psi[0] = 0
psi[-1] = 0

a1 = 1+h*1j*hbar/(2*m*a**2)
a2 = -h*1j*hbar/(4*m*a**2)
b1 = 1 - 1j*h*hbar/(2*m*a**2)
b2 = h*1j*hbar/(4*m*a**2)


#make our banded matrix
center = np.full(N,a1)
upperband = np.full(N,a2)
lowerband = np.full(N,a2)
upperband[0] = 0
lowerband[-1] = 0

A = np.asarray([upperband,center,lowerband])

psi_data = []
for t in np.arange(0, .05,h):
    psi_data.append(psi.copy()) #save data
    
    #make a copy because otherwise youre doing weird pointer stuff that doesnt do what you want it to do
    #psi_calc = psi.copy()
    psi_calc = np.concatenate(([0],psi,[0])) #add outside for vectorized neighbor interaction
    v = b1*psi_calc[1:-1]+ b2*(psi_calc[2:]+psi_calc[:-2]) #neigbor interaction
    
    psi = scipy.linalg.solve_banded((1,1),A,v) #solve the matrix
    #psi = banded(A,v,1,1)
    psi[0] = 0 #set boundary condition again
    psi[-1]= 0
    
psi_values = np.asarray(psi_data)

plt.plot(xs, psi_values[0].real,linewidth = 1,label = 'psi, t = 0')
plt.plot(xs, psi_values[4000].real,linewidth = 1, label= 'psi, t = ' + str(4000*h))
plt.plot(xs, psi_values[31000].real,linewidth = 1, label = 'psi, t = ' + str(31000*h))
plt.xlabel('Position in box (nm)')
plt.ylabel('Amplitude')
plt.legend()
plt.title('real part of wave function')
plt.savefig('wavefunction.png')

