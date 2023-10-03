

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import roots_legendre
from scipy.special import roots_hermite

def hermite(x,n):
    # returns nth hermite polynomial value at x point
    if n == 0:
        return 1
    elif n == 1:
        return 2*x
    else:
        return 2*x*hermite(x,n-1) - 2*(n-1)*hermite(x,n-2)
def psi(x,n):
    val1 = 1/(2**n*math.factorial(n))**.5/(math.pi**.25) #values out front
    val2 = np.exp(-.5*x**2) # exponential
    return val1*val2*hermite(x,n) 

def x_transform(x): 
    # this function changes x > z/(1-z^2) so we can integrate between -1 and 1
    # instead of between -inf and positive inf
    return x/(1-x**2)
def transformed_varfunction(x,n):
    # the first two terms are from going from dx > dz, second two terms are transformed second moment of psi
    return (1+x**2)/(1-x**2)**2*(x_transform(x))**2*psi(x_transform(x),n)**2

def calc_rms_legendre(num_pts,n):
    # calculates variance of psi, requires boundaries to be -inf, pos inf
    roots, weights = roots_legendre(num_pts)
    s=0 #sum
    for k in np.arange(num_pts):
        s+= weights[k]*transformed_varfunction(roots[k],5)
    return np.sqrt(s)

def hermitefunction(x,n):
    #hermite gaussian quadrature integrates functions multiplied by e^-x between -inf and positive inf
    return x*x/(2**n*math.factorial(n))/np.sqrt(math.pi)*hermite(x,n)**2

def calc_rms_hermite(num_pts,n):
    roots, weights = roots_hermite(num_pts)
    s=0 #sum
    for k in np.arange(num_pts):
        s+= weights[k]*hermitefunction(roots[k],5)
    return np.sqrt(s)

# run calculation of it using two different methods
print(calc_rms_legendre(100,5))
print(calc_rms_hermite(10,5))

# plots comment or uncomment to run different plots

# n = np.arange(0,4)
# x_vals = np.linspace(-4,4,1000)
# for i in n:
#     plt.plot(x_vals,psi(x_vals,i))
# plt.xlabel('Position')
# plt.ylabel('Psi')
# plt.legend(['n = 0', 'n =1', 'n =2', 'n =3', 'n =4']); # different ns of the harmonic oscillator
# plt.title('Four Lowest Wavefunctions of Quantum HO')
# plt.savefig('four_wavefunctions.png')


# x_vals = np.linspace(-10,10,1000)
# n = (30) 
# plt.plot(x_vals,psi(x_vals,n));
# plt.xlabel('Position')
# plt.ylabel('Psi')
# plt.title('Wavefunction for n = 30 for Quantum HO')
# plt.savefig('n30psi.png')

# n = 40
# vals_hermite = np.zeros(n-1)
# vals_legendre = np.zeros(n-1)
# for i in np.arange(1,n):
#     vals_hermite[i-1] = calc_rms_hermite(i,10)
#     vals_legendre[i-1] = calc_rms_legendre(i,10)
# plt.plot(np.arange(1,n),vals_hermite)
# plt.plot(np.arange(1,n),vals_legendre)
# plt.xlabel('Number of sampling points')
# plt.ylabel('Integral Value')
# plt.title('Hermite quadrature vs. Legendre Quadrature')
# plt.legend(['Hermite method', 'Legendre method'])
# plt.savefig('hermiteVslegendre.png')





