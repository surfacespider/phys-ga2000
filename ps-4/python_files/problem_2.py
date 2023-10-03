#!/usr/bin/env python
# coding: utf-8


import numpy as np
import scipy
import matplotlib.pyplot as plt

m = 1 #mass

amplitude_array = np.linspace(.1,2,50)
periods_array = np.zeros(len(amplitude_array))

for i,a in enumerate(amplitude_array):
    func = lambda x: 1/np.sqrt(a**4-x**4) #redefine function based on initial amplitude
    integral = scipy.integrate.fixed_quad(func,0,a,n=20) #calculate amplitude integral
    periods_array[i] = integral[0]*np.sqrt(8*m) #mass equals 1 but included anyway

plt.plot(amplitude_array,periods_array);
plt.xlabel('Initial Amplitude (meters)')
plt.ylabel('Period (sqrt(kg) per meter)')
plt.title('Anharmonic Oscillator Period for differing amplitudes');
plt.savefig('anharmonic_period.png')

