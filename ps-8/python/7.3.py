

import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
piano = np.genfromtxt("piano.txt", delimiter=" ")
trumpet = np.genfromtxt("trumpet.txt", delimiter=" ")
N = int(len(piano))
frequency = fft.fftfreq(N, d=1/44100)[:(N+2)//2]
trumpet_fourier = np.fft.rfft(trumpet)
piano_fourier = np.fft.rfft(piano)


max_y_piano = max(piano_fourier)  # Find the maximum y value
max_x_piano = frequency[piano_fourier.argmax()]

max_y_trumpet = max(trumpet_fourier)  # Find the maximum y value
max_x_trumpet = frequency[trumpet_fourier.argmax()]

print('principle component of piano is', max_x_piano, 'HZ')
print('principle component of trumpet is', max_x_trumpet, 'HZ')
# #plot piano
# plt.plot(np.arange(N)/44100,piano)
# plt.xlabel('time (seconds)')
# plt.ylabel('waveform')
# plt.title('piano waveform')
# #plt.savefig('piano.png')

# #plot trumpet
# plt.plot(np.arange(N)/44100,trumpet,linewidth = .1)
# plt.xlabel('time (seconds)')
# plt.ylabel('waveform')
# plt.title('trumpet waveform')
# plt.savefig('trumpet.png')

# # plot piano frequencies
# plt.plot(frequency,np.abs(piano_fourier))
# plt.xlim(0,3000)
# plt.ylabel('amplitude')
# plt.xlabel('frequency (HZ)')
# plt.title('piano frequencies')
# #plt.savefig('piano_freq.png')

# # plot plot trumpet freq
# plt.plot(frequency,np.abs(trumpet_fourier))
# plt.xlim(0,5000)
# plt.ylabel('amplitude')
# plt.xlabel('frequency (HZ)')
# plt.title('trumpet frequencies')
# #plt.savefig('trumpet_freq.png')






