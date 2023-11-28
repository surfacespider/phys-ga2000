
# $\frac{dx}{dt} = \sigma (y - x)   \frac{dy}{dy} = rx - y - xz    \frac{dz}{dt} = xy - bz$


import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

sigma = 10
r = 28
b = 8/3
# 0 < t < 50
def dwdt(t, w):
    x, y, z = w
    xdot = sigma*(y-x)
    ydot = r*x - y -x*z
    zdot = x*y-b*z
    return np.array([xdot, ydot, zdot])
results = scipy.integrate.solve_ivp(dwdt, [0., 50.], [0., 1., 0.], method='DOP853')

# plt.plot(results.t, results.y[1, :], label="Y") #plot of time vs y value 
# plt.ylabel('y')
# plt.xlabel('Time')
# plt.title('y values')
# plt.savefig('yvals.png')
# plt.plot(results.y[0,:],results.y[2,:]) #plot of x values vs z values
# plt.xlabel('x')
# plt.ylabel('z')
# plt.title('strange attractor')
# plt.savefig('strange.png')





