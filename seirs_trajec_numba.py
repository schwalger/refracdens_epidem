# simulation meso SEIRS model
#
# copyright Tilo Schwalger, May 2024

from sim3_numba import sim
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
import time


M =1
N=2000000 * np.ones(M)
Nrecord = 1 # number populations to be recorded

T = 1400.0
dt_record = 1.0  #bin size of population activity in seconds, coarse-graining time step
dt = 0.1         # fine-grained simulation time step
Nbin = round(T/dt_record)
tt= dt_record * np.arange(Nbin)

seed=1          #seed for finite-size noise

tau = 50.0 
k=np.zeros((Nbin,M,M))
stepdown=int(700/dt_record)
k[:stepdown] = 2.4
k[stepdown:] = 0.6
c = 0.015
m = 1
vth = 2.
tauR = 20.0
tau_r = 1.0
tau_I = 10.0
a = 100.0
I_0 = 0.01
          
start = time.perf_counter()
nu, nu_N, I = sim(T, dt, dt_record, c, m, vth, tau, tauR, tau_r, tau_I, a, I_0, k, M, N, Nrecord, seed)
end = time.perf_counter()
print("Elapsed = {}s".format((end - start)))



plt.figure(1)
plt.clf()
plt.subplot(211)
plt.plot(tt,nu_N[0,:])
plt.plot(tt,nu[0,:])
plt.ylabel(r'$\nu$ (days$^{-1}$)')
plt.title('N=%d'%(N,))
plt.subplot(212)
plt.plot(tt,I[0,:])
plt.ylabel("I")
plt.xlabel("time [s]")
plt.subplots_adjust(top=0.88, bottom=0.235, left=0.235, right=0.9, hspace=0.2, wspace=0.2)
plt.savefig('meso_seirs_escapenoise_N%g.png'%(N,))
plt.show()

