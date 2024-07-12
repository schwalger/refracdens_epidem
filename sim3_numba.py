# Simulation routine for mesoscopic multi-population model
# copyright Tilo Schwalger, May 2024

import numpy as np
from math import exp, expm1, trunc
#from numpy.random import binomial
from numba import njit




@njit
def sim(T, dt, dt_rec, c, mm, vth, tauV, tauR, tau_r, tau_I, a, I_0, k, M, N, Nrecord, seed):

    def hazard(u, ts, c, m, tauR, vth):
        if ts>=tauR:
            return c*np.maximum(0,u/vth)**m             # hazard rate
        else:
            return 0.

    def Pfire(u, ts, c, m, tauR, vth, lambda_old, dt):
        lam = hazard(u, ts, c, m, tauR, vth)
        Plam = 0.5 * (lambda_old + lam) * dt
        if (Plam>0.01):
            Plam = -expm1(-Plam)
        return Plam, lam


    n_I = round(tau_I/dt)
    tau = tauV * np.ones(M)
#    tau = tauV * (np.zeros(M)+1)
    
    #quantities to be recorded
    Nsteps = round(T/dt)
    Nsteps_rec = round(T/dt_rec)
    Nbin = round(dt_rec/dt) #bin size for recording 
    nu = np.zeros((Nrecord, Nsteps_rec))
    nu_N = np.zeros((Nrecord, Nsteps_rec), dtype=np.float32)
    Irec = np.zeros((Nrecord, Nsteps_rec))

    #initialization
    L = np.zeros(M, dtype=np.int64)
    for i in range(M):
        L[i] = round((5 * tau[i] + tauR) / dt) + 1 #history length of population i
    Lmax = np.max(L)
    S = np.ones((M, Lmax))
    u = 0 * np.ones((M, Lmax), dtype=np.float32)
    n = np.zeros((M, Lmax))
    lam = np.zeros((M, Lmax))
    x = (1-I_0) * np.ones(M)
    z = np.zeros(M)
    I = I_0 * np.ones(M)
    for i in range(M):
        n[i,L[i]-1] = I_0 # initial condition
        u[i,L[i]-1] = vth


    ts= dt* np.arange(Lmax)
    D = 4*a*np.exp(-ts/tau_r) - a*np.exp(-0.25*ts/tau_r)
            
    h = np.zeros(M) #susceptibles have V(0)=0 
    lambdafree = np.zeros(M)
    for i in range(M):
        lambdafree[i]=hazard(h[i], ts[-1], c, mm, tauR, vth)

        
    #begin main simulation loop
    for ti in range(Nsteps):
        t = dt*ti
        i_rec = trunc(ti/Nbin)

        Input = np.zeros(M)
        Input = k[i_rec] @ I 
        
        for i in range(M):
            x[i] += S[i,0] * n[i, 0] #fraction of individuals with tstar > history length L
            z[i] += (1 - S[i,0]) *  S[i,0] * n[i, 0]
            h[i] += dt * (Input[i] - h[i] / tau[i])
            Plamfree, lambdafree[i] = Pfire(h[i], ts[-1], c, mm, tauR, vth, lambdafree[i], dt)
            W = Plamfree * x[i]
            X = x[i]
            Z = z[i]
            Y = Plamfree * z[i]
            z[i] = (1-Plamfree)**2 * z[i] + W
            x[i] -= W
            
            for l in range(1,L[i]):
                u[i, l-1] = u[i,l] + dt * (Input[i] - u[i,l] / tau[i] + D[L[i]-l])
                Plam, lam[i,l-1] = Pfire(u[i,l-1], ts[L[i]-l], c, mm, tauR, vth, lam[i,l], dt)
                m = S[i, l] * n[i,l]    # m = rho * dt
                v = (1 - S[i,l]) * m
                W += Plam * m
                X += m
                Y += Plam * v
                Z += v
                S[i,l-1] = (1 - Plam) * S[i, l]
                n[i,l-1] = n[i,l]
   
            if (Z>0):
                PLAM = Y/Z
            else:
                PLAM = 0
            
            nmean = max(0, W +PLAM * (1 - X))
            if nmean>1:
                nmean = 1

            n[i, L[i]-1] = np.random.binomial(int(N[i]), float(nmean)) / N[i] # population activity (fraction of neurons spiking)
            
            I[i] = np.sum(n[i, L[i] - n_I : L[i]])  # fraction of infected people

            if (i <= Nrecord):
                nu[i, i_rec] += nmean
                nu_N[i,i_rec] += n[i,L[i]-1]
                Irec[i, i_rec] += I[i]
                
        # if np.mod(ti+1,Nsteps/100) == 0:  #print percent complete
        #     print("\r%d%% "%(np.round(100*ti/Nsteps),))

        
    nu /= (Nbin * dt)
    nu_N  /= (Nbin * dt)
    Irec /= Nbin


    

    print("\r")
    
    return nu, nu_N, Irec
