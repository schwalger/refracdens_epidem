import copy
import math
import numpy as np
import random
from PopulationEq_Escape import Hazard


#{ Parameters: }
class NeuronProperties:
    tau_d: float
    tau_r: float
    tau_a: float
    VT: float
    Vreset: float
    tau: float
    I_disease_ampl :float
    I_disease_ratio :float
    If_CBRD: int
    If_escape: int
    c_escape: float
    m_escape: float
    sgmV_escape: float
    If_A_AB_or_power_escape: int

#{ Variables: }
class NeuronVariables:
    V: float
    DVDt: float
    tLastSpike: float

#{==================================================================}
class TNrn:
    NP: NeuronProperties
    NV: NeuronVariables

    def __init__(self, NP_):
        # Properties:
        self.NP = copy.deepcopy(NP_)
        # Variables:
        self.NV = NeuronVariables()

    def InitialConditions(self):
        self.NV.V = 0
        self.NV.DVDt = 0.0
        self.NV.tLastSpike = -888

    def ResetToThreshold(self, t):
        # Setting V equal to VT
        self.NV.V = self.NP.VT
        self.NV.tLastSpike = t
        self.NV.DVDt = 0

    # ************************************************************************************************
    def MembranePotential(self, uu, t, dt):
    # Solves eq. for the potential "V" ***************************************************************
        ratio = self.NP.I_disease_ratio # 4
        I_disease_ = self.NP.I_disease_ampl * (ratio*np.exp(-(t - self.NV.tLastSpike) / self.NP.tau_r) \
                     - np.exp(-(t - self.NV.tLastSpike) / (self.NP.tau_r*ratio)))
        # Update voltage ****************************************
        self.NV.DVDt = -self.NV.V / self.NP.tau + I_disease_ + uu
        # *******************************************************
        if (self.NP.I_disease_ampl != 0) or (t >= self.NV.tLastSpike + self.NP.tau_a):
            self.NV.V = self.NV.V + dt * self.NV.DVDt
        # *******************************************

        # Check for spikes
        if self.NP.If_escape == 0:
            if (self.NP.If_CBRD == 0) and (self.NV.V > self.NP.VT) and (t - self.NV.tLastSpike > self.NP.tau_d):
                if (self.NP.I_disease_ampl == 0):
                    self.NV.V = self.NP.Vreset
                    self.NV.DVDt = 0
                    self.NV.tLastSpike = t
                else:  # if I_disease is turned on
                    dV = self.NV.DVDt * dt
                    if (self.NV.V - dV < self.NP.VT):  # upward
                        self.NV.tLastSpike = t
        else: # ESCAPE-noise
            if self.NP.If_CBRD == 0:
                match self.NP.If_A_AB_or_power_escape:
                    case 1:
                        # A-function as a self-similar solution of FP-eq. for LIF with white noise and stationary stimulus
                        H_ = Hazard(self.NP.sgmV_escape,         -1.0, self.NP.tau, self.NV.V, self.NP.VT, t - self.NV.tLastSpike, self.NP.tau_d)
                    case 2:
                        # H = A + B
                        H_ = Hazard(self.NP.sgmV_escape, self.NV.DVDt, self.NP.tau, self.NV.V, self.NP.VT, t - self.NV.tLastSpike, self.NP.tau_d)
                    case 3:
                        # power-function
                        H_ = self.NP.c_escape * max(0, self.NV.V/self.NP.VT) ** self.NP.m_escape
                        if (t - self.NV.tLastSpike <= self.NP.tau_d):
                            H_ = 0

                prob_ = 1 - np.exp(- H_ * dt / 1e3) # (t - self.NV.tLastSpike) / 1e3
                rnd_ = np.random.random() #np.random.exponential(1)
                if prob_ > rnd_:
                    if (self.NP.I_disease_ampl == 0):
                        self.NV.V = self.NP.Vreset
                        self.NV.DVDt = 0
                        self.NV.tLastSpike = t
                    else:  # if I_disease is turned on
                        self.NV.V = self.NP.VT                       # 27.03.2024
                        self.NV.tLastSpike = t
    # ************************************************************************************************

# if __name__ == '__main__':
#     #from Corona_main import TParameters, TStim, DefaultParameters
#
#     #Pars, Stim, NP0, CoronaPars = DefaultParameters()
#     # Neuron parameters
#     NP0 = NeuronProperties()
#     NP0.If_CBRD = 0
#     NP0.tau = 30
#     NP0.VT = 10 # mV
#     NP0.tau_d = 20
#     NP0.Vreset = 0
#     #
#     dt = 0.15 # ms
#     Stim_I = 0.300  # pA
#     NP0.I_disease_ratio = 4
#     NP0.I_disease_ampl = 50
#     NP0.tau_r = 1  # ms
#     NP0.If_escape = 0
#
#     n=TNrn(NP0)
#     n.InitialConditions()
#     n.MembranePotential(Stim_I,0,dt)
#     print(dt)
