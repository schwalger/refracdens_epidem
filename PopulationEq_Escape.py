import copy

from NeuronPatientEq import TNrn, NeuronProperties
import math
import numpy as np


def Hazard(sgmV_,dVdt_,taum_,U_,VT_,ts_,tau_d_):

    def F_tilde(x):
        # A component of the hazard-function
        if x > 4:
            return x * math.sqrt(2)
        elif x < -5:
            return 0
        else:
            return math.sqrt(2 / math.pi) * math.exp(-x * x) / (1 - math.erf(x))

    def fA_T(T):
        # A component of the hazard-function
        if abs(T) <= 2.5:
            T3 = T * T * T
            A0 = math.exp(-2.478E-3 - 1.123 * T - 0.2375 * T * T - 0.06567 * T3
                          - 0.01813 * T3 * T - 1.713E-003 * T3 * T * T + 5.484E-004 * T3 * T3
                          + 1.034E-004 * T3 * T3 * T)
        elif T > 2.5:
            A0 = 0
        else:
            A0 = 7
        return max(A0, 0)

    V_ = (U_ - VT_) / (sgmV_ * np.sqrt(2))
    # Hazard-function component that is derived from "frozen" Gauss
    if dVdt_ <= 0:
        B = 0
    else:
        B = F_tilde(V_) / sgmV_ * dVdt_ * 1e3 \
            # Hazard-function component that is derived from self-similar Fokker-Planck
    A = fA_T(-V_) / taum_ * 1e3  # Hz
    H_ = A + B
    # Absolute refractory period
    if (ts_ <= tau_d_):
        H_ = 0
    return H_


class PopulationProperties:
# Defines properties of a population
    NP: NeuronProperties
    dts: float
    ts_end: float
    sgm_V: float
    Nts: int

class TPopulation:
# Defines class of a population, including methods to solve equation for density "ro" and potential "U"
    def __init__(self, PP_,ANrn):
        MaxPh = PP_.Nts+1
        # Control variables
        self.Isyn = 0.0
        # Variables
        self.nu = 0.0
        self.U = 0.0
        self.dUdt = 0.0
        self.Bum = 0.0
        self.tBum = 0.0
        self.ro = [0.0] * MaxPh
        self.ts = [0.0] * MaxPh
        self.ro_Hzrd = [0.0] * MaxPh
        self.AB = [0.0] * MaxPh
        self.V = [0.0] * MaxPh
        # Fulfil parameters
        self.PP = copy.deepcopy(PP_)
        # Create set of neurons
        self.Nrn = [TNrn] * MaxPh
        for i in range(PP_.Nts + 1):
            self.Nrn[i] = TNrn(PP_.NP)
            self.Nrn[i].NP.If_CBRD = 1

    def nts_by_ts(self, ts):
    # Finds the number of a particle just after ts or precisely at ts
        if self.PP.Nts == 0:
            return 0
        t2 = 1e8
        i2 = -1
        for i in range(self.PP.Nts + 1):
            if self.ts[i] >= ts and self.ts[i] < t2:
                i2 = i
                t2 = self.ts[i]
        # Exception
        if i2 == -1:
            i2 = self.PP.Nts
        return i2

    def InitialConditions(self, dt):
    # Sets initial conditions for "ro" and "U"
        # Set conditions at rest
        for i in range(self.PP.Nts + 1):
            self.Nrn[i].InitialConditions()

        self.PP.NP = self.Nrn[self.PP.Nts].NP

        # Set conditions at spike
        for i in range(self.PP.Nts):   # except the last one
            self.ConditionsAtSpike(self.Nrn[i], 0)

        # Integrate till "ts"
        self.ts[0] = 0
        self.ts[self.PP.Nts] = self.PP.Nts * self.PP.dts
        for i in range(1, self.PP.Nts):    # except 0 and the last ones
            self.ts[i] = i * self.PP.dts
            nt_ = 0
            while nt_ < self.ts[i] / dt:
                nt_ += 1
                # ******* One step of integration **
                if nt_ * dt > self.PP.NP.tau_a:
                    self.Nrn[i].MembranePotential(0, nt_ * dt, dt)
                # **********************************

        # Time of last spike is negative
        for i in range(0, self.PP.Nts):  # except the last ones
            self.Nrn[i].NV.tLastSpike = -i * self.PP.dts  # time of last spike
        self.Nrn[self.PP.Nts].NV.tLastSpike=-888

        # Fulfil density
        for i in range(self.PP.Nts + 1):
            self.ro[i] = 0
        self.ro[self.PP.Nts] = 1 / self.PP.dts * 1e3  # dts is in ms, ro is in Hz

        # Output variables
        self.nu = 0
        self.U = 0
        self.dUdt = 0

    def ResetFractionToThreshold(self, fraction, dt):
        dro = min(self.ro[self.PP.Nts], fraction/(self.PP.dts/1000))
        self.ro[self.PP.Nts] = self.ro[self.PP.Nts] - dro
        self.nu              = self.nu              + dro * self.PP.dts / dt
        self.Bum             = self.Bum             + dro * self.PP.dts / 1e3

    def ConditionsAtSpike(self, ANrn_, t):
    # Resets
        if (ANrn_.NP.I_disease_ampl == 0):  # if I_disease is turned off
            ANrn_.NV.V = ANrn_.NP.Vreset
            ANrn_.NV.DVDt = 0
        else:                               # if I_disease is turned on
            ANrn_.NV.V = self.PP.NP.VT
            ANrn_.NV.tLastSpike = t

    # *************************************************************************************************
    def MembranePotential(self, t, dt):
    # Solves transport eq. for the potential "U" ******************************************************
        for i in range(self.PP.Nts + 1):
            if self.ts[i] >= self.PP.NP.tau_a:
                self.Nrn[i].MembranePotential(self.Isyn, t, dt)
            if i < self.PP.Nts:
                self.ts[i] += dt

        # Return those who cross ts_end
        for i in range(self.PP.Nts):      # except the last one
            if self.ts[i] >= self.PP.ts_end:
                self.ts[i] -= self.PP.ts_end
                self.ConditionsAtSpike(self.Nrn[i], t)
                self.ro[self.PP.Nts] += self.ro[i]
                if t == self.tBum:
                    self.tBum = -self.tBum
                self.ro[i] = self.Bum / (t - self.tBum) * 1e3
                self.tBum = t
                self.Bum = 0

        # Rename
        for i in range(self.PP.Nts+1):
            self.V[i] = self.Nrn[i].NV.V
        self.U = self.Nrn[self.PP.Nts].NV.V
        self.dUdt = self.Nrn[self.PP.Nts].NV.DVDt


    # { Comments to NUMERICAL METHOD.
    #   Left cell at ts=0:       ro=nu,      U=Vreset, cell size - (t-tBum).
    #   Right cell at ts=ts_end: ro=ro[Nts], U=U[Nts], cell size - dts.
    #   Next to right cell i: ro=ro[i], U=U[i], cell size - ts_end-ts[i]=dts+(t-tBum)
    # }

    # ********************************************************************************************
    def Density(self, dt):
    # Solves transport eq. for density "ro" ******************************************************
        S = 0
        for i in range(self.PP.Nts, -1, -1):  # loop for phase

            # *** Hazard ***
            if (self.PP.NP.If_escape == 0) or (self.PP.NP.If_A_AB_or_power_escape == 2):
                #*******************************************************************************************************************************
                H_ = Hazard(self.PP.sgm_V, self.Nrn[i].NV.DVDt, self.PP.NP.tau, self.Nrn[i].NV.V, self.PP.NP.VT, self.ts[i], self.PP.NP.tau_d)
                #*******************************************************************************************************************************
            elif self.PP.NP.If_A_AB_or_power_escape == 1:  # ESCAPE-noise with only A
                H_ = Hazard(self.PP.sgm_V,                -1.0, self.PP.NP.tau, self.Nrn[i].NV.V, self.PP.NP.VT, self.ts[i], self.PP.NP.tau_d)
            elif self.PP.NP.If_A_AB_or_power_escape == 3:  # ESCAPE-noise of power type
                H_ = self.PP.NP.c_escape * max(0, self.Nrn[i].NV.V / self.PP.NP.VT) ** self.PP.NP.m_escape
                if (self.ts[i] <= self.PP.NP.tau_d):
                    H_ = 0

            # *** Source ***
            dro = min(self.ro[i], self.ro[i] * (1 - np.exp(- H_ * dt / 1e3)))
            # *** Density eq. ***********
            self.ro[i] = self.ro[i] - dro
            # ***************************
            self.ro_Hzrd[i] = dro / dt * 1e3            # remember for plotting
            self.AB[i] = H_                             # remember for plotting
            S = S + dro

        # Firing rate:
        self.nu = S * self.PP.dts / dt
        self.Bum = self.Bum + S * self.PP.dts / 1e3
        return self.nu
#****************************************************************************************************

def CreatePopulationBy_NP_O(Name, Nts, ts_end, sgm_V, NP,ANrn):
# Creates Object-Population
    PP_ = PopulationProperties()
    PP_.NP = NP
    PP_.Nts = Nts
    PP_.ts_end = ts_end
    PP_.dts = ts_end / Nts
    PP_.sgm_V = sgm_V
    APop = TPopulation(PP_,ANrn)
    return APop

