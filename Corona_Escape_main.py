
from NeuronPatientEq_Escape import TNrn, NeuronProperties
from PopulationEq_Escape import CreatePopulationBy_NP_O
import random
import numpy as np
import matplotlib.pyplot as plt


class TParameters:
# Defines class of properties
    # Time
    t_end: float
    dt: float
    nt_end: int
    # Phase space and intrinsic noise
    Nts: int
    ts_end: float
    # Other
    n_show: int
    n_DrawPhase: int
    n_write: int
    N_MC: int

class TStim:
# Defines class of stimulation properties
    kI0: float
    Duration: float
    t_start: float
    kI0_0: float
    If_JustCurrent_Or_ResetFraction :int

class TCoronavirus:
# Defines class of coronavirus recurrent term properties
    k: float
    k0: float
    Summer_k: float
    Summer_T: float
    sgm_V0: float
    sgm_V: float
    t_start: float
    t_end: float
    tau_I: float
    NoisyI: float

def Current_kI0(t, Stim):
# Stimulation
    if t > Stim.t_start and t <= Stim.t_start + Stim.Duration:
        MeanCurr = Stim.kI0_0 + Stim.kI0
    else:
        MeanCurr = Stim.kI0_0
    return MeanCurr

def Contamination_k(t, Corona):
# Contamination
    if (t > Corona.t_start) and (t < Corona.t_end):
        Contamination = Corona.k
        if int((t-Corona.t_start)/Corona.Summer_T) % 2 != 0:
            Contamination = Corona.Summer_k
    else:
        Contamination = Corona.k0
    return Contamination

def Contamination_sgmV(t, Corona):
# Contamination
    if (t > Corona.t_start):# and (t < Corona.t_end):
        Contamination = Corona.sgm_V
    else:
        Contamination = Corona.sgm_V0
    return Contamination

# ***********************************************************************************************
def SingleNeuron(NP0, Stim, Pars, Repr_sgm_V_):
# Simulates a single neuron-patient *************************************************************
    # Create representative neuron
    ANrn = TNrn(NP0)
    ANrn.NP = NP0
    ANrn.InitialConditions()
    t_arr = np.linspace(0, 1, Pars.nt_end+1)
    V_arr = np.linspace(ANrn.NV.V, ANrn.NV.V, Pars.nt_end+1)
    t = 0
    nt = 0
    while nt < Pars.nt_end:
        # ----- Step -----
        t = t + Pars.dt
        nt = nt + 1
        # Representative neuron
        if ANrn.NP.If_escape == 0:
            Noise_ = Repr_sgm_V_ * np.sqrt(2 / Pars.dt / NP0.tau) * random.gauss(0, 1)  # muA/cm^2
        else: # ESCAPE-noise
            Noise_ = 0
            ANrn.NP.sgmV_escape = Repr_sgm_V_
        uu_ = Current_kI0(t, Stim) + Noise_
        #**************************************
        ANrn.MembranePotential(uu_, t, Pars.dt)
        #**************************************
        V_arr[nt] = ANrn.NV.V
        t_arr[nt] = t

    return t_arr, V_arr

# ***********************************************************************************************
def MonteCarlo(NP0, Stim, Pars, Corona):
# Simulates population of coupled neurons-patients **********************************************
    dt_bin = Pars.dt #ms  time bin
    if NP0.If_escape == 0:
        dt_bin = 1# Pars.dt #0.1#1
    dnt_I = int((Corona.tau_I+dt_bin)/Pars.dt)
    maxNSpikes = int(Pars.t_end/dt_bin)                  # number of bins
    Spikes   = np.linspace(0, 0,     maxNSpikes+1)  # number of spikes in bins
    tMC_arr  = np.linspace(0, Pars.t_end, maxNSpikes+1)  # time space discretized for spike counting
    nu_arr   = np.linspace(0, 0, Pars.nt_end+1)
    kI_arr = np.linspace(0, 0, Pars.nt_end+1)
    VlastNrn_arr = np.linspace(0, 0, Pars.nt_end+1)
    ANrn = [TNrn] * (Pars.N_MC+1)
    # Introduce neurons
    for i in range(1,Pars.N_MC+1):
        ANrn[i] = TNrn(NP0)
        ANrn[i].NP = NP0
        ANrn[i].InitialConditions()
    t = 0
    nt = 0
    nu_ = 0
    ### Loop for time steps
    while True:
        # ----- Step -----
        t = t + Pars.dt
        nt = nt + 1
        ### Input
        if Stim.If_JustCurrent_Or_ResetFraction == 1:
            Current_kI0_ = Current_kI0(t, Stim)
        else:     ### Set Fraction Of Population To Be Ill
            Current_kI0_ = Stim.kI0_0
            if abs(t-Stim.t_start)<Pars.dt:
                N_fraction = int(Pars.N_MC * Stim.kI0/Corona.k)
                for i in range(1, N_fraction + 1):
                    ANrn[i].ResetToThreshold(t)
        k_nu2_ = Contamination_k(t,              Corona) * nu_
        k_nu1_ = Contamination_k(t-Corona.tau_I, Corona) * nu_arr[nt - dnt_I]
        kI_arr[nt] = kI_arr[nt-1] + Pars.dt/1000 * (k_nu2_ - k_nu1_)
        Common_Noise_ = Corona.NoisyI * kI_arr[nt] * random.gauss(0, 1)
        ### Loop for neurons
        for i in range(1,Pars.N_MC+1):
            ### Input
            sgm_V_ = Contamination_sgmV(t, Corona)
            ANrn[i].NP.sgmV_escape = sgm_V_
            if ANrn[i].NP.If_escape == 0:   # WHITE noise
                Noise_ = sgm_V_ * np.sqrt(2 / Pars.dt / NP0.tau) * random.gauss(0, 1)
            else:                           # ESCAPE noise
                Noise_ = 0
            uu_ = Current_kI0_ + kI_arr[nt] + Common_Noise_ + Noise_
            # ****************************************
            ANrn[i].MembranePotential(uu_, t, Pars.dt)
            # ****************************************

            ### Count spikes
            bin = int(t/dt_bin)  # each ms
            if abs(ANrn[i].NV.tLastSpike - t) < Pars.dt / 2:
                Spikes[bin] += 1  # spikes in a bin

        nu_= Spikes[max(bin-1,0)]/Pars.N_MC/dt_bin*1000 # measured with a delay less than 1ms
        nu_arr[nt] = nu_
        VlastNrn_arr[nt] = ANrn[Pars.N_MC].NV.V
        if nt % 100 == 0 :
            print('MC: t=',t)
        if nt >= Pars.nt_end or bin >= maxNSpikes:
            break # end of time loop

    return tMC_arr, Spikes/Pars.N_MC/dt_bin*1000, Pars.N_MC, kI_arr, VlastNrn_arr


# ***********************************************************************************************
def CBRD(NP0, Stim, Pars, Corona, t_plot):
# Simulates CBRD model for LIF and conductance-based neurons ************************************
    # Create representative neuron
    ANrn = TNrn(NP0)
    ANrn.NP = NP0
    ANrn.InitialConditions()
    # Create and initiate Population
    Pop = CreatePopulationBy_NP_O('P', Pars.Nts, Pars.ts_end, Corona.sgm_V0, NP0, ANrn)
    Pop.InitialConditions(Pars.dt)
    # Arrays to plot
    t_arr  = np.linspace(0, 1, Pars.nt_end+1)
    nu_arr = np.linspace(0, 0, Pars.nt_end+1)
    V_arr = np.linspace(ANrn.NV.V, ANrn.NV.V, Pars.nt_end+1)
    kI_arr = np.linspace(0,0, Pars.nt_end+1)
    kI0_arr = np.linspace(0,0, Pars.nt_end+1)
    U_arr = np.linspace(0,0, Pars.nt_end+1)
    kItot_arr = np.linspace(0,0, Pars.nt_end+1)
    ro_ts_arr = np.zeros((len(t_plot),Pars.Nts+1))
    U_ts_arr = np.zeros((len(t_plot),Pars.Nts+1))
    H_ts_arr = np.zeros((len(t_plot),Pars.Nts+1))
    ts_arr = np.zeros((len(t_plot),Pars.Nts+1))

    t = 0
    nt = 0
    while nt < Pars.nt_end:
        # ----- Step -----
        t += Pars.dt
        nt += 1

        Pop.PP.sgm_V = Contamination_sgmV(t,Corona)
        ANrn.NP.sgmV_escape = Pop.PP.sgm_V

        # ******************
        Pop.Density(Pars.dt)
        # ******************

        # Input
        if Stim.If_JustCurrent_Or_ResetFraction == 1:
            kI0_arr[nt] =  Current_kI0(t, Stim)
        else:
            kI0_arr[nt] =  Stim.kI0_0
            if abs(t-Stim.t_start)<Pars.dt:
                Pop.ResetFractionToThreshold(Stim.kI0/Corona.k, Pars.dt)
        k_nu2_ = Contamination_k(t,              Corona) * Pop.nu
        k_nu1_ = Contamination_k(t-Corona.tau_I, Corona) * nu_arr[nt-int(Corona.tau_I/Pars.dt)]
        kI_arr[nt] = kI_arr[nt-1] + Pars.dt/1000 * (k_nu2_ - k_nu1_)
        Common_Noise_ = Corona.NoisyI * kI_arr[nt] * random.gauss(0, 1)
        Pop.Isyn = kI0_arr[nt] + kI_arr[nt] + Common_Noise_

        # *******************************
        Pop.MembranePotential(t, Pars.dt)
        # *******************************

        # Representative neuron
        if ANrn.NP.If_escape == 0:
            Noise_ = Pop.PP.sgm_V * np.sqrt(2 / Pars.dt / NP0.tau) * random.gauss(0, 1)  # muA/cm^2
        else: # ESCAPE-noise
            Noise_ = 0
        ANrn.MembranePotential(Pop.Isyn + Noise_, t, Pars.dt)
        # *************
        t_arr[nt] = t
        V_arr[nt] = ANrn.NV.V
        nu_arr[nt] = Pop.nu
        U_arr[nt] = Pop.U
        kItot_arr[nt] = Pop.Isyn
        for it in range(len(t_plot)):
            if np.abs(t-t_plot[it])<Pars.dt/2:
                for i in range(Pop.PP.Nts + 1):
                    U_ts_arr[it,i]  = Pop.Nrn[i].NV.V
                ro_ts_arr[it,:] = Pop.ro[:]
                ts_arr[it, :] = Pop.ts[:]
                H_ts_arr[it, :] = Pop.AB[:] #A  .ro_Hzrd[:]

        # Plot the updated solution at each time step
        if nt % 200 == 0 :
            plt.clf()  # Clear the plot for the next time step
            plt.scatter(Pop.ts, Pop.V, label=f'V')
            plt.scatter(Pop.ts[:Pop.PP.Nts], Pop.ro[:Pop.PP.Nts], label=f'ro at t={t:.0f}')
            plt.scatter(Pop.ts, Pop.AB, label='H')
            plt.xlabel('t*')
            plt.ylabel('ro')
            plt.legend()
            plt.pause(0.001)  # Adjust the pause duration as needed

    return t_arr, nu_arr, V_arr, kI_arr, U_arr, kItot_arr, ro_ts_arr, ts_arr, U_ts_arr, H_ts_arr, kI0_arr
# *******************************************************************************************************

def DefaultParameters():
# Sets all properties
    Pars = TParameters()
    # Phase space, noise and time
    Pars.Nts = 300              # numerical parameter of t*-space discretization
    Pars.ts_end = 150 # ms      # numerical parameter of t*-space limit
    Pars.t_end = 700  # ms      # numerical: time of integration
    Pars.dt = 0.1 #A 0.15  # ms        # numerical: time step
    Pars.nt_end = int(Pars.t_end / Pars.dt)  # numerical: number of time steps
    # Neuron parameters
    NP0 = NeuronProperties()
    NP0.If_CBRD = 0             # 0 for neuron with reset; 1 without
    NP0.tau = 30 # ms           # relaxation time scale
    NP0.VT = 4 # mV             # threshold
    NP0.I_disease_ampl = 50     # intensity of disease - amplitude of current describing the disease explicitly ("a" in the text)
    NP0.I_disease_ratio = 4
    NP0.tau_r = 1 #ms           # duration of disease (for double exponential term)
    NP0.tau_a = 10 #ms          # absolute refractoriness time period
    NP0.tau_d = 20 #ms          # duration of disease = no-spike interval (\tau_R in text)
    NP0.Vreset = 0              # reset level
    if NP0.I_disease_ampl != 0:
        NP0.tau_a = 0
    NP0.If_escape = 0           # escape (1) or white (0) noise
    NP0.If_A_AB_or_power_escape = 1     # A-function of self-similar solution for white noise and stationary input is used for escape-noise
    NP0.c_escape = 100          # c=10 and n=2 approximate typeA-function
    NP0.m_escape = 2            # c=10 and n=2 approximate typeA-function
    # Coronavirus
    Corona = TCoronavirus()
    Corona.k = 1.2 #0.011 #0.02 #2*0.01#7/1000      # contamination from population; it is k*R_0(t) in the text
    Corona.k0 = 0*Corona.k #0.005           # contamination before epidemy
    Corona.sgm_V = 2.0              # noise amplitude
    Corona.sgm_V0 = 0.02            # noise amplitude before epidemy
    Corona.t_start = 100 # ms       # onset for Corona.k
    Corona.t_end = 1000 # ms        # return from Corona.k to Corona.k0
    Corona.tau_I = 2*5 #ms          # delay of contamination or duration of Infected period
    Corona.NoisyI = 0*0.5           # noisy part of recurrent current
    # Stimul
    Stim = TStim()
    Stim.kI0 = 0  #0.4 #0.1                # external source (in addition to kI0_0)
    Stim.Duration = Corona.tau_I #20 # ms     # external source duration
    Stim.t_start = 200 # ms     # external source onset
    Stim.kI0_0 = 0              # external source background
    Stim.If_JustCurrent_Or_ResetFraction = 2    # if needed, reset fraction of particles to threshold state
    # ***************
    Pars.N_MC = 100  # Number of particles in Monte-Carlo simulations
    # ***************

    ### Case with sudden appearance of ill people                                                   # case 4
    NP0.If_escape = 0
    NP0.If_A_AB_or_power_escape = 3
    NP0.tau = 50  # ms           # relaxation time scale
    NP0.c_escape = 10           # c=10 and n=2 approximate typeA-function
    NP0.m_escape = 2            # variant 5,6,7 correspond to m=1, 2 and 0.5, respectfully
    Corona.k0 = 0.3 #1.2     # contamination from population; it is k*R_0(t) in the text
    Corona.Summer_k = Corona.k/2 # contamination during summer
    Corona.Summer_T = 2000 #200            # summer-winter duration             # Summer-Winter
    Corona.sgm_V0 = 0.5          # noise amplitude before epidemy
    Corona.t_start = 100
    Corona.t_end = 1000 #*10                                                    # Summer-Winter
    Pars.Nts = 400  # numerical parameter of t*-space discretization
    Pars.ts_end = 200  # ms      # numerical parameter of t*-space limit
    Pars.dt = 0.1
    Pars.t_end = 1500 #*2  # ms      # numerical: time of integration           # Summer-Winter
    Pars.nt_end = int(Pars.t_end / Pars.dt)  # numerical: number of time steps
    Stim.t_start = 100 # ms     # external source onset
    Stim.kI0 = 0.01*Corona.Summer_k

    # ### Case with ESCAPE noise and sudden appearance of ill people                              # cases 5,6,7
    # NP0.If_escape = 1
    # NP0.If_A_AB_or_power_escape = 3
    # NP0.tau = 50  # ms           # relaxation time scale
    # NP0.VT = 2 # mV             # threshold
    # NP0.c_escape = 15  # c=10 and n=2 approximate typeA-function
    # NP0.m_escape = 1  # variant 5,6,7 correspond to m=1, 2 and 0.5, respectfully
    # NP0.I_disease_ampl = 100     # intensity of disease - amplitude of current describing the disease explicitly ("a" in the text)
    # NP0.I_disease_ratio = 4
    # Corona.k = 2*1.2
    # Corona.k0 = 2*0.3  # 1.2     # contamination from population; it is k*R_0(t) in the text
    # Corona.Summer_k = Corona.k/2 # contamination during summer
    # Corona.Summer_T = 2000#200            # summer-winter duration
    # Corona.sgm_V = 2#1.2                # noise amplitude
    # Corona.sgm_V0 = 0.25             # noise amplitude before epidemy
    # Corona.t_start = 100
    # Corona.t_end = 800#*10
    # Corona.tau_I = 10        # delay of contamination or duration of Infected period
    # Pars.Nts = 300  # numerical parameter of t*-space discretization
    # Pars.ts_end = 300  # ms      # numerical parameter of t*-space limit
    # Pars.dt = 0.1
    # Pars.t_end = 1500#*2  # ms      # numerical: time of integration
    # Pars.nt_end = int(Pars.t_end / Pars.dt)  # numerical: number of time steps
    # Stim.t_start = 100  # ms     # external source onset
    # Stim.kI0 = 0.01*Corona.k  # 0.1
    # Stim.kI0_0 = 0.0

    # ### Test with ESCAPE or WHITE noise and step without a recurrent source
    # Stim.If_JustCurrent_Or_ResetFraction = 1
    # NP0.If_escape = 0
    # NP0.If_A_AB_or_power_escape = 2
    # NP0.c_escape = 10
    # NP0.m_escape = 2  # 0.25
    # NP0.tau = 100  # ms           # relaxation time scale
    # Corona.k0 = 0
    # Corona.sgm_V0 = 2          # noise amplitude before epidemy
    # Corona.t_start = 1000
    # Pars.Nts = 300  # numerical parameter of t*-space discretization
    # Pars.ts_end = 300  # ms      # numerical parameter of t*-space limit
    # Pars.dt = 0.15
    # Pars.nt_end = int(Pars.t_end / Pars.dt)  # numerical: number of time steps
    # Stim.t_start = 100  # ms     # external source onset
    # Stim.Duration = 400
    # Stim.kI0 = 0.4
    # Stim.kI0_0 = 0.0

    return Pars, Stim, NP0, Corona

# ***********************************************************************************************
if __name__ == '__main__':

    Pars, Stim, NP0, Corona = DefaultParameters()
    t_plot = [Stim.t_start + x for x in [120,140,160,180,200,220]]   # time moments to plot distributions in t*-space
    #                                                 ****************************************************
    t_arr,         VnoNoise_arr                     = SingleNeuron(NP0, Stim, Pars, Corona.sgm_V/1000)
    tMC_arr, nuMC_arr, iMC, kIrecMC_arr, V_MC_arr   = MonteCarlo(NP0, Stim, Pars, Corona)
    t_arr, nu_arr, V_arr, kI_arr, U_arr, kItot_arr, ro_ts_arr, ts_arr, U_ts_arr, H_ts_arr, kI0_arr\
                                                    = CBRD(NP0, Stim, Pars, Corona, t_plot)
    #                                                 ****************************************************

    # Writing **********************************************************************************************************
    yyy = open('t_I_G_U_nu.dat', 'w')
    rrr = open('Repr_t_V_a.dat', 'w')
    for i in range(len(t_arr)):
        yyy.write('{:8.1f} {:8.5f} {:8.1f} {:8.1f} {:8.1f} {:8.1f} {:8.5f} {:8.5f} {:8.3f} {:8.3f}\n'.
                    #    V1        V2                  V3        V4
                  format(t_arr[i], kI_arr[i]/Corona.k, U_arr[i], nu_arr[i],
                    #    V5      V6                                         V7                       V8          V9                                   V10
                         NP0.VT, Contamination_k(t_arr[i],Corona)/Corona.k, kIrecMC_arr[i]/Corona.k, kI0_arr[i], Contamination_sgmV(t_arr[i],Corona), t_arr[i]-Stim.t_start))
                    #                                                        V1        V2               V3                     V4        V5           V6
        rrr.write('{:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f}\n'.format(t_arr[i], VnoNoise_arr[i], kItot_arr[i]/Corona.k, V_arr[i], V_MC_arr[i], t_arr[i]-Stim.t_start))
    yyy.close()
    rrr.close()

    # Writing from Monte-Carlo
    qqq = open('MC.dat', 'w')
    for i in range(len(tMC_arr)):
        #                                             V1          V2           V3
        qqq.write('{:8.3f} {:8.3f} {:8.3f} \n'.format(tMC_arr[i], nuMC_arr[i], tMC_arr[i]-Stim.t_start))
    qqq.close()

    # Writing ts-space
    for i in range(len(t_plot)):
        ttt = open(f'ts-space{t_plot[i]:.0f}.dat', 'w')
        for nts in range(Pars.Nts+1):
            #                                                                    V1             V2               V3               V4                V5      V6
            ttt.write('{:8.2f} {:8.1f} {:8.1f} {:8.1f} {:8.1f} {:8.1f}\n'.format(ts_arr[i,nts], U_ts_arr[i,nts], H_ts_arr[i,nts], ro_ts_arr[i,nts], NP0.VT, H_ts_arr[i,nts]*ro_ts_arr[i,nts]))
        ttt.close()

    # Write settings to a txt file
    def write_objects_to_file(filename, *objects):
        with open(filename, 'w') as file:
            for obj in objects:
                file.write(f"{type(obj).__name__}:\n")
                for key, value in obj.__dict__.items():
                    file.write(f"- {key}: {value}\n")
                file.write("\n")

    write_objects_to_file('Settings.txt', Pars,NP0,Stim,Corona)
    print("Objects have been written")

    # Plotting *********************************************************************************************************
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 12))
    axes[0,0].plot(t_arr, kI0_arr, "-b", label="u(t)", linewidth=2)
    axes[0,0].plot(t_arr, kI_arr, "-r", label="kIrec(t)", linewidth=1)
    axes[0,0].plot(t_arr, kIrecMC_arr, "-y", label="kIrecMC(t)", linewidth=1)
    axes[0,0].plot(t_arr, kItot_arr, "-g", label="u(t)+k*I(t)", linewidth=1)
    axes[1,0].plot(t_arr, V_arr, "-b", label="V(t)", linewidth=1)
    axes[1,0].plot(t_arr, V_MC_arr, "-y", label="V_MC(t)", linewidth=1)
    axes[1,0].plot(t_arr, VnoNoise_arr, "-r", label="V(t), noNoise, no rec.", linewidth=2)
    axes[1,0].plot(t_arr, U_arr, "-g", label="U(t)", linewidth=2)
    axes[1,0].plot(t_arr, t_arr*0+NP0.VT, "black", label="VT", linewidth=2)
    axes[2,0].plot(tMC_arr, nuMC_arr, "-r", label="nu(t) from MC, w/o recurent term", linewidth=1)
    axes[2,0].plot(t_arr, nu_arr, "-b", label="nu(t)", linewidth=2)
    for i in range(len(t_plot)):
        axes[0,1].plot(ts_arr[i,:Pars.Nts+1],  U_ts_arr[i,:Pars.Nts+1], "*", label=" U("+str(t_plot[i])+",t*)", linewidth=1)
        axes[1,1].plot(ts_arr[i,:Pars.Nts+1],  H_ts_arr[i,:Pars.Nts+1], "*", label=" H("+str(t_plot[i])+",t*)", linewidth=1)
        axes[2,1].plot(ts_arr[i,:Pars.Nts],    ro_ts_arr[i,:Pars.Nts],  "*", label="ro("+str(t_plot[i])+",t*)", linewidth=1)

    axes[0,0].legend()
    axes[0,1].legend()
    axes[1,0].legend()
    axes[1,1].legend()
    axes[2,0].legend()
    axes[2,1].legend()
    axes[0,0].set_ylabel('Current')
    axes[1,0].set_ylabel('Potential')
    axes[2,0].set_ylabel('Rate')
    axes[0,1].set_ylabel('Potential')
    axes[1,1].set_ylabel('Hazard')
    axes[2,1].set_ylabel('Density')
    axes[2,1].set_xlabel("Phase variable t*")
    axes[2,0].set_xlabel("Time t")
    fig.savefig('Coronavirus.png', dpi=300)
    plt.show()
