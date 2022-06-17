#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 16:07:40 2022

@author: ana_nunez
"""

'''
This file is to create 2 different population groups and 4 different couplings 
to see if that works instead of one population

'''

# --------------------------------Imports------------------------------
import os
import numpy as np
from random import choices
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
from brian2 import ms, mV

def Network_model_2(sim_dur=6000*ms, pre_run_dur=100*ms, total_neurons=1000, scale_factor=1):
    '''
    Function to implement a network with two populations excitatory and inhibitory. 
    Recurrent and inh-exc sznapses on top of external Poissonian spike trains.
    
    sim_dur: Time duration of the simulation
    pre_run_dur: Time tu run the network before actuallz stating to record from it.
    total_neurons: Total number of neurons that would be then splited into two populations
    scale_factor: In case we want to reduce the network siye we can use the scale factor
        But it is actually not recommended, since the random network need the 1000 neurons to be involved. In some cases more.    
    Network type can be: None, subpopulation or feedforward
    
    Returns network and monitors
    '''
# ----------------------------------Initialization---------------------------
    start_scope()
    defaultclock.dt = 0.001*ms

    seed(333)
# ------------------------------Parameters for populations-------------------------

    # Constant variables for populations
    Vr = -65 *mV
    Eex = 0 *mV
    Ein = -75 *mV
    tau_ax = 0.6 *ms # Value should be from 0.3 to 1.3 How should we select it? Here is constant
    
    # Constant variables for excitatory population
    Cm_ex = 400 *pF
    gL_ex = 25 *nS
    Vt_ex= -45 *mV
    tref_ex = 3 *ms
    
    # Constant variables for inhibitory population
    Cm_in = 200 *pF    # From 200 to 100 pF in figure S5 from Memesheimmer 2010
    gL_in = 25 *nS     # From 25 to 33 nS in figure S5 of Memmesheimer 2010
    Vt_in = -55 *mV
    tref_in = 2 *ms
    
    # Constant variables for dendritic spike
    A = 55 *nA
    B = 64 *nA
    C = 9 *nA
    tauDS1 = 0.2 *ms
    tauDS2 = 0.3 *ms
    tauDS3 = 0.7 *ms
    tauDS = 2.7 * ms
#    g_theta = 8.65 * nS
    tref_D = tauDS + 2.5 * ms

# ------------------------------Parameters for couplings-------------------------

    # Parameters for exc-exc (from excitatory to excitatory)
    p_exex = 0.08
    tau_exex = 1 *ms
    gmax_exex = 2.3 *nS
    tauA1_ee = 2.5 *ms
    tauA2_ee = 0.5 *ms
    delta_t = 2 * ms
    n_th = 3
    
    # Parameters for exc-inh (from excitatory to inhibitory)
    p_inex = 0.1
    tau_inex = 0.5 * ms
    gmax_inex = 3.2 * nS
    tauA1_ie = 2 * ms
    tauA2_ie = 0.35 * ms
    
    # Parameters for exc-inh (from inhibitory to excitatory)
    p_exin = 0.1
    tau_exin = 1 * ms
    gmax_exin = 5 * nS
    tauG1_ei = 4 * ms
    tauG2_ei = 0.3 * ms
    
    # Parameters for exc-inh (from inhibitory to inhibitory)
    p_inin = 0.02
    tau_inin = 0.5 * ms
    gmax_inin = 4 *nS
    tauG1_ii = 2.5 * ms
    tauG2_ii = 0.4 * ms

# ------------------------------Parameter for NETWORKS-------------------------
    N = int(total_neurons*scale_factor)    # Total number of neurons
    N_ex = int(0.9 * N)
    N_in = N - N_ex

# ----------------------------Parameters for inputs-----------------------------
    rate_to_exc = 2.3 * kHz
    rate_to_inh = 0.5 * kHz
    exc_input_ratio = 0.75
    inh_input_ratio = 1 - exc_input_ratio

# ----------------------------Initial STIMULATION-----------------------------
    # seed at the beginning
    exc_input_to_exc = PoissonGroup(N_ex, rates=rate_to_exc * exc_input_ratio, name='Poissongroup_exex')
    inh_input_to_exc = PoissonGroup(N_ex, rates=rate_to_exc * inh_input_ratio, name='Poissongroup_inex')
    exc_input_to_inh = PoissonGroup(N_in, rates=rate_to_inh * exc_input_ratio, name='Poissongroup_exin')
    inh_input_to_inh = PoissonGroup(N_in, rates=rate_to_inh * inh_input_ratio, name='Poissongroup_inin')

# ---------------------------Populations --------------------------------
    # the variable r is a parameter of linear decay used to set the refractory period of the dendritic spike.
    eqs_neuron_exc = '''
    dv/dt = (1/Cm) * (gL * (Vr - v) + I_A + I_G + I_DS) : volt (unless refractory)
    
    
    I_A = (g_A_ext + g_A_rec) * (Eex - v) : amp
    
    dg_A_ext/dt = - g_A_ext / tauA1_ee + x1 : siemens
    dx1/dt = -x1/tauA2_ee : siemens/second
    dg_A_rec/dt = - g_A_rec / tauA1_ee + x2 : siemens
    dx2/dt = -x2/tauA2_ee : siemens/second
    
    tau_rise_A = tauA1_ee * tauA2_ee / (tauA1_ee - tauA2_ee) : second
    B_factor_A = 1 / ( (tauA2_ee/tauA1_ee)**(tau_rise_A/tauA1_ee) - (tauA2_ee/tauA1_ee)**(tau_rise_A/tauA2_ee) ) : 1
    
    
    I_G = (g_G_ext + g_G_rec) * (Ein - v) : amp
    
    dg_G_ext/dt = - g_G_ext / tauG1_ei + w1 : siemens
    dw1/dt = -w1/tauG2_ei : siemens/second
    dg_G_rec/dt = - g_G_rec / tauG1_ei + w2 : siemens
    dw2/dt = -w2/tauG2_ei : siemens/second
    
    tau_rise_G = tauG1_ei * tauG2_ei / (tauG1_ei - tauG2_ei) : second
    B_factor_G = 1 / ( (tauG2_ei/tauG1_ei)**(tau_rise_G/tauG1_ei) - (tauG2_ei/tauG1_ei)**(tau_rise_G/tauG2_ei) ) : 1
    
    
    dI_DS/dt = 1/tauDS1 * (-I_DS + y + z) : amp 
    dy/dt = -1/tauDS2 * y : amp 
    dz/dt = -1/tauDS3 * z : amp 
    dr/dt = -1/tref_D : 1
    
    
    n : 1
    Cm : farad
    gL : siemens
    
    refrac : second    
    v_threshold : volt
    '''
    
    eqs_neuron_inh = '''
    dv/dt = (1/Cm) * (gL * (Vr - v) + I_A + I_G) : volt (unless refractory)
    
    
    I_A = (g_A_ext + g_A_rec) * (Eex - v) : amp
    
    dg_A_ext/dt = - g_A_ext / tauA1_ie + x1 : siemens
    dx1/dt = -x1/tauA2_ie : siemens/second
    dg_A_rec/dt = - g_A_rec / tauA1_ie + x2 : siemens
    dx2/dt = -x2/tauA2_ie : siemens/second
    
    tau_rise_A = tauA1_ie * tauA2_ie / (tauA1_ie - tauA2_ie) : second
    B_factor_A = 1 / ( (tauA2_ie/tauA1_ie)**(tau_rise_A/tauA1_ie) - (tauA2_ie/tauA1_ie)**(tau_rise_A/tauA2_ie) ) : 1
    
    
    I_G = (g_G_ext + g_G_rec) * (Ein - v) : amp
    
    dg_G_ext/dt = - g_G_ext / tauG1_ii + w1 : siemens
    dw1/dt = -w1/tauG2_ii : siemens/second
    dg_G_rec/dt = - g_G_rec / tauG1_ii + w2 : siemens
    dw2/dt = -w2/tauG2_ii : siemens/second
    
    tau_rise_G = tauG1_ii * tauG2_ii / (tauG1_ii - tauG2_ii) : second
    B_factor_G = 1 / ( (tauG2_ii/tauG1_ii)**(tau_rise_G/tauG1_ii) - (tauG2_ii/tauG1_ii)**(tau_rise_G/tauG2_ii) ) : 1
    
    Cm : farad
    gL : siemens
    
    refrac : second    
    v_threshold : volt
    '''

    # Excitatory population
    G_ex = NeuronGroup(N_ex, eqs_neuron_exc, threshold = 'v>v_threshold', reset = 'v=Vr',  
                    refractory = 'refrac', method = 'euler', name='Neurongroup_ex',
                    events={'dendritic_event': 'n>n_th and r<=0'})
#    G_ex.run_on_event('dendritic_event', 'r=1; y += clip(1.46 - 0.053*(g_A_ext + g_A_rec)/nS, 0, Inf) * (tauDS2 - tauDS1)/tauDS2 * B; z += clip(1.46 - 0.053*(g_A_ext + g_A_rec)/nS, 0, Inf) * (tauDS1 - tauDS3)/tauDS3 * C' )
    G_ex.run_on_event('dendritic_event', 'r=1; y += clip(1.46 - 0.053*4*gmax_exex/nS, 0, Inf) * (tauDS2 - tauDS1)/tauDS2 * B; z += clip(1.46 - 0.053*4*gmax_exex/nS, 0, Inf) * (tauDS1 - tauDS3)/tauDS3 * C' )
    G_ex.Cm = Cm_ex
    G_ex.gL = gL_ex
    G_ex.refrac = tref_ex
    G_ex.v_threshold = Vt_ex
    G_ex.v = Vr
    #G_ex.v = 'rand() * 20 * mV + Vr'  # Initialize from Vrest to Vthreshold from -65 to -45
    G_ex.n = 0
    G_ex.I_DS = 0*nA
    G_ex.g_A_ext = 0*nS
    G_ex.g_A_rec = 0*nS
    G_ex.g_G_ext = 0*nS
    G_ex.g_G_rec = 0*nS
    #G_ex.n_type = 0

    # Inhibitory population
    G_in = NeuronGroup(N_in, eqs_neuron_inh, threshold = 'v>v_threshold', reset = 'v=Vr', 
                       refractory = 'refrac', method = 'euler', name='Neurongroup_in')
    G_in.Cm = Cm_in
    G_in.gL = gL_in
    G_in.refrac = tref_in
    G_in.v_threshold = Vt_in
    G_in.v = Vr
    #G_in.v = 'rand() * 10 * mV + Vr'  # Initialize from Vrest to Vthreshold from -65 to -55
    G_in.g_A_ext = 0*nS
    G_in.g_A_rec = 0*nS
    G_in.g_G_ext = 0*nS
    G_in.g_G_rec = 0*nS

# --------------- Couplings ----------------------------------------

    eqs_pre_exc_A_ext = '''
    x1 += gmax_exex * B_factor_A * (tauA1_ee - tauA2_ee)/(tauA1_ee*tauA2_ee) '''
    eqs_pre_exc_A_rec = '''
    x2 += gmax_exex * B_factor_A * (tauA1_ee - tauA2_ee)/(tauA1_ee*tauA2_ee)'''
    
    eqs_pre_exc_G_ext = '''
    w1 += gmax_exin * B_factor_G * (tauG1_ei - tauG2_ei)/(tauG1_ei*tauG2_ei)'''
    eqs_pre_exc_G_rec = '''
    w2 += gmax_exin * B_factor_G * (tauG1_ei - tauG2_ei)/(tauG1_ei*tauG2_ei)'''
    
    eqs_pre_inh_A_ext = '''
    x1 += gmax_inex * B_factor_A * (tauA1_ie - tauA2_ie)/(tauA1_ie*tauA2_ie) '''
    eqs_pre_inh_A_rec = '''
    x2 += gmax_inex * B_factor_A * (tauA1_ie - tauA2_ie)/(tauA1_ie*tauA2_ie) '''
    
    eqs_pre_inh_G_ext = '''
    w1 += gmax_inin * B_factor_G * (tauG1_ii - tauG2_ii)/(tauG1_ii*tauG2_ii)'''
    eqs_pre_inh_G_rec = '''
    w2 += gmax_inin * B_factor_G * (tauG1_ii - tauG2_ii)/(tauG1_ii*tauG2_ii)'''


# --------------------Input connections ------------------------------------
    Sinput_exex = Synapses(exc_input_to_exc, G_ex, on_pre = eqs_pre_exc_A_ext,
                                                   delay = tau_exex + tau_ax,
                                                   method ='euler',
                                                   name = 'Synapses_input_exex')
    Sinput_exex.connect(j='i')
    
    Sinput_exin = Synapses(inh_input_to_exc, G_ex, on_pre = eqs_pre_exc_G_ext,
                                                   delay = tau_exin + tau_ax,
                                                   method ='euler',
                                                   name = 'Synapses_input_exin')
    Sinput_exin.connect(j='i')
    
    Sinput_inex = Synapses(exc_input_to_inh, G_in, on_pre = eqs_pre_inh_A_ext,
                                                   delay = tau_inex + tau_ax,
                                                   method ='euler',
                                                   name = 'Synapses_input_inex')
    Sinput_inex.connect(j='i')
    
    Sinput_inin = Synapses(inh_input_to_inh, G_in, on_pre = eqs_pre_inh_G_ext,
                                                   delay = tau_inin + tau_ax,
                                                   method ='euler',
                                                   name = 'Synapses_input_inin')
    Sinput_inin.connect(j='i')
    

# ---------------------------Recurrent connections---------------------------
    S_ee = Synapses(G_ex, G_ex, on_pre ={'up': 'n=clip(n+1, 0, inf)', 
                                         'down': 'n = clip(n-1, 0, inf)',
                                         'pre': eqs_pre_exc_A_rec},
                                delay ={'up':   tau_exex + tau_ax + tauDS, 
                                        'down': tau_exex + tau_ax + tauDS + delta_t,
                                        'pre':  tau_exex + tau_ax},
                                method ='euler',
                                name = 'Synapses_ee')
    S_ee.connect(p=p_exex) # Excitatory to excitatory neurons
    
    S_ie = Synapses(G_ex, G_in, on_pre = eqs_pre_inh_A_rec,
                                delay = tau_inex + tau_ax,
                                method ='euler',
                                name = 'Synapses_ie')
    S_ie.connect(p=p_inex) # Excitatory to inhibitory neurons
    
    S_ei = Synapses(G_in, G_ex, on_pre = eqs_pre_exc_G_rec,
                                delay = tau_exin + tau_ax,
                                method ='euler',
                                name = 'Synapses_ei')
    S_ei.connect(p=p_exin) # Inhibitory to excitatory neurons
    
    S_ii = Synapses(G_in, G_in, on_pre = eqs_pre_inh_G_rec,
                                delay = tau_inin + tau_ax,
                                method ='euler',
                                name = 'Synapses_ii')
    S_ii.connect(p=p_inin) # Inhibitory to inhibitory neurons
    
#    if network_type == 'subpopulation':
#        # Excitatory subpopulation        
#        G_ex_sub = NeuronGroup(500, eqs_neuron_exc, threshold = 'v>v_threshold', reset = 'v=Vr',  
#                        refractory = 'refrac', method = 'euler', name='Neurongroup_ex_sub')
#        G_ex_sub.Cm = Cm_ex
#        G_ex_sub.gL = gL_ex
#        G_ex_sub.refrac = tref_ex
#        G_ex_sub.v_threshold = Vt_ex
#        G_ex_sub.v = Vr
#        #G_ex.v = 'rand() * 20 * mV + Vr'  # Initialize from Vrest to Vthreshold from -65 to -45
#        G_ex.g_A_ext = 0*nS
#        G_ex.g_A_rec = 0*nS
#        G_ex.g_G_ext = 0*nS
#        G_ex.g_G_rec = 0*nS
#
#        Sinput_exex_sub = Synapses(exc_input_to_exc, G_ex_sub, on_pre = eqs_pre_exc_A_ext,
#                                                       delay = tau_exex + tau_ax,
#                                                       method ='euler',
#                                                       name = 'Synapses_input_exex_sub')
#        Sinput_exex_sub.connect(j='i')
#        
#        Sinput_exin_sub = Synapses(inh_input_to_exc, G_ex_sub, on_pre = eqs_pre_exc_G_ext,
#                                                       delay = tau_exin + tau_ax,
#                                                       method ='euler',
#                                                       name = 'Synapses_input_exin_sub')
#        Sinput_exin_sub.connect(j='i')
    
# ------------Network, record and run -----------------------------------
    net = Network(collect(),  name='Network')

    net.run(pre_run_dur)

    neurons_exc = array([103])
    neurons_inh = array([50])

    M_E = SpikeMonitor(G_ex, record=True, name='Spikemonitor')
    M_I = SpikeMonitor(G_in, record=True, name='Spikemonitor2')
    R_E = PopulationRateMonitor(G_ex, name='Ratemonitor')
    R_I = PopulationRateMonitor(G_in, name='Ratemonior2')
    #DS = EventMonitor(G_ex, record=True, 'dendritic_event', name='DSpikemonitor')
    
    SM_G_ex = StateMonitor(G_ex, ('v', 'g_A_ext', 'g_A_rec', 'g_G_ext', 'g_G_rec', 'I_DS', 'n', 'y', 'z', 'r'), record=neurons_exc, name='Statemonitor')
    SM_G_in = StateMonitor(G_in, ('v', 'g_A_ext', 'g_A_rec', 'g_G_ext', 'g_G_rec'), record=neurons_inh, name='Statemonitor2')
    
    monitors = [ M_E, M_I, R_E, R_I, SM_G_ex, SM_G_in]

    # Store the current state of the network
    net.add(monitors)
    net.run(sim_dur)
    
    seed()
    
    return net, monitors

# ------------------PLOT------------------------------------------------------
'''
spik_mon = net.sorted_objects[8]

neuron_index = 0

fig, ax = plt.subplots(8,1, figsize=(6,10), sharex=True)

ax[0].plot(SM_G_ex.t/ms, SM_G_ex.g_A_ext[neuron_index,:]/nS)
ax[0].set(ylabel='nS', title='AMPA conductance external')
ax[0].grid()

ax[1].plot(SM_G_ex.t/ms, SM_G_ex.g_G_ext[neuron_index, :]/nS)
ax[1].set(ylabel='nS', title='GABA conductance external')
ax[1].grid()

ax[2].plot(SM_G_ex.t/ms,SM_G_ex.v[neuron_index,:]/mV)
ax[2].set( ylabel='mV', title='Membrane potential')
ax[2].grid()

ax[3].plot(SM_G_ex.t/ms, SM_G_ex.I_DS[neuron_index,:]/nA)
ax[3].set( ylabel='nA', title='Dendritic current')
ax[3].grid()

ax[4].plot(SM_G_ex.t/ms, SM_G_ex.r[neuron_index,:])
ax[4].set(ylabel='a.u.', title='r variable')
ax[4].grid()

ax[5].plot(SM_G_ex.t/ms, SM_G_ex.n[neuron_index, :])
ax[5].set(ylabel='a.u.', title='n number of spikes')
ax[5].grid()

ax[6].plot(SM_G_ex.t/ms, SM_G_ex.g_A_rec[neuron_index,:]/nS)
ax[6].set(ylabel='nS', title='AMPA conductance recurrent')
ax[6].grid()

ax[7].plot(SM_G_ex.t/ms, SM_G_ex.g_G_rec[neuron_index, :]/nS)
ax[7].set(ylabel='nS', title='GABA conductance recurrent')
ax[7].grid()
#ax[7].set_xticks(np.arange(0,sim_dur/ms+1,1))


fig.tight_layout()



fig, ax = plt.subplots(5,1, figsize=(6,10), sharex=True)

ax[0].plot(SM_G_in.t/ms, SM_G_in.g_A_ext[neuron_index,:]/nS)
ax[0].set(ylabel='nS', title='AMPA conductance external')
ax[0].grid()

ax[1].plot(SM_G_in.t/ms, SM_G_in.g_G_ext[neuron_index, :]/nS)
ax[1].set(ylabel='nS', title='GABA conductance external')
ax[1].grid()

ax[2].plot(SM_G_in.t/ms,SM_G_in.g_A_rec[neuron_index, :]/nS)
ax[2].set(ylabel='nS', title='AMPA conductance recurrent')
ax[2].grid()

ax[3].plot(SM_G_in.t/ms, SM_G_in.g_G_rec[neuron_index, :]/nS)
ax[3].set(ylabel='nS', title='GABA conductance recurrent')
ax[3].grid()

ax[4].plot(SM_G_in.t/ms,SM_G_in.v[neuron_index,:]/mV)
ax[4].set( ylabel='mV', title='Membrane potential')
ax[4].grid()
#ax[4].set_xticks(np.arange(0,sim_dur/ms+1,1))

fig.tight_layout()

plt.figure()
plt.plot(M_E.t/ms, M_E.i, '.k', ms=3)


'''