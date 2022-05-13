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

# ----------------------------------Initialization---------------------------

start_scope()
defaultclock.dt = 0.001*ms
sim_dur = 50*ms
pre_run_dur = 0*ms

# ------------------------------Parameter for populations-------------------------

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
Cm_in = 100 *pF
gL_in = 33 *nS
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
g_theta = 8.65 * nS
tref_D = tauDS + 2.5 * ms

# ------------------------------Parameter for couplings-------------------------

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
scale_factor = 0.1
N = int(1000*scale_factor)    # Total number of neurons
N_ex = int(0.9 * N)
N_in = N - N_ex

# ----------------------------Initial STIMULATION-----------------------------
N_input = 60  #Neurons in the input
times = zeros(N_input) * ms
seed(5)
#target_neurons = choices(arange(0,N_ex), k=N_input)
target_neurons = arange(0,N_input)

inp = SpikeGeneratorGroup(N_input, target_neurons, times)
seed()

# ---------------------------Populations --------------------------------
# the variable r is a parameter of linear decay used to set the refractory period of the dendritic spike.
eqs_neuron = '''
dv/dt = (1/Cm) * (gL * (Vr - v) + gA * (Eex - v) + gG * (Ein - v) + I_DS) : volt (unless refractory)

dI_DS/dt = 1/tauDS1 * (-I_DS + y + z) : amp 
dy/dt = -1/tauDS2 * y : amp 
dz/dt = -1/tauDS3 * z : amp 
dr/dt = -1/tref_D : 1
c = clip(1.46 - 0.053*gA/nS, 0, Inf) : 1

n : 1
Cm : farad
gL : siemens

gA : siemens
gG : siemens

refrac : second    
v_threshold : volt
#n_type : 1
'''
#
#N_input = 16  #Neurons in the input
#times = array([0, 0.2,0.4,0.6, 1,1.2,1.4,1.6, 4,4.2,4.4,4.6, 8, 8.1, 8.2, 8.3]) * ms
#inp = SpikeGeneratorGroup(N_input, arange(N_input), times)
#
#N2 = 5  #Neurons in the input
#times = array([0.5, 1, 1.5, 1.6, 2 ]) * ms
#inp2 = SpikeGeneratorGroup(N2, arange(N2), times)

# Excitatory population
seed(10)
G_ex = NeuronGroup(N_ex, eqs_neuron, threshold = 'v>v_threshold', reset = 'v=Vr',  
                refractory = 'refrac', method = 'euler', name='Neurongroup_ex',
                events={'dendritic_event': 'n>n_th and r<=0'})
G_ex.run_on_event('dendritic_event', 'r=1; y += clip(1.46 - 0.053*gA/nS, 0, Inf) * (tauDS2 - tauDS1)/tauDS2 * B; z += (1.46 - 0.053*gA/nS) * int((1.46 - 0.053*gA/nS)>0) * (tauDS1 - tauDS3)/tauDS3 * C' )

G_ex.Cm = Cm_ex
G_ex.gL = gL_ex
G_ex.refrac = tref_ex
G_ex.v_threshold = Vt_ex
G_ex.v = - 45.5*mV
#G_ex.v = 'rand() * 20 * mV + Vr'  # Initialize from Vrest to Vthreshold from -65 to -45
G_ex.n = 0
G_ex.I_DS = 0*nA
G_ex.gA = 0*nS
G_ex.gG = 0*nS
#G_ex.n_type = 0
seed()

seed(20)
G_in = NeuronGroup(N_in, eqs_neuron, threshold = 'v>v_threshold', reset = 'v=Vr', 
                   refractory = 'refrac', method = 'euler', name='Neurongroup_in')
G_in.Cm = Cm_in
G_in.gL = gL_in
G_in.refrac = tref_in
G_in.v_threshold = Vt_in
G_in.v = 'rand() * 10 * mV + Vr'  # Initialize from Vrest to Vthreshold from -65 to -55
G_in.n = 0
G_in.I_DS = 0*nA
G_in.gA = 0*nS
G_in.gG = 0*nS
#G_in.n_type = 1
seed()

# --------------- Couplings ----------------------------------------

eqs_syn_AMPA = '''
dg1/dt = - g1 / tau1 + x1  : siemens (clock-driven)
dx1/dt = - x1 / tau2 : siemens/second (clock-driven)

gA_post = g1 : siemens 

tau1 : second
tau2 : second

tau_rise1 = tau1 * tau2 / (tau1 - tau2) : second
B_factor1 = 1 / ( (tau2/tau1)**(tau_rise1/tau1) - (tau2/tau1)**(tau_rise1/tau2) ) : 1
g_max1 : siemens
'''

eqs_syn_GABA = '''
dg2/dt = - g2 / tau3 + x2  : siemens (clock-driven)
dx2/dt = - x2 / tau4 : siemens/second (clock-driven)

gG_post = g2 : siemens 

tau3 : second
tau4 : second

tau_rise2 = tau3 * tau4 / (tau3 - tau4) : second
B_factor2 = 1 / ( (tau4/tau3)**(tau_rise2/tau3) - (tau4/tau3)**(tau_rise2/tau4) ) : 1
g_max2 : siemens
'''

# Input
S_input = Synapses(inp, G_ex, model = eqs_syn_AMPA, on_pre ={'up': 'n=clip(n+1, 0, inf)', 
                                                       'down': 'n = clip(n-1, 0, inf)',
                                                       'pre': 'x1 += g_max1 * B_factor1 * (tau1 - tau2)/(tau1*tau2)'},
                                         delay ={'up':   tau_exex + tau_ax + tauDS, 
                                                 'down': tau_exex + tau_ax + tauDS + delta_t,
                                                 'pre':  tau_exex + tau_ax},
                                         method ='euler',
                                         name = 'Synapses_input')
S_input.connect(j='i') # Input to excitatory neurons
S_input.g_max1 = gmax_exex
S_input.tau1 = tauA1_ee
S_input.tau2 = tauA2_ee
#S_input.g1 = 0 * siemens
#S_input.x1 = 0 * siemens/second



# Excitatory to excitatory
seed(30)
S_ee = Synapses(G_ex, G_ex, model = eqs_syn_AMPA, on_pre ={'up': 'n=clip(n+1, 0, inf)', 
                                                       'down': 'n = clip(n-1, 0, inf)',
                                                       'pre': 'x1 += g_max1 * B_factor1 * (tau1 - tau2)/(tau1*tau2)'},
                                         delay ={'up':   tau_exex + tau_ax + tauDS, 
                                                 'down': tau_exex + tau_ax + tauDS + delta_t,
                                                 'pre':  tau_exex + tau_ax},
                                         method ='euler',
                                         name = 'Synapses_ee')
S_ee.connect(p=0.08) # Excitatory to excitatory neurons
S_ee.g_max1 = gmax_exex
S_ee.tau1 = tauA1_ee
S_ee.tau2 = tauA2_ee
S_ee.g1 = 0 * siemens
S_ee.x1 = 0 * siemens/second

seed()


# ------------Network, record and run -----------------------------------
net = Network(inp, G_ex, S_input,  name='Network')

#net.run(pre_run_dur)

neurons = np.arange(0,3,1)
#M_I = SpikeMonitor(G_in, record=True, name='Spikemonitor')
M_E = SpikeMonitor(G_ex, record=True, name='Spikemonitor2')
SM_G_ex = StateMonitor(G_ex, ('v', 'gA', 'I_DS', 'n', 'y', 'z', 'r', 'c', 'gG'), record=neurons, name='Statemonitor')
#SM_See = StateMonitor(S_ee, ('x1', 'g1'), record=neurons, name='Statemonitor2')
monitors = [ M_E, SM_G_ex]

# Store the current state of the network
net.add(monitors)
net.run(sim_dur)

# ------------------PLOT------------------------------------------------------
'''
spik_mon = net.sorted_objects[8]

neuron_index = 1

fig, ax = plt.subplots(6,1, figsize=(6,10), sharex=True)

ax[0].plot(SM_G_ex.t/ms, SM_G_ex.gA[neuron_index,:]/nS)
#ax[0].plot(Mgspike.t/ms, Mgspike.gA/nS, 'og')
ax[0].set(ylabel='nS', title='AMPA conductance')
ax[0].grid()

ax[1].plot(SM_G_ex.t/ms, SM_G_ex.gG[neuron_index, :]/nS)
#ax[1].plot(Mgspike.t/ms, Mgspike.c, 'og')
ax[1].set(ylabel='nS', title='GABA conductance')
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
ax[5].set_xticks(np.arange(0,sim_dur/ms+1,1))

fig.tight_layout()
'''