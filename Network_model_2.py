#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 11:42:03 2022

@author: ana_nunez
"""

# --------------------------------Imports------------------------------
import os
import numpy as np
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
from brian2 import ms, mV

# ----------------------------------Initialization---------------------------

start_scope()
defaultclock.dt = 0.001*ms
sim_dur = 20*ms

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
scale_factor = 0.5
N = int(1000*scale_factor)    # Total number of neurons
N_ex = int(0.9 * N)

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
n_type : 1
'''
#
#N = 16  #Neurons in the input
#times = array([0, 0.2,0.4,0.6, 1,1.2,1.4,1.6, 4,4.2,4.4,4.6, 8, 8.1, 8.2, 8.3]) * ms
#inp = SpikeGeneratorGroup(N, arange(N), times)
#
#N2 = 5  #Neurons in the input
#times = array([0.5, 1, 1.5, 1.6, 2 ]) * ms
#inp2 = SpikeGeneratorGroup(N2, arange(N2), times)

G = NeuronGroup(N, eqs_neuron, threshold = 'v>v_threshold', reset = 'v=Vr',  
                refractory = 'refrac', method = 'euler', name="neurongroup",
                events={'dendritic_event': 'n>n_th and r<=0'})
#                events={'dendritic_event': 'n>n_th and update_variables ',
#                        'end_dendritic_event' : 'n<=n_th and not update_variables'})
G_ex = G[:N_ex]
G_in = G[N_ex:]
G_ex.run_on_event('dendritic_event', 'r=1; y += clip(1.46 - 0.053*gA/nS, 0, Inf) * (tauDS2 - tauDS1)/tauDS2 * B; z += (1.46 - 0.053*gA/nS) * int((1.46 - 0.053*gA/nS)>0) * (tauDS1 - tauDS3)/tauDS3 * C' )
#G.run_on_event('end_dendritic_event', 'update_variables=True')

# Excitatory population
G_ex.Cm = Cm_ex
G_ex.gL = gL_ex
G_ex.refrac = tref_ex
G_ex.v_threshold = Vt_ex
G_ex.v = Vr
G_ex.n = 0
G_ex.I_DS = 0*nA
G_ex.gA = 0*nS
G_ex.gG = 0*nS
G_ex.n_type = 0

G_in.Cm = Cm_in
G_in.gL = gL_in
G_in.refrac = tref_in
G_in.v_threshold = Vt_in
G_in.v = Vr
G_in.n = 0
G_in.I_DS = 0*nA
G_in.gA = 0*nS
G_in.gG = 0*nS
G_in.n_type = 1


# --------------- Couplings ----------------------------------------

eqs_syn_AMPA = '''
dg/dt = - g / tau1 + x  : siemens (clock-driven)
dx/dt = - x / tau2 : siemens/second (clock-driven)

gA_post = g : siemens (summed)

tau1 : second
tau2 : second

tau_rise = tau1 * tau2 / (tau1 - tau2) : second
B_factor = 1 / ( (tau2/tau1)**(tau_rise/tau1) - (tau2/tau1)**(tau_rise/tau2) ) : 1
g_max : siemens
'''

eqs_syn_GABA = '''
dg/dt = - g / tau1 + x  : siemens (clock-driven)
dx/dt = - x / tau2 : siemens/second (clock-driven)

gG_post = g : siemens (summed)

tau1 : second
tau2 : second

tau_rise = tau1 * tau2 / (tau1 - tau2) : second
B_factor = 1 / ( (tau2/tau1)**(tau_rise/tau1) - (tau2/tau1)**(tau_rise/tau2) ) : 1
g_max : siemens
'''
# Excitatory to excitatory
seed(20)
S_ee = Synapses(G, G, model = eqs_syn_AMPA, on_pre ={'up': 'n=clip(n+1, 0, inf)', 
                                                       'down': 'n = clip(n-1, 0, inf)',
                                                       'pre': 'x += g_max * B_factor * (tau1 - tau2)/(tau1*tau2)'},
                                         delay ={'up':   tau_exex + tau_ax + tauDS, 
                                                 'down': tau_exex + tau_ax + tauDS + delta_t,
                                                 'pre':  tau_exex + tau_ax},
                                         method ='euler',
                                         name = 'Synapses_ee')
S_ee.connect('n_type_pre == 0 and n_type_post == 0', p=p_exex) # Excitatory to excitatory neurons
S_ee.g_max = gmax_exex
S_ee.tau1 = tauA1_ee
S_ee.tau2 = tauA2_ee
S_ee.g = 0 * siemens
S_ee.x = 0 * siemens/second

# Excitatory to inhibitory
S_ie = Synapses(G, G, model = eqs_syn_AMPA, on_pre ={'pre': 'x += g_max * B_factor * (tau1 - tau2)/(tau1*tau2)'},
                                         delay ={'pre': tau_exin + tau_ax},
                                         method ='euler',
                                         name = 'Synapses_ie')
S_ie.connect('n_type_pre == 0 and n_type_post == 1', p=p_inex) # Excitatory to inhibitory neurons
S_ie.g_max = gmax_inex
S_ie.tau1 = tauA1_ie
S_ie.tau2 = tauA2_ie
S_ie.g = 0 * siemens
S_ie.x = 0 * siemens/second


# Inhibitory to excitatory
S_ei = Synapses(G, G, model = eqs_syn_GABA, on_pre ={'pre': 'x += g_max * B_factor * (tau1 - tau2)/(tau1*tau2)'},
                                         delay ={'pre': tau_exin + tau_ax},
                                         method ='euler',
                                         name = 'Synapses_ei')
S_ei.connect('n_type_pre == 1 and n_type_post == 0', p=p_exin) # Inhibitory to excitatory neurons
S_ei.g_max = gmax_exin
S_ei.tau1 = tauG1_ei
S_ei.tau2 = tauG2_ei
S_ei.g = 0 * siemens
S_ei.x = 0 * siemens/second

# Inhibitory to inhibitory
S_ii = Synapses(G, G, model = eqs_syn_GABA, on_pre ={'pre': 'x += g_max * B_factor * (tau1 - tau2)/(tau1*tau2)'},
                                         delay ={'pre': tau_exin + tau_ax},
                                         method ='euler',
                                         name = 'Synapses_ii')
S_ii.connect('n_type_pre == 1 and n_type_post == 1', p=p_inin) # Inhibitory to excitatory neurons
S_ii.g_max = gmax_inin
S_ii.tau1 = tauG1_ii
S_ii.tau2 = tauG2_ii
S_ii.g = 0 * siemens
S_ii.x = 0 * siemens/second
seed()

# ------------Network, record and run -----------------------------------
net = Network(G, S_ee, S_ei, S_ie, S_ii, name='Network')

M_I = SpikeMonitor(G_in, record=True, name='Spikemonitor')
M_E = SpikeMonitor(G_ex, record=True, name='Spikemonitor2')
SM_G = StateMonitor(G, ('v', 'gA', 'I_DS', 'n', 'y', 'z', 'r', 'c', 'gG'), record=(), name='Statemonitor')
SM_See = StateMonitor(S_ee, ('x', 'g'), record=(), name='Statemonitor2')
monitors = [ M_I, M_E, SM_G, SM_See]

# Store the current state of the network
net.add(monitors)
net.run(sim_dur)
  
#spike_mon = SpikeMonitor(G)
#Mgspike = EventMonitor(G, 'dendritic_event', variables=['gA', 'n', 'c', 'gG'])
#M_S = StateMonitor(S_ee, ['x'], record=True)
#S_mon = StateMonitor(G, ('v', 'gA', 'I_DS', 'n', 'y', 'z', 'r', 'c', 'gG'), record=True)
## ...
#run(20*ms)

#
#ids_bool = True

fig, ax = plt.subplots(6,1, figsize=(6,10), sharex=True)

ax[0].plot(S_mon.t/ms, S_mon.gA[0]/nS)
ax[0].plot(Mgspike.t/ms, Mgspike.gA/nS, 'og')
ax[0].set(ylabel='nS', title='AMPA conductance')
ax[0].grid()

ax[1].plot(S_mon.t/ms, S_mon.gG[0]/nS)
#ax[1].plot(Mgspike.t/ms, Mgspike.c, 'og')
ax[1].set(ylabel='nS', title='GABA conductance')
ax[1].grid()

ax[2].plot(S_mon.t/ms,S_mon.v[0]/mV)
ax[2].set( ylabel='mV', title='Membrane potential')
ax[2].grid()

ax[3].plot(S_mon.t/ms, S_mon.I_DS[0]/nA)
ax[3].set( ylabel='nA', title='Dendritic current')
ax[3].grid()

ax[4].plot(S_mon.t/ms, S_mon.r[0])
ax[4].set(ylabel='a.u.', title='r variable')
ax[4].grid()

ax[5].plot(S_mon.t/ms, S_mon.n[0])
ax[5].set(ylabel='a.u.', title='n number of spikes')
ax[5].grid()
ax[5].set_xticks(np.arange(0,sim_dur/ms+1,1))

fig.tight_layout()
