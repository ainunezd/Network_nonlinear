#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 08:56:13 2022

@author: ana_nunez
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
from brian2 import ms, mV
#----------------------MODEL 2 ---------------------------------

start_scope()
defaultclock.dt = 0.001*ms

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

A = 55 *nA
B = 64 *nA
C = 9 *nA
tauDS1 = 0.2 *ms
tauDS2 = 0.3 *ms
tauDS3 = 0.7 *ms
tauDS = 2.7 * ms
g_theta = 8.65 * nS
tref_D = tauDS + 2.5 * ms

#Parameters for couplings
p_exex = 0.08
tau_exex = 1 *ms
gmax_exex = 2.3 *nS
tauA1_ee = 2.5 *ms
tauA2_ee = 0.5 *ms
delta_t = 2 * ms
n_th = 3

# the variable r is a parameter of linear decay used to set the refractory period of the dendritic spike.
eqs_neuron = '''
dv/dt = (1/Cm) * (gL * (Vr - v) + gA * (Eex - v) + I_DS) : volt (unless refractory)

dI_DS/dt = 1/tauDS1 * (-I_DS + y + z) : amp 
dy/dt = -1/tauDS2 * y : amp 
dz/dt = -1/tauDS3 * z : amp 
dr/dt = -1/tref_D : 1

n : 1
Cm : farad
gL : siemens

gA : siemens

refrac : second    
v_threshold : volt
'''

N = 4  #Neurons in the input
times = array([0, 0,0,0]) * ms
inp = SpikeGeneratorGroup(N, arange(N), times)
G = NeuronGroup(N, eqs_neuron, threshold = 'v>v_threshold', reset = 'v=Vr',  
                refractory = 'refrac', method = 'euler', name="neurongroup",
                events={'dendritic_event': 'n>n_th and r<=0'})
#                events={'dendritic_event': 'n>n_th and update_variables ',
#                        'end_dendritic_event' : 'n<=n_th and not update_variables'})
G.run_on_event('dendritic_event', 'r=1; y += clip(1.46 - 0.053*gA/nS, 0, Inf) * (tauDS2 - tauDS1)/tauDS2 * B; z += (1.46 - 0.053*gA/nS) * int((1.46 - 0.053*gA/nS)>0) * (tauDS1 - tauDS3)/tauDS3 * C' )
#G.run_on_event('end_dendritic_event', 'update_variables=True')

G.Cm = Cm_ex
G.gL = gL_ex
G.refrac = tref_ex
G.v_threshold = Vt_ex
#G.tau1 = tauA1_ee
#G.tau2 = tauA2_ee
G.v = Vr
G.n = 0
#G.n_th = n_th
G.I_DS = 0*nA

eqs_syn = '''
dg1/dt = - g1 / tau1 + x  : siemens (clock-driven)
dx/dt = - x / tau2 : siemens/second (clock-driven)

gA_post = g1 : siemens (summed)

tau1 : second
tau2 : second
tau_rise = tau1 * tau2 / (tau1 - tau2) : second
B_factor = 1 / ( (tau2/tau1)**(tau_rise/tau1) - (tau2/tau1)**(tau_rise/tau2) ) : 1
g_max : siemens
'''

S_ee = Synapses(inp, G, model = eqs_syn, on_pre ={'up': 'n=clip(n+1, 0, inf)',
                                                  'down': 'n = clip(n-1, 0, inf)',
                                                  'pre': 'x += g_max * B_factor * (tau1 - tau2)/(tau1*tau2)'
                                                  },
#                                        on_post = {'path_event': },
#                                        on_event={'path_event':'dendritic_event'},
#                                         delay ={'up': 0*ms, 
#                                                 'down':delta_t ,
#                                                 'pre': 0*ms},
                                         delay ={'up': tau_exex + tauDS + tau_ax, 
                                                 'down': tau_exex + delta_t + tauDS + tau_ax,
                                                 'pre': tau_exex + tau_ax},
                                         method ='euler')

#S_ee = Synapses(inp, G, model = eqs_syn, on_pre ={'pre': 'x += g_max * B_factor * (tau1 - tau2)/(tau1*tau2)'
#                                                  },
#                                         delay ={'pre': tau_exex + tau_ax},
#                                         method ='euler')
#S_ee.pre = {'pre': 'x += g_max * B_factor * (tau1 - tau2)/(tau1*tau2)'}
#S_ee.pre.delay = {'up': tau_exex + tauDS + tau_ax,'down': tau_exex + delta_t + tauDS + tau_ax,'pre': tau_exex + tau_ax}

S_ee.connect(i=arange(N), j=arange(N))
S_ee.g_max = gmax_exex
S_ee.tau1 = tauA1_ee
S_ee.tau2 = tauA2_ee



spike_mon = SpikeMonitor(G)
Mgspike = EventMonitor(G, 'dendritic_event', variables=['gA', 'n'])
#M_S = StateMonitor(S_ee, ['x'], record=True)
S_mon = StateMonitor(G, ('v', 'gA', 'I_DS', 'n', 'y', 'z', 'r'), record=True)
# ...
run(30*ms)


ids_bool = True

fig, ax = plt.subplots(5,1, figsize=(6,10), sharex=True)

ax[0].plot(S_mon.t/ms, S_mon.gA[0]/nS)
ax[0].plot(Mgspike.t/ms, Mgspike.gA/nS, 'og')
ax[0].set(ylabel='nS', title='gA conductance')
ax[0].grid()

ax[1].plot(S_mon.t/ms,S_mon.v[0]/mV)
ax[1].set( ylabel='mV', title='Membrane potential')
ax[1].grid()

ax[2].plot(S_mon.t/ms, S_mon.I_DS[0]/nA)
ax[2].set( ylabel='nA', title='Dendritic current')
ax[2].grid()

ax[3].plot(S_mon.t/ms, S_mon.r[0])
ax[3].set(ylabel='a.u.', title='r variable')
ax[3].grid()

ax[4].plot(S_mon.t/ms, S_mon.n[0])
ax[4].set(ylabel='a.u.', title='n number of spikes')
ax[4].grid()
ax[4].set_xticks(np.arange(0,31,1))

fig.tight_layout()


#ax[4].plot(S_mon.t/ms, S_mon.y[0]/nA)
#ax[4].set(xlabel = 'Time (ms)', ylabel='nA', title='y auxiliar function')
#
#ax[5].plot(S_mon.t/ms, S_mon.z[0]/nA)
#ax[5].set(xlabel = 'Time (ms)', ylabel='nA', title='z auxiliar function')
##
#ax[5].plot(S_mon.t/ms, S_mon.update_variables[0])
#ax[5].set(xlabel = 'Time (ms)', ylabel='bool', title='update_variables')
#ax[5].grid()

#def simulation_model_2():
#    
    
#sim_dur = 2*second
#start_scope()
#
## Constant variables for neuron dynamics
#Vr = -65 *mV
#Eex = 0 *mV
#Ein = -75 *mV
#
#Cm_ex = 400 *pF
#gL_ex = 25 *nS
#Vt_ex= -45 *mV
#tref_ex = 3 *ms
#
#Cm_in = 100 *pF     # Change from 200 to 100 pF 
#gL_in = 33 *nS      # Changre from 25 to 33 nS
#Vt_in = -55 *mV
#tref_in = 2.5 *ms   # From 2 to 2.5 ms
#
#A = 55 *nA
#B = 64 *nA
#C = 9 *nA
#tauDS1 = 0.2 *ms
#tauDS2 = 0.3 *ms
#tauDS3 = 0.7 *ms
#tauDS = 2.7 * ms
##--------- Creation of neuron groups------------
#N = 1000 
#N_ex = int(0.9 * N)
## Define differential equation for the voltage
#eqs = '''
#dv/dt = (1/Cm) * (gL * (Vr - v) + gA * (Eex - v) + gG * (Ein - v) + I_DS) : volt (unless refractory)
#
#gA = gAee + gAie : siemens
#gG = gGei + gGii : siemens
#
#gAee: siemens
#gAie: siemens
#gGei: siemens
#gGii: siemens
#
#Cm : farad
#gL : siemens
#
#I_DS : amp
#refrac : second    
#v_threshold : volt
#'''
## Define neuron group
#G = NeuronGroup(N, eqs, threshold = 'v>v_threshold', reset = 'v=Vr', refractory = 'refrac', method = 'euler', name="neurongroup")
#
#G_ex = G[:N_ex]
#G_ex.Cm = Cm_ex
#G_ex.gL = gL_ex
#G_ex.refrac = tref_ex
#G_ex.v_threshold = Vt_ex
#
#G_in = G[N_ex:]
#G_ex.Cm = Cm_in
#G_ex.gL = gL_in
#G_ex.refrac = tref_in
#G_ex.v_threshold = Vt_in
#
##Parameters for couplings
#p_exex = 0.08
#tau_exex = 1 *ms
#gmax_exex = 2.3 *nS
##    gtheta = 8.65 *nS
#tauA1_ee = 2.5 *ms
#tauA2_ee = 0.5 *ms
#
##    trefD_ex = tauDS_exc + 2.5 *ms
#
#p_inex = 0.25            # Changre from 0.1 to 0.25
#tau_inex = 0.5 *ms
#gmax_inex = 2.85 *nS    # Changre from 3.2 to 2.85 nS
#tauA1_ie = 0.7 *ms        # Change from 3 to 0.7 ms
#tauA2_ie = 0.25 *ms     # From 0.35 to 0.25
#
#p_exin = 0.1
#tau_exin = 1 *ms
#gmax_exin = 4 * nS      # From 5 to 4 nS
#tauG1_ei = 4 *ms
#tauG2_ei = 0.3 *ms
#
#p_inin = 0.02
#tau_inin = 0.5 *ms
#gmax_inin = 4 * nS
#tauG1_ii = 2.5 *ms
#tauG2_ii = 0.4 *ms
#    
#delta_t = 2 * ms
#
## --------------Creation of synapses-----------------    
## Equations for synapses
#eqs_syn_ee = '''
#dgA1/dt = - gA1/tau1 : 1 (clock-driven)
#dgA2/dt = - gA2/tau2 : 1 (clock-driven)
#
#tau_rise = tau1 * tau2 / (tau1 - tau2) : second
#B = 1 / ( (tau2/tau1)**(tau_rise/tau1) - (tau2/tau1)**(tau_rise/tau2) ) : 1
#gAee_post = gmax * B * (gA1-gA2) : siemens (summed)
#
#gmax : siemens
#w : siemens
#tau1 : second
#tau2 : second
#'''
#
#eqs_syn_ie = '''
#dgA1/dt = - gA1/tau1 : 1 (clock-driven)
#dgA2/dt = - gA2/tau2 : 1 (clock-driven)
#
#tau_rise = tau1 * tau2 / (tau1 - tau2) : second
#B = 1 / ( (tau2/tau1)**(tau_rise/tau1) - (tau2/tau1)**(tau_rise/tau2) ) : 1
#gAie_post = gmax * B * (gA1-gA2) : siemens (summed)
#
#gmax : siemens
#w : siemens
#tau1 : second
#tau2 : second
#'''
#
#eqs_syn_ei = '''
#dgG1/dt = - gG1/tau1 : 1 (clock-driven)
#dgG2/dt = - gG2/tau2 : 1 (clock-driven)
#
#tau_rise = tau1 * tau2 / (tau1 - tau2) : second
#B = 1 / ( (tau2/tau1)**(tau_rise/tau1) - (tau2/tau1)**(tau_rise/tau2) ) : 1
#gGei_post = gmax * B * (gG1-gG2) : siemens (summed)
#
#gmax : siemens
#w : siemens
#tau1 : second
#tau2 : second
#'''
#
#eqs_syn_ii = '''
#dgG1/dt = - gG1/tau1 : 1 (clock-driven)
#dgG2/dt = - gG2/tau2 : 1 (clock-driven)
#
#tau_rise = tau1 * tau2 / (tau1 - tau2) : second
#B = 1 / ( (tau2/tau1)**(tau_rise/tau1) - (tau2/tau1)**(tau_rise/tau2) ) : 1
#gGii_post = gmax * B * (gG1-gG2) : siemens (summed)
#
#gmax : siemens
#w : siemens
#tau1 : second
#tau2 : second
#'''
#
#S_ee = Synapses(G_ex, G_ex, eqs_syn_ee, on_pre = 'gmax+=w', method ='euler', delay =tau_exex)
#S_ee.connect(p=p_exex)
#S_ee.w = gmax_exex
#S_ee.tau1 = tauA1_ee
#S_ee.tau2 = tauA2_ee
#
#S_ie = Synapses(G_ex, G_in, eqs_syn_ie, on_pre = 'gmax+=w', method ='euler', delay =tau_inex)  # From exc to inhibitory
#S_ie.connect(p=p_inex)
#S_ie.w = gmax_inex
#S_ie.tau1 = tauA1_ie
#S_ie.tau2 = tauA2_ie
#
#S_ei = Synapses(G_in, G_ex, eqs_syn_ei, on_pre = 'gmax+=w', method ='euler', delay =tau_exin)
#S_ei.connect(p=p_exin)
#S_ei.w = gmax_exin
#S_ei.tau1 = tauG1_ei
#S_ei.tau2 = tauG2_ei
#
#S_ii = Synapses(G_in, G_in, eqs_syn_ii, on_pre = 'gmax+=w', method ='euler', delay =tau_inin)
#S_ii.connect(p=p_inin)
#S_ii.w = gmax_inin
#S_ii.tau1 = tauG1_ii
#S_ii.tau2 = tauG2_ii
#
#
#net = Network(G, S_ee, S_ie, S_ei, S_ii, name="network")
#
## Run and record 
#M_I = SpikeMonitor(G_in, record=True, name="spikemonitor")
#M_E = SpikeMonitor(G_ex, record=True, name="spikemonitor2")
#SM_G = StateMonitor(G, ('v'), record=True, name="statemonitor")
#SM_S_ee = StateMonitor(S_ee, ('gAee', 'gmax', 'gA1', 'gA2'), record=True, name="statemonitor2")
#SM_S_ii = StateMonitor(S_ii, ('gGii', 'gmax', 'gG1', 'gG2'), record=True, name="statemonitor3")
#monitors = [ M_I, M_E, SM_G, SM_S_ee, SM_S_ii]
#
## Store the current state of the network
#net.add(monitors)
#net.run(sim_dur)


#def c_correction(g_value):
#    return max(1.46-0.053*g_value/nS, 0)
#
#def time_window(time_array, spike_time=50*ms):
#    y = np.zeros(len(time_array))
#    start = np.where(time_array >= spike_time-2*ms)[0][0]
#    end = np.where(time_array < spike_time)[0][-1]
#    y[start:end] = 1
#    return y
#    


#
#from brian2 import *
## Input Poisson spikes
#inp = PoissonGroup(1, rates=250*Hz)
## First group G
##eqs_G = '''
##dv/dt = (g-v)/(50*ms) : 1
##dg/dt = -g/(10*ms) : 1
##allow_gspike : boolean
##'''
## Constant variables for neuron dynamics
#Vr = -65 *mV
#Eex = 0 *mV
#Ein = -75 *mV
#
#Cm_ex = 400 *pF
#gL_ex = 25 *nS
#Vt_ex= -45 *mV
#tref_ex = 3 *ms
#
#A = 55 *nA
#B = 64 *nA
#C = 9 *nA
#tauDS1 = 0.2 *ms
#tauDS2 = 0.3 *ms
#tauDS3 = 0.7 *ms
#tauDS = 2.7 * ms
#g_theta = 8.65 *nS
#
##Parameters for couplings
#p_exex = 0.08
#tau_exex = 1 *ms
#gmax_exex = 2.3 *nS
#tauA1_ee = 2.5 *ms
#tauA2_ee = 0.5 *ms
#delta_t = 2 * ms
#n_th = 4
#
#eqs_neuron = '''
#dv/dt = (1/Cm) * (gL * (Vr - v) + gA * (Eex - v) + I_DS) : volt (unless refractory)
#dgA/dt = - gA / tau1 + x  : siemens
#dx/dt = - x / tau2 : siemens/second
#
#dI_DS/dt = 1/tauDS1 * (-I_DS + y + z) : amp  
#dy/dt = -1/tauDS2 * y : amp 
#dz/dt = -1/tauDS3 * z : amp 
#
#count : 1
#n : 1
#Cm : farad
#gL : siemens
#
#tau1 : second
#tau2 : second
#allow_gspike:boolean
#
#'''
#G = NeuronGroup(5, eqs_neuron, threshold='v>Vr',
#                reset='v = Vr; allow_gspike = True;', refractory='tref_ex',
#                events={'gspike': 'gA>g_theta and allow_gspike and n>=4',
#                        'end_gspike': 'gA<g_theta and not allow_gspike'},
#                        method ='euler')
#G.run_on_event('gspike', 'allow_gspike = False')
#G.run_on_event('end_gspike', 'allow_gspike = True')
#G.Cm = Cm_ex
#G.gL = gL_ex
#G.tau1 = tauA1_ee
#G.tau2 = tauA2_ee
#G.v = Vr
#G.n = 0
##G.n_th = n_th
#G.I_DS = 0*nA
#eqs_syn = '''
#tau_rise = tau1 * tau2 / (tau1 - tau2) : second
#B_factor = 1 / ( (tau2/tau1)**(tau_rise/tau1) - (tau2/tau1)**(tau_rise/tau2) ) : 1
#g_max : siemens
#'''
#
#
#H = NeuronGroup(1, eqs_neuron, threshold='v>Vr',
#                reset='v = Vr; allow_gspike = True;', refractory='tref_ex',
#                events={'gspike': 'gA>g_theta and allow_gspike and n>=4',
#                        'end_gspike': 'gA<g_theta and not allow_gspike'},
#                        method ='euler')
## Synapses from input Poisson group to G
#Sin = Synapses(inp, G, eqs_syn, on_pre='x_post += g_max * B_factor * (tau1 - tau2)/(tau1*tau2)')
#Sin.connect()
## Synapses from G to H
#S = Synapses(G, H, model = eqs_syn, on_pre ={'up': 'n_post+=1',
#                                                  'down': 'n_post-=1',
#                                                  'pre': 'x_post += g_max * B_factor * (tau1 - tau2)/(tau1*tau2)',
#                                                  'dend_current': 'count+=1 ; y += (1.46 - 0.053*gA/nS) * int((1.46 - 0.053*gA/nS)>0) * (tauDS2 - tauDS1)/tauDS2 * B; z += (1.46 - 0.053*gA/nS) * int((1.46 - 0.053*gA/nS)>0) * (tauDS1 - tauDS3)/tauDS3 * C'},
##                                         on_post = {'dend_current' : 'count+=1 ; y += (1.46 - 0.053*gA/nS) * int((1.46 - 0.053*gA/nS)>0) * (tauDS2 - tauDS1)/tauDS2 * B; z += (1.46 - 0.053*gA/nS) * int((1.46 - 0.053*gA/nS)>0) * (tauDS1 - tauDS3)/tauDS3 * C'}, 
#                                         on_event={'up': 'spike',
#                                                   'down': 'spike',
#                                                   'pre': 'spike',
#                                                   'dend_current': 'gspike'},
#                                         delay ={'up': 0*ms, 
#                                                 'down': delta_t,
#                                                 'pre': tau_exex,
#                                                 'dend_current': 0*ms
#                                                 },
#                                         method ='euler')
#
#S.connect()
## Monitors
#Mstate = StateMonitor(G, ('v', 'gA', 'I_DS', 'x', 'y', 'z', 'count', 'n'), record=True)
#Mgspike = EventMonitor(G, 'gspike', 'gA')
#Mspike = SpikeMonitor(G, 'v')
#MHstate = StateMonitor(H, ('v', 'gA', 'I_DS', 'x', 'y', 'z', 'count', 'n'), record=True)
## Initialise and run
#G.allow_gspike = True
#run(20*ms)
## Plot
#figure(figsize=(10, 4))
#subplot(281)
#plot(Mstate.t/ms, Mstate.gA[0], '-g', label='gA')
#plot(Mgspike.t/ms, Mgspike.gA, 'og', label='_nolegend_')
#subplot(282)
#plot(Mstate.t/ms, Mstate.v[0], '-b', lw=2, label='V')
#plot(Mspike.t/ms, Mspike.v, 'ob', label='_nolegend_')
#subplot(283)
#plot(Mstate.t/ms, Mstate.I_DS[0], '-b', lw=2, label='V')
#
#subplot(122)
#plot(MHstate.t/ms, MHstate.y[0], '-r', label='y')
#plot(MHstate.t/ms, MHstate.x[0], '-k', lw=2, label='x')
#xlabel('Time (ms)')
#title('Postsynaptic group H')
#legend(loc='best')
#tight_layout()
#show()