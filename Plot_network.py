#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 15:59:15 2022

@author: nunez
"""

'''
This file is to plot the network monitors using the function Network_model_2() from the file
Network_model_2_two_populations

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
import matplotlib.gridspec as gridspec

from Network_model_2_two_populations import Network_model_2

path_to_save_figures = '/home/nunez/Network_nonlinear/Plots/'
path_networks = '/home/nunez/Network_nonlinear/stored_networks/'

name_figures = 'network_multiply_rates_by_pop_size_cg_changing'
name_network = 'short_random_network'

dur_simulation=20
network, monitors = Network_model_2(sim_dur=dur_simulation*ms, pre_run_dur=0*ms, total_neurons=1000, scale_factor=1)
#network.store(name='rand_net', filename = path_networks + name_network)

network.restore(name='rand_net', filename = path_networks + 'long_random_network')


M_E = monitors[0]
M_I = monitors[1]
R_E = monitors[2]
R_I = monitors[3]   
SM_G_ex = monitors[4]
SM_G_in = monitors[5]

cm = 2.54
dt = defaultclock.dt
# ------------Replicate figure 2 from main paper-------------------------
fig = figure(figsize=(20/cm, 10/cm))
gs0 = gridspec.GridSpec(2, 2, figure=fig, hspace=1/cm, wspace=1/cm, height_ratios=[1,2], width_ratios=[1,1])

ax1 = fig.add_subplot(gs0[0,0])
ax2 = fig.add_subplot(gs0[0,1])
ax3 = fig.add_subplot(gs0[1,0], sharex=ax1)
ax4 = fig.add_subplot(gs0[1,1], sharex=ax2)

#bin_list = arange(0,dur_simulation+1, 1) *ms
mask = M_E.i<100 # Only plottiing 300 neurons from excitatory population
mask2 = M_I.i<100 
ax1.plot(R_I.t / ms, R_I.smooth_rate(width=1*ms)*100 / kHz, color='k')
ax1.set(ylabel='Rate [kHz]')
ax2.plot(R_E.t / ms, R_E.smooth_rate(width=1*ms)*900 / kHz, color='k')   
ax2.set(ylabel='Rate [kHz]')
ax3.plot(M_I.t[mask2]/ms, M_I.i[mask2], '.k', ms=3)
ax3.set(xlabel='Time [ms]', ylabel='Neuron index')
ax4.plot(M_E.t[mask]/ms, M_E.i[mask], '.k', ms=3)
ax4.set(xlabel='Time [ms]', ylabel='Neuron index')

savefig(path_to_save_figures+name_figures+'_rates_spikes'+'.png')



neuron_index = 0

fig, ax = plt.subplots(8,1, figsize=(24/cm,24/cm), sharex=True)

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
ax[7].set(xlabel = 'Time [ms]')
#ax[7].set_xticks(np.arange(0,sim_dur/ms+1,1))


fig.tight_layout()
savefig(path_to_save_figures+name_figures+'_exc_neuron'+'.png')



fig, ax = plt.subplots(5,1, figsize=(15/cm,24/cm), sharex=True)

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
ax[4].set(xlabel = 'Time [ms]')
#ax[4].set_xticks(np.arange(0,sim_dur/ms+1,1))

fig.tight_layout()
savefig(path_to_save_figures+name_figures+'_inh_neuron'+'.png')



