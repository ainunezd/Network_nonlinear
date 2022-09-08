#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 11:20:21 2022

@author: nunez
"""

'''
This file is to analyze different configurations of the excitation network with changing parameters

'''

# --------------------------------Imports------------------------------
import os
import numpy as np
from random import choices
import pandas as pd
from scipy import stats
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
from brian2 import ms, mV
import matplotlib.gridspec as gridspec
import cmasher as cmr
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator


from Network_model_2_two_populations import Network_model_2
from Network_ripple_analysis import find_events, define_event, prop_events
# ----------------------------names and folders-------------------------

#path_to_save_figures = '/home/ana_nunez/Documents/BCCN_berlin/Master_thesis/Plots/'
#path_networks = '/home/ana_nunez/Documents/BCCN_berlin/Master_thesis/'

path_to_save_figures = '/home/nunez/New_repo/Plots/'
path_networks = '/home/nunez/New_repo/stored_networks/'


#name_network = 'long_random_network_8_tref_ex_1'


# ----------------------------DECREASE of exc-exc delay-------------------------

def store_networks(parameter_tochange = 'tau_exex', dur_simulation = 2000):
    '''
    Function to store different networks depending on the change of parameters. 
    parameter_tochange can be tau_exex or tauDS
    tau_exex is suggested in Memmesheimer (2010) to be 1 ms
    tauDS is suggested to be 2.7 ms
    dur_simulation in ms.        I will use 2000 for the random_network_8 to observe 3 peaks.
    
    '''
    if parameter_tochange == 'tau_exex':
        range_ofchange = np.arange(0.5, 1.6, 0.1)
        for value in np.round(range_ofchange,2):
            print(parameter_tochange, value)
            network, monitors = Network_model_2(seed_num=8, sim_dur=dur_simulation*ms, pre_run_dur=0*ms, total_neurons=1000, 
                                                scale_factor=1, dendritic_interactions=True, 
                                                neurons_exc = arange(2), neurons_inh=arange(1),
                                                tau_exex = value *ms)
            network.store(name='rand_net', filename = path_networks + name_network + f'_{parameter_tochange}_{value}')
    if parameter_tochange == 'tauDS':
        range_ofchange = np.arange(2.2, 3.3, 0.1)
        for value in np.round(range_ofchange,2):
            print(parameter_tochange, value)
            network, monitors = Network_model_2(seed_num=8, sim_dur=dur_simulation*ms, pre_run_dur=0*ms, total_neurons=1000, 
                                                scale_factor=1, dendritic_interactions=True, 
                                                neurons_exc = arange(2), neurons_inh=arange(1),
                                                tauDS = value *ms)
            network.store(name='rand_net', filename = path_networks + name_network + f'_{parameter_tochange}_{value}')
    if parameter_tochange == 'tref_ex':
        range_ofchange = np.arange(2.5, 3.6, 0.1)
        for value in np.round(range_ofchange,2):
            print(parameter_tochange, value)
            network, monitors = Network_model_2(seed_num=8, sim_dur=dur_simulation*ms, pre_run_dur=0*ms, total_neurons=1000, 
                                                scale_factor=1, dendritic_interactions=True, 
                                                neurons_exc = arange(2), neurons_inh=arange(1),
                                                tref_ex = value *ms)
            network.store(name='rand_net', filename = path_networks + name_network + f'_{parameter_tochange}_{value}')
    if parameter_tochange == 'tref_in':
        range_ofchange = np.arange(1.5, 2.6, 0.1)
        for value in np.round(range_ofchange,2):
            print(parameter_tochange, value)
            network, monitors = Network_model_2(seed_num=8, sim_dur=dur_simulation*ms, pre_run_dur=0*ms, total_neurons=1000, 
                                                scale_factor=1, dendritic_interactions=True, 
                                                neurons_exc = arange(2), neurons_inh=arange(1),
                                                tref_in = value *ms)
            network.store(name='rand_net', filename = path_networks + name_network + f'_{parameter_tochange}_{value}')
    
def store_networks_texex_tDS(dur_simulation = 2000):
    for param in ['tref_ex', 'tref_in']:
        store_networks(parameter_tochange = param, dur_simulation = dur_simulation)


#name_network = 'long_random_network_8_tau_exex_0.9'
#network, monitors = Network_model_2(seed_num=8, sim_dur=10*ms, pre_run_dur=0*ms, total_neurons=1000, 
#                                    scale_factor=1, dendritic_interactions=True, 
#                                    neurons_exc = arange(2), neurons_inh=arange(1))
#
#
network.restore(name='rand_net', filename = path_networks + name_network)

# Get monitors from the network
M_E = network.sorted_objects[24]
M_I = network.sorted_objects[25]
M_DS = network.sorted_objects[-16]
R_E = network.sorted_objects[-1]
R_I = network.sorted_objects[-2]  
State_G_ex = network.sorted_objects[2]
State_G_in = network.sorted_objects[3]
G_E = network.sorted_objects[0]
G_I = network.sorted_objects[1]

dt = 0.001
cm = 2.54
