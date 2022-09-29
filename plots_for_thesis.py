#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 17:18:15 2022

@author: ana_nunez
"""

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
from matplotlib.ticker import MaxNLocator
from sklearn.svm import SVR


from Network_model_2_two_populations import Network_model_2
#from functions_from_Natalie import f_oscillation_analysis_transient
#from Network_ripple_analysis import find_events, prop_events

# ----------------------------names and folders-------------------------

path_to_save_figures = '/home/ana_nunez/Documents/BCCN_berlin/Master_thesis/Plots/'
path_figures_thesis = '/home/ana_nunez/Documents/BCCN_berlin/Master_thesis/Plots_thesis/'
path_networks = '/home/ana_nunez/Documents/BCCN_berlin/Master_thesis/'

general_name_networks = 'long_50000_network_tauax_change_distrib_.pkl'

name_dataframe_all = 'Properties_ALL_events_' + general_name_networks
name_dataframe_exc = 'Properties_exitatory_ALL_events_selected_' +  general_name_networks
name_dataframe_inh = 'Properties_inhibitory_ALL_events_selected_' +  general_name_networks

dt = 0.01
cm = 2.54

def plot_events_statistics(population='Excitatory', lenght_events='short'):
    '''
    Function to plot different histograms of values from events
    print(also total_events)
    The parameters that will be plotted: duration, mean_freq, num_peaks, max_pop_rate
        
    population can be:
        Excitatory
        Inhibitory
    lenght_events can be:
        short
        long
        None
    depending on num_peaks. <=13 are short events
    '''
    if population=='Excitatory': df_population = pd.read_pickle(path_to_save_figures + name_dataframe_exc)
    elif population=='Inhibitory': df_population = pd.read_pickle(path_to_save_figures + name_dataframe_inh)
    else: return None
    if lenght_events=='short': df_population = df_population[df_population.Num_peaks<13]
    elif lenght_events == 'long': df_population = df_population[df_population.Num_peaks>=13]
    else: pass
    
    values_duration = df_population['Duration'].values
    values_numpeaks = df_population['Num_peaks'].values
    values_freq = df_population['Mean_frequency'].values
    values_maxpop_rate = np.zeros(len(values_duration))
    for j, index in enumerate(df_population.index):
        values_maxpop_rate[j] = np.max(df_population.loc[index]['Peaks_heights'])
        
    pdf_file_name = f'All_events_histograms_overview_{population}_{lenght_events}' + general_name_networks[:-5]
    with PdfPages(path_figures_thesis + pdf_file_name + '.pdf') as pdf:        
        fig, ax = plt.subplots(1, 4, figsize=(21/cm, 10/cm))
                
        ax[0].hist(values_duration, bins=50, histtype='step', color='k')
        ax[0].set(xlabel='Duration of event [ms]')
        ax[1].hist(values_numpeaks, bins=50, histtype='step', color='k')
        ax[1].set(xlabel='Peaks/event')
        ax[2].hist(values_maxpop_rate, bins=50, histtype='step', color='k')
        ax[2].set(xlabel='Max peak height [kHz]')
        ax[3].hist(values_freq, bins=50, histtype='step', color='k')
        ax[3].set(xlabel='Mean frequency [Hz]')
        ax[3].axvline(x=np.mean(values_freq), color='gray', linestyle='--')
        print(f'Mean frequency: {np.round(np.mean(values_freq),2)}')
        
        fig.tight_layout()
        pdf.savefig(fig) 
#        plt.close()    
def plot_histograms_overview_statistics(dict_information, n_group, threshold_in_sd=3):
    '''
    Function to plot histograms of the overview of all events. 
    In principle: Duration of event, number of peaks, height of max peak and mean frequency
    '''
    num_events = len(dict_information.keys())
    num_peaks = np.zeros(num_events)
    peak_max_heights = np.zeros(num_events)
    durations = np.zeros(num_events)
    mean_freq = np.zeros(num_events)

    for i, event in enumerate(dict_information.keys()):
        num_peaks[i] = dict_information[event]['Num_peaks']
        durations[i] = dict_information[event]['Duration']
        mean_freq[i] = dict_information[event]['Mean_frequency']
        peak_max_heights[i] = np.max(dict_information[event]['Peaks_heights'])

    pdf_file_name = f'All_events_histograms_overview_G_{n_group.name[-2].upper()}_tr_{threshold_in_sd}_{name_network}'
    with PdfPages( path_to_save_figures + pdf_file_name + '.pdf') as pdf:
        fig, ax = plt.subplots(1, 4, figsize=(21/cm,10/cm))
        
        ax[0].hist(durations, 20, color='k', histtype='step')
        ax[0].set(xlabel='Duration of event [ms]')
        ax[0].axvline(x=np.mean(durations), ymin=0, ymax=ax[0].get_ylim()[1], color='gray', linestyle='--')
        
        ax[1].hist(num_peaks, 20, color='k', histtype='step')
        ax[1].set(xlabel='Peaks/event')
        ax[1].axvline(x=np.mean(num_peaks), ymin=0, ymax=ax[1].get_ylim()[1], color='gray', linestyle='--')
        
        ax[2].hist(peak_max_heights, 20, color='k', histtype='step')
        ax[2].set(xlabel='Maximum peak height [kHz]')
        ax[2].axvline(x=np.mean(peak_max_heights), ymin=0, ymax=ax[2].get_ylim()[1], color='gray', linestyle='--')

        ax[3].hist(mean_freq, 20, color='k', histtype='step')
        ax[3].set(xlabel='Mean frequency [Hz]')
        ax[3].axvline(x=np.mean(mean_freq), ymin=0, ymax=ax[3].get_ylim()[1], color='gray', linestyle='--')
        
        fig.tight_layout()
        plt.show()
        savefig(path_to_save_figures + pdf_file_name +'.png')
        pdf.savefig(fig)
        
def plot_example_of_event(num_event_all=116):
    '''
    Fucntion to plot one event. The input should be the number of event from the dataframe where exc and inh are collected.
    '''
    df_all = pd.read_pickle(path_to_save_figures + name_dataframe_all)
    df_event = df_all.loc[num_event_all]
    num_event_ex = int(df_event['Event_ex'])
    num_event_in = int(df_event['Event_in'])
    
    time_exc, signal_exc, peaktimes_ex = df_event['Time_exc'], df_event['Signal_exc'], df_event['Peaktimes_exc']
    time_inh, signal_inh, peaktimes_in = df_event['Time_inh'], df_event['Signal_inh'], df_event['Peaktimes_inh']
    peakvalues_ex, peakvalues_in = np.zeros(len(peaktimes_ex)), np.zeros(len(peaktimes_in))
    for i, aa in enumerate(peaktimes_ex): peakvalues_ex[i] = signal_exc[np.where(time_exc == aa)[0]]
    for i, aa in enumerate(peaktimes_in): peakvalues_in[i] = signal_inh[np.where(time_inh == aa)[0]]
    
    pdf_file_name = f'Event_{num_event_all}_' + general_name_networks[:-5]
    with PdfPages(path_figures_thesis + pdf_file_name + '.pdf') as pdf:        
        fig, ax = plt.subplots(1, 1, figsize=(14/cm, 8/cm))
                
        ax.plot(time_exc, signal_exc, color='navy')
        ax.plot(time_inh, signal_inh, color='gold')
        ax.scatter(peaktimes_ex, peakvalues_ex, color='r', marker='.')
        ax.scatter(peaktimes_in, peakvalues_in, color='r', marker='*')
        ax.set(xlabel='Time w.r.t. max peak [ms]', ylabel='Smoothed population rates [kHz]')
        
        fig.tight_layout()
        pdf.savefig(fig) 
    
def plot_section_of_simulation(start_time=100, end_time=200, window=0.5, smoothing=False):
    '''
    Function to plot only one event of the network to reproduce Memmesheimer's figure 3
    start_time in ms. For the start time of simulation
    end_time in ms. For the end time to plot simulation
    window can be any value and if smoothing is true it will be the std of the kernel, if smoothing 
        is false then it will be the bin size
    '''
    name_network = 'long_50000_network_tauax_change_distrib_1' #Neurons false
    dur_simulation=5
    network, monitors = Network_model_2(seed_num=1001, sim_dur=dur_simulation*ms, pre_run_dur=0*ms, total_neurons=1000, 
                                        scale_factor=1, dendritic_interactions=True, neurons_exc = False, neurons_inh = False,
                                        tau_ax_method = 'distance_dist', resolution=0.01)

    network.restore(name='rand_net', filename = path_networks + name_network)

    ## Get monitors from the network
    M_E = network.sorted_objects[24]
    M_I = network.sorted_objects[25]
    M_DS = network.sorted_objects[-16]
    R_E = network.sorted_objects[-1]
    R_I = network.sorted_objects[-2]  
    State_G_ex = network.sorted_objects[2]
    State_G_in = network.sorted_objects[3]
    G_E = network.sorted_objects[0]
    G_I = network.sorted_objects[1]
    ##
    dt = 0.01
    cm = 2.54
    
        
    spike_times_ex = M_E.t/ms
    mask_ex = np.where((spike_times_ex>=start_time)&(spike_times_ex<=end_time))[0]
    spike_index = M_E.i
    spike_times_in = M_I.t/ms
    mask_in = np.where((spike_times_in>=start_time)&(spike_times_in<=end_time))[0]
    time = np.arange(start_time, end_time, dt)
    
#    time = R_E.t/ms
    bin_list = np.arange(start_time, end_time+window, window)
    
    kernel = stats.norm.pdf(np.arange(-3,3,dt), loc=0, scale=window)
    hist_rate_ex, _ = np.histogram(spike_times_ex[mask_ex], bins=np.arange(start_time, end_time+dt, dt))
    smooth_spike_rate_ex = np.convolve(hist_rate_ex, kernel, mode='same')
    hist_rate_in, _ = np.histogram(spike_times_in[mask_in], bins=np.arange(start_time, end_time+dt, dt))
    smooth_spike_rate_in = np.convolve(hist_rate_in, kernel, mode='same')    
    
    pdf_file_name = f'Summary_rates_and_spikes_smooth_{int(smoothing)}_{window}_' + name_network
    with PdfPages(path_figures_thesis + pdf_file_name + '.pdf') as pdf:    
        fig, ax= plt.subplots(3, 1, figsize=(10.5/cm, 10.5/cm), gridspec_kw={'height_ratios': [1, 1, 2]}, sharex=True)
        
        if smoothing:
            ax[0].plot(time, smooth_spike_rate_ex*window, color='k')
            ax[1].plot(time, smooth_spike_rate_in*window, color='k')
            ax[0].set(ylim=(0,40), ylabel='Rate [kHz]')
            ax[1].set(ylim=(0,40), ylabel='Rate [kHz]')
        else:
            ax[0].hist(spike_times_ex[mask_ex], bins=bin_list, color='k', histtype='step')
            ax[1].hist(spike_times_in[mask_in], bins=bin_list, color='k', histtype='step')
            ax[0].set(ylim=(0,40), ylabel='Rate [sp/bin]')
            ax[1].set(ylim=(0,40), ylabel='Rate [sp/bin]')
        
        ax[2].scatter(spike_times_ex[mask_ex], spike_index[mask_ex], marker='.', color='k')
        

        ax[2].set(xlim=(start_time,end_time), xlabel='Time [ms]', ylabel='Neuron')
#        ax1.set_xticks([])
#        ax2.set_xticks([])
        
        fig.tight_layout()
        pdf.savefig(fig) 
    
    
    
    

