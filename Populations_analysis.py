#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 14:02:43 2022

@author: nunez
"""
'''
File to analyze the two populations in the network. 
Excitatory and inhibitory populations and the interactions between them:
    Correlations, 
    Percetange spikes per event
    Frequencies and period
    

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
from sklearn.svm import SVR


from Network_model_2_two_populations import Network_model_2
from functions_from_Natalie import f_oscillation_analysis_transient
from Network_ripple_analysis import find_events, prop_events

# ----------------------------names and folders-------------------------

#path_to_save_figures = '/home/ana_nunez/Documents/BCCN_berlin/Master_thesis/Plots/'
#path_networks = '/home/ana_nunez/Documents/BCCN_berlin/Master_thesis/'

path_to_save_figures = '/home/nunez/New_repo/Plots_populations_comparison/'
path_networks = '/home/nunez/New_repo/stored_networks/'


##name_figures = 'network_multiply_rates_by_pop_size_cg_changing'
#
##name_network = 'long_baseline_no_dendritic_8'
#name_network = 'long_random_network_8'
name_network = 'long_50000_network_8' #Neurons false
#name_network = 'long_allneurons_event_network_3'  # For the event and all neurons the prerun is 1600 ms. Neurons true
#
## In order to restore the long network it is necessary to first create an object.
## Therefore, here I create a network of only 20 ms and on it, I restore the long one.
dur_simulation=10
network, monitors = Network_model_2(seed_num=1001, sim_dur=dur_simulation*ms, pre_run_dur=0*ms, total_neurons=1000, 
                                    scale_factor=1, dendritic_interactions=True, neurons_exc = False, neurons_inh = False)
##network.store(name='rand_net', filename = path_networks + name_network)

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
#
dt = 0.001
cm = 2.54


#events_dict_ex, simulation_dict_ex = prop_events(G_E, R_E, 3, plot_peaks_bool=False)
#df_ex = pd.DataFrame.from_dict(events_dict_ex, orient='index')
#df_ex.to_csv(path_to_save_figures+f'Properties_events_ex_{name_network}.csv')
#df_ex.to_pickle(path_to_save_figures+f'Properties_events_ex_{name_network}.pkl')
#
#events_dict_in, simulation_dict_in = prop_events(G_I, R_I, 3, plot_peaks_bool=False)
#df_in = pd.DataFrame.from_dict(events_dict_in, orient='index')
#df_in.to_csv(path_to_save_figures+f'Properties_events_in_{name_network}.csv')
#df_in.to_pickle(path_to_save_figures+f'Properties_events_in_{name_network}.pkl')


def plot_both_population_rates(threshold_in_sd=3, name_net=name_network, smoothing = 0.5, baseline_start=0, baseline_end=400, dt=dt):
    
    rate_signal_ex = R_E.smooth_rate(width=smoothing*ms)*len(G_E) / kHz
    rate_signal_in = R_I.smooth_rate(width=smoothing*ms)*len(G_I) / kHz
#    thr = mean(rate_signal[:int(400/dt)]) + threshold_in_sd * std(rate_signal) 
    thr_ex = mean(rate_signal_ex[int(baseline_start/dt):int(baseline_end/dt)]) + threshold_in_sd * std(rate_signal_ex) 
    thr_in = mean(rate_signal_in[int(baseline_start/dt):int(baseline_end/dt)]) + threshold_in_sd * std(rate_signal_in) 
    peaks_ex, _ = signal.find_peaks(rate_signal_ex, height = thr_ex, distance = 100/dt)
    peaks_in, _ = signal.find_peaks(rate_signal_in, height = thr_in, distance = 100/dt)
    time = R_E.t / ms
    pdf_file_name = f'Population_rates_tr_{threshold_in_sd}_{name_net}_windowsmooth_{smoothing}'
    with PdfPages( path_to_save_figures + pdf_file_name + '.pdf') as pdf:
        fig, ax = plt.subplots(2,1, figsize=(21/cm, 14/cm), sharex=True, sharey=True)
        ax[0].plot(time, rate_signal_ex, c='navy', label='Excitatory population')
        ax[0].scatter(time[0] + peaks_ex*dt, rate_signal_ex[peaks_ex], c='r')
        ax[0].axhline(y=thr_ex, c='gray', linestyle='--')
        ax[0].legend(loc=1)
        ax[0].set(ylabel='Rates [kHz]', xlim=(min(time),max(time)))        
        ax[1].plot(time, rate_signal_in, c='gold', label='Inhibitory population')
        ax[1].axhline(y=thr_in, c='gray', linestyle='--')
        ax[1].scatter(time[0] + peaks_in*dt, rate_signal_in[peaks_in], c='r')
        ax[1].legend(loc=1)
        ax[1].set(xlabel='Time [ms]', ylabel='Rates [kHz]', xlim=(min(time),max(time)))
        savefig(path_to_save_figures + pdf_file_name +'.png')
#        savefig(path_to_save_figures + pdf_file_name +'.eps')
        pdf.savefig(fig)


def get_events_ex_in(name_net=name_network):
    
    data_ex = pd.read_pickle(path_to_save_figures+f'Properties_events_ex_{name_network}.pkl')
    data_in = pd.read_pickle(path_to_save_figures+f'Properties_events_in_{name_network}.pkl')
    headers = ['Event_id', 'Event_ex', 'Event_in','Difference_max_peak_time', 'Time_exc', 'Time_inh', 'Signal_exc', 'Signal_inh', 
               'Peaktimes_exc', 'Peaktimes_inh',  'Difference_between_peaks', 'Zeropeak_index']
    events_ex_in = pd.DataFrame(columns = headers)
    index = 0
    
    max_peaks_ex = np.array(data_ex['Max_peak_time'].values)
    max_peaks_in = np.array(data_in['Max_peak_time'].values)
    array_index_ex = np.array(data_ex.index)
    array_index_in = np.array(data_in.index)
    
    list_events = np.zeros((1,2))
    
    for j, peak_ex in enumerate(max_peaks_ex):
        index_ex = array_index_ex[j]
        indexes_in_in = np.where(abs(max_peaks_in-peak_ex)<10)[0]
        if len(indexes_in_in)>0:
            list_events = np.append(list_events,np.array([[index_ex, array_index_in[indexes_in_in[0]]]]), axis=0)
        else: continue
    
    events_both = list_events[1:,:]    
    for evs in np.arange(events_both.shape[0]):
        a,b = events_both[evs,:]
        print(a,b)
        diff_peak_times = np.round(data_ex.loc[a]['Max_peak_time'] - data_in.loc[b]['Max_peak_time'], 3)
        if diff_peak_times>10:continue
        time_exc = data_ex.loc[a]['Time_array']
        time_inh = data_in.loc[b]['Time_array'] - diff_peak_times
        signal_exc = data_ex.loc[a]['Signal_array']
        signal_inh = data_in.loc[b]['Signal_array']
        peak_times_exc = time_exc[data_ex.loc[a]['Peaks_indexes']]
        peak_times_inh = time_inh[data_in.loc[b]['Peaks_indexes']]
        first_peaks_inh = np.where((peak_times_inh - np.min(peak_times_exc))>0)[0]
        if len(first_peaks_inh)>0: start_time = np.round(peak_times_exc[np.where(peak_times_exc<peak_times_inh[first_peaks_inh[0]])[0][-1]] -1, 3)
        last_peaks_exc = np.where((peak_times_exc - np.max(peak_times_inh))<0)[0]
        if len(last_peaks_exc)>0: end_time = np.round(peak_times_inh[np.where(peak_times_inh>peak_times_exc[last_peaks_exc[-1]])[0][0]] +1,3)
        index_start_exc, index_end_exc = np.where(time_exc>=start_time)[0][0], np.where(time_exc<=end_time)[0][-1]
        index_start_inh, index_end_inh = np.where(time_inh>=start_time)[0][0], np.where(time_inh<=end_time)[0][-1]
        min_time, max_time = time_exc[index_start_exc], time_inh[index_end_inh]
        time_exc, signal_exc = time_exc[index_start_exc:index_end_exc], signal_exc[index_start_exc:index_end_exc]
        time_inh, signal_inh = time_inh[index_start_inh:index_end_inh], signal_inh[index_start_inh:index_end_inh]
        peak_times_exc = peak_times_exc[np.where((peak_times_exc>min_time) & (peak_times_exc<max_time))[0]]
        peak_times_inh = peak_times_inh[np.where((peak_times_inh>min_time) & (peak_times_inh<max_time))[0]]
        zero_index = np.where(peak_times_exc ==0)[0]
        if len(zero_index)>0: zero_peak_index = zero_index[0]
        else: zero_peak_index = None
        if (len(peak_times_exc) != len(peak_times_inh)): continue
        else: events_ex_in.loc[index] = [evs+1,a,b, diff_peak_times,time_exc, time_inh, signal_exc, signal_inh, peak_times_exc,
                              peak_times_inh, peak_times_inh-peak_times_exc, zero_peak_index]
        index +=1

    events_ex_in.to_csv(path_to_save_figures+f'Properties_populations_events_in_{name_network}.csv')
    events_ex_in.to_pickle(path_to_save_figures+f'Properties_populations_events_in_{name_network}.pkl')



def check_average_shapes_both_pop(name_net=name_network):
    '''
    Function to corrobotare figure S5 B from Memmesheimmer supplementary material
    '''
    long_bad_events = [1,6,11,23,31,34, 4,10,12,13,14,21,30,38,39,43,46]
    events_ex_in = pd.read_pickle(path_to_save_figures+f'Properties_populations_events_in_{name_network}.pkl')
    mask = np.in1d(events_ex_in['Event_ex'].values, long_bad_events)
    events_index = events_ex_in['Event_ex'].index[~mask]

    pdf_file_name = f'Population_rates_averages_only_shortEvents_{name_net}'
    with PdfPages( path_to_save_figures + pdf_file_name + '.pdf') as pdf:
        fig, ax = plt.subplots(1,1, figsize=(21/cm, 14/cm))
        for evs in events_index:
            time_exc = events_ex_in.loc[evs]['Time_exc']
            time_inh = events_ex_in.loc[evs]['Time_inh']
            signal_exc = events_ex_in.loc[evs]['Signal_exc']
            signal_inh = events_ex_in.loc[evs]['Signal_inh']
            ax.plot(time_exc, signal_exc, color='navy')
            ax.plot(time_inh, signal_inh, color='gold')
            ax.axvline(x=0, c='gray', linestyle='--')
            ax.set(ylabel='Rates [kHz]',xlabel='Time w.r.t. max peak [ms]')        

        savefig(path_to_save_figures + pdf_file_name +'.png')
        pdf.savefig(fig)


    
    
def plot_all_accepted_events(name_net=name_network):
    data_ex = pd.read_pickle(path_to_save_figures+f'Properties_events_ex_{name_network}.pkl')
    data_in = pd.read_pickle(path_to_save_figures+f'Properties_events_in_{name_network}.pkl')
    events_ex_in = pd.read_pickle(path_to_save_figures+f'Properties_populations_events_in_{name_network}.pkl')
    
    events_ex = events_ex_in['Event_ex'].values
    events_in = events_ex_in['Event_in'].values
    event_array = np.vstack((events_ex, events_in)).T
    pdf_file_name = f'All_accepted_events_{name_net}_2'
    with PdfPages( path_to_save_figures + pdf_file_name + '.pdf') as pdf:
        for x in np.arange(event_array.shape[0]):
            a,b = event_array[x,:]
            fig = plt.figure(figsize=(21/cm, 12/cm))
            diff_peak_times = np.round(data_ex.loc[a]['Max_peak_time'] - data_in.loc[b]['Max_peak_time'], 3)
            time_exc = data_ex.loc[a]['Time_array']
            time_inh = data_in.loc[b]['Time_array'] - diff_peak_times
            signal_exc =  data_ex.loc[a]['Signal_array']
            signal_inh =  data_in.loc[b]['Signal_array']
            peak_times_exc = time_exc[data_ex.loc[a]['Peaks_indexes']]
            peak_heights_exc = data_ex.loc[a]['Peaks_heights']
            peak_times_inh = time_inh[data_in.loc[b]['Peaks_indexes']]
            peak_heights_inh = data_in.loc[b]['Peaks_heights']  
            
            plt.plot(time_exc, signal_exc, color='navy', label='Excitatory')
            plt.scatter(peak_times_exc,  peak_heights_exc, color='r')
            plt.plot(time_inh, signal_inh, color='gold', label='Inhibitory')
            plt.scatter(peak_times_inh,  peak_heights_inh, color='r')
            plt.gca().set(xlabel='Time w.r.t. maximum excitatory peak', ylabel='Population rates [kHz]', title=f'Events {str(a)} {str(b)}')
            plt.legend(loc=1)
            plt.show()
            pdf.savefig(fig)
            plt.close('all')
            
            
            
            
            
            
                       
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
    