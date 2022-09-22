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

path_to_save_figures = '/home/ana_nunez/Documents/BCCN_berlin/Master_thesis/Plots/'
path_networks = '/home/ana_nunez/Documents/BCCN_berlin/Master_thesis/'

#path_to_save_figures = '/home/nunez/New_repo/Plots_populations_comparison/'
#path_networks = '/home/nunez/New_repo/stored_networks/'


#name_figures = 'network_multiply_rates_by_pop_size_cg_changing'

#name_network = 'long_baseline_no_dendritic_8'
#name_network = 'long_random_network_8'
name_network = 'long_6000_network_3_tauax_change' #Neurons false
#name_network = 'long_allneurons_event_network_3'  # For the event and all neurons the prerun is 1600 ms. Neurons true
#
## In order to restore the long network it is necessary to first create an object.
## Therefore, here I create a network of only 20 ms and on it, I restore the long one.
dur_simulation=10
network, monitors = Network_model_2(seed_num=1001, sim_dur=dur_simulation*ms, pre_run_dur=0*ms, total_neurons=1000, 
                                    scale_factor=1, dendritic_interactions=True, neurons_exc = False, neurons_inh = False, 
                                    tau_ax_method = 'distance_dist')
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
##
dt = 0.001
cm = 2.54


events_dict_ex, simulation_dict_ex = prop_events(net_name_aux = name_network, n_group=G_E, pop_rate_monitor=R_E, pop_spikes_monitor=M_E, pop_ds_spikes_monitor=M_DS, threshold_in_sd=3, plot_peaks_bool=False)
df_ex = pd.DataFrame.from_dict(events_dict_ex, orient='index')
df_ex.to_csv(path_to_save_figures+f'Properties_events_ex_{name_network}.csv')
df_ex.to_pickle(path_to_save_figures+f'Properties_events_ex_{name_network}.pkl')

#events_dict_in, simulation_dict_in = prop_events(n_group=G_I, pop_rate_monitor=R_I, pop_spikes_monitor=M_I, pop_ds_spikes_monitor=M_DS, threshold_in_sd=3, plot_peaks_bool=False)
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


def get_events_ex_in(name_net):
    
    data_ex = pd.read_pickle(path_to_save_figures+f'Properties_events_ex_{name_net}.pkl')
    data_in = pd.read_pickle(path_to_save_figures+f'Properties_events_in_{name_net}.pkl')
    headers = ['Name_network', 'Event_id', 'Event_ex', 'Event_in','Difference_max_peak_time', 'Time_exc', 'Time_inh', 'Signal_exc', 'Signal_inh', 
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
        else: events_ex_in.loc[index] = [name_net, evs+1,a,b, diff_peak_times,time_exc, time_inh, signal_exc, signal_inh, peak_times_exc,
                              peak_times_inh, peak_times_inh-peak_times_exc, zero_peak_index]
        index +=1

    events_ex_in.to_csv(path_to_save_figures+f'Properties_populations_events_in_{name_net}.csv')
    events_ex_in.to_pickle(path_to_save_figures+f'Properties_populations_events_in_{name_net}.pkl')
    return events_ex_in



def check_average_shapes_both_pop(name_net=name_network):
    '''
    Function to corrobotare figure S5 B from Memmesheimmer supplementary material
    '''
#    long_bad_events = [1,6,11,23,31,34, 4,10,12,13,14,21,30,38,39,43,46] #network 8
    long_bad_events = [2,4,8,11,13,14,15,16,18,27,31,32,33,34,35,41,48,49,52,53] #network 2
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
            
def check_dendritic_spikes_populations(name_net=name_network, spikes_monitor=M_E, denspikes_monitor=M_DS,spikes_inh_monitor = M_I, normalized=False):
    
    data_ex = pd.read_pickle(path_to_save_figures+f'Properties_events_ex_{name_network}.pkl')
    list_events = [1,3,6,7,9,10,17,22,24,26,28,29,30, 36,37,38,39,40, 43,45,46,47,50,55]
    delay_times_before_peak = []
    delay_times_after_peak = []
    pdf_file_name = f'Some_examples_spikes_dendritic_populations_{name_net}_normalized_{int(normalized)}'
    with PdfPages( path_to_save_figures + pdf_file_name + '.pdf') as pdf:
        for event_exc in list_events:
            print(event_exc)
            start_index = data_ex.loc[event_exc]['Index_start']
            end_index = data_ex.loc[event_exc]['Index_end']
            max_peak_time = data_ex.loc[event_exc]['Max_peak_time']
            kernel = stats.norm.pdf(np.arange(-3,3,dt), loc=0, scale=0.2)
            time_array = np.round(np.arange(start_index, end_index)*dt,3)
            
        
            all_time_spikes = spikes_monitor.t/ms
            mask_spikes = np.where((all_time_spikes<end_index*dt)&(all_time_spikes>start_index*dt))[0]
            time_spikes = all_time_spikes[mask_spikes]
            bin_list = np.round(np.arange(start_index, end_index+1)*dt,3)
            
            all_time_spikes_ds = denspikes_monitor.t/ms
            mask_spikes_ds = np.where((all_time_spikes_ds<end_index*dt)&(all_time_spikes_ds>start_index*dt))[0]
            time_spikes_ds = all_time_spikes_ds[mask_spikes_ds]

            all_time_spikes_in = spikes_inh_monitor.t/ms
            mask_spikes_in = np.where((all_time_spikes_in<end_index*dt)&(all_time_spikes_in>start_index*dt))[0]
            time_spikes_in = all_time_spikes_in[mask_spikes_in]
            
            hist, _ = np.histogram(time_spikes, bins=bin_list)
            smooth_spike_rate = np.convolve(hist, kernel, mode='same')
            hist_ds, _ = np.histogram(time_spikes_ds, bins=bin_list)
            smooth_spike_rate_ds = np.convolve(hist_ds, kernel, mode='same')
            hist_in, _ = np.histogram(time_spikes_in, bins=bin_list)
            smooth_spike_rate_in = np.convolve(hist_in, kernel, mode='same')
            
            fig = plt.figure(figsize=(21/cm, 12/cm))
            if normalized: 
                plt.plot(time_array, smooth_spike_rate/max(smooth_spike_rate), color='navy', label='Excitatory spikes')
                plt.plot(time_array, smooth_spike_rate_ds/max(smooth_spike_rate_ds), color='blue', label='Dendritic spikes')
                plt.plot(time_array, smooth_spike_rate_in/max(smooth_spike_rate_in), color='gold', label='Inhibitory spikes')
            else:
                plt.plot(time_array, smooth_spike_rate, color='navy', label='Excitatory spikes')
                plt.plot(time_array, smooth_spike_rate_ds, color='blue', label='Dendritic spikes')
                plt.plot(time_array, smooth_spike_rate_in, color='gold', label='Inhibitory spikes')

            plt.axvline(x=max_peak_time, color='gray', linestyle='--', alpha=0.5)
            plt.gca().set(xlabel='Time event [ms]', ylabel='Normalized rates', title=f'Event {event_exc}')
            plt.legend(loc=1)
            
            peaks_spikes, _ = signal.find_peaks(smooth_spike_rate, distance=3/dt, prominence=0.5)
            peaks_spikes_ds, _ = signal.find_peaks(smooth_spike_rate_ds, distance=3/dt, prominence=0.5)
            if peaks_spikes[0]<2000: 
                peaks_spikes = peaks_spikes[1:]
                peaks_spikes_ds = peaks_spikes_ds[1:]
            start_times_ds = np.zeros(len(peaks_spikes))
            peak_times = time_array[peaks_spikes]
            peak_times_ds = time_array[peaks_spikes_ds]
            for i, peak_index in enumerate(peaks_spikes):
                start_times_ds[i] = time_array[np.where(hist_ds[peak_index-2000:peak_index]>0)[0][0] + peak_index-2000]
        
            x_positions = (np.diff(peak_times)/2)+peak_times[:-1]
            mask_ba = np.where(x_positions<max_peak_time)[0]
            mask_after = np.where(x_positions>max_peak_time)[0]
            distance_between_peak_and_ds_start = start_times_ds[1:] - peak_times[:-1]
            distance_between_peak_and_ds_peak = peak_times_ds[1:] - peak_times[:-1]
            for j, d in enumerate(distance_between_peak_and_ds_peak):
                plt.text(x=x_positions[j], y=0.8, s=f'{np.round(d,3)}')
            delay_times_before_peak.append(distance_between_peak_and_ds_peak[mask_ba])
            delay_times_after_peak.append(distance_between_peak_and_ds_peak[mask_after])
            pdf.savefig(fig)
            plt.close('all')
            
    return delay_times_before_peak, delay_times_after_peak        
    

def plot_delays_peaks_desdritic_spikes(name_net=name_network):

    delays_before, delays_after = check_dendritic_spikes_populations(normalized=True)
    
    delays_before, delays_after = np.hstack(delays_before), np.hstack(delays_after)
    bin_list = np.arange(4,6, 0.02)
    pdf_file_name = f'Delay_distribution_between_spiking_and_dendritic_spiking_{name_net}'
    with PdfPages( path_to_save_figures + pdf_file_name + '.pdf') as pdf:
        fig = plt.figure(figsize=(21/cm, 12/cm))
        plt.hist(delays_before, bins=bin_list, histtype='step', color='indigo', label='Before max peak')
        plt.hist(delays_after, bins=bin_list, histtype='step', color='green', label='After max peak')
        plt.gca().set(xlabel='Time between population spikes and next dendritic population spikes [ms]')
        plt.legend(loc=1)
    
        plt.axvline(x = np.mean(delays_before), color='indigo', alpha=0.5, linestyle='--')
        plt.axvline(x = np.mean(delays_after), color='green', alpha=0.5, linestyle='--')
        plt.axvline(x = np.median(delays_before), color='indigo', alpha=0.5, linestyle='dotted')
        plt.axvline(x = np.median(delays_after), color='green', alpha=0.5, linestyle='dotted')         
        p_val = stats.ttest_ind(delays_before, delays_after, equal_var=False).pvalue
        print(f'p value is {p_val}')
                
        savefig(path_to_save_figures + pdf_file_name +'.png')
        pdf.savefig(fig)            
            
            
def create_dataframe_all_long_networks(net_name = 'long_6000_network_tauax_change_', networks_names =['1', '2', '3']):
    '''
    Function to define a huge dataframe with several networks
    
    '''
#    net_name = 'long_50000_network_' 
#    networks_names = ['2','3','6','7','8']
    data = []
    df_all_ex = pd.DataFrame(data)
    df_all_in = pd.DataFrame(data)
    df_all_ex_in = pd.DataFrame(data)
    # For loop for all networks
    for j, num_net in enumerate(networks_names):
        name_network = net_name+num_net
        print(name_network)
        dur_simulation=10
        network, monitors = Network_model_2(seed_num=1001, sim_dur=dur_simulation*ms, pre_run_dur=0*ms, total_neurons=1000, 
                                            scale_factor=1, dendritic_interactions=True, neurons_exc = False, neurons_inh = False)
        network.restore(name='rand_net', filename = path_networks + name_network)
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


        events_dict_ex, simulation_dict_ex = prop_events(net_name_aux=name_network, n_group=G_E, pop_rate_monitor=R_E, pop_spikes_monitor=M_E, 
                                                         pop_ds_spikes_monitor=M_DS, threshold_in_sd=3, plot_peaks_bool=False)
        df_ex = pd.DataFrame.from_dict(events_dict_ex, orient='index')
        df_ex.to_csv(path_to_save_figures+f'Properties_events_ex_{name_network}.csv')
        df_ex.to_pickle(path_to_save_figures+f'Properties_events_ex_{name_network}.pkl')
        
        events_dict_in, simulation_dict_in = prop_events(net_name_aux=name_network, n_group=G_I, pop_rate_monitor=R_I, pop_spikes_monitor=M_I, 
                                                         pop_ds_spikes_monitor=M_DS, threshold_in_sd=3, plot_peaks_bool=False)
        df_in = pd.DataFrame.from_dict(events_dict_in, orient='index')
        df_in.to_csv(path_to_save_figures+f'Properties_events_in_{name_network}.csv')
        df_in.to_pickle(path_to_save_figures+f'Properties_events_in_{name_network}.pkl')
            
        events_ex_in = get_events_ex_in(name_net=name_network)    
            
        df_all_ex = df_all_ex.append(df_ex, ignore_index=True)  
        df_all_in = df_all_in.append(df_in, ignore_index=True)  
        df_all_ex_in = df_all_ex_in.append(events_ex_in, ignore_index=True) 
        
    df_all_ex.to_csv(path_to_save_figures+f'Properties_exitatory_ALL_events_{net_name}.csv')
    df_all_ex.to_pickle(path_to_save_figures+f'Properties_exitatory_ALL_events_{net_name}.pkl')
    df_all_in.to_csv(path_to_save_figures+f'Properties_inhibitory_ALL_events_{net_name}.csv')
    df_all_in.to_pickle(path_to_save_figures+f'Properties_inhibitory_ALL_events_{net_name}.pkl')        
    df_all_ex_in.to_csv(path_to_save_figures+f'Properties_ALL_events_{net_name}.csv')
    df_all_ex_in.to_pickle(path_to_save_figures+f'Properties_ALL_events_{net_name}.pkl')               
            
def get_spikes_and_dendritic_spikes(dict_spikes, dict_dendritic_spikes, num_neurons=900):
    '''
    Function to get the previous dendritic spike to a spike
    '''
    dict_prev_ds_spike = {}
    for n in np.arange(num_neurons):
        if (n in dict_spikes.keys()) and (n in dict_dendritic_spikes.keys()):
            
            i_to_delete = []

#            print(n)
            s, ds = dict_spikes[n]*1000, dict_dendritic_spikes[n]*1000
            dict_prev_ds_spike[n] = np.zeros((len(s), 2))
            for i, spike in enumerate(s):
                mask_sp = np.where(ds<spike)[0]
                if len(mask_sp)>0:
                    d_spike = ds[mask_sp[-1]]
                    dict_prev_ds_spike[n][i,:] = d_spike, spike
                else: i_to_delete.append(i)
            
            if len(i_to_delete)>0: dict_prev_ds_spike[n] = np.delete(dict_prev_ds_spike[n], i_to_delete, 0)
            if len(dict_prev_ds_spike[n])==0: dict_prev_ds_spike.pop(n)
        
        else: continue
    
    return np.vstack(dict_prev_ds_spike.values())    
    
def plot_spikes_vs_tauDSAP():    
#    data_ex = pd.read_pickle(path_to_save_figures+f'Properties_exitatory_ALL_events.pkl')    
    data_ex = pd.read_pickle(path_to_save_figures+f'Properties_exitatory_ALL_events_long_6000_network_tauax_change_.pkl')
    min_dur, max_dur = [13, 53]   # in ms
    division_array = np.arange(-max_dur+0.5, max_dur-0.5, 5)
    pdf_file_name = f'Scatter_someEvents_time_spike_tauDSAP_and_period_tauax_changed'
    with PdfPages(path_to_save_figures + pdf_file_name + '.pdf') as pdf:        
        for ind in data_ex.index:
#        for ind in np.arange(0,260,10):   # only plot max 25 cases
            if (data_ex.loc[ind]['Duration']<min_dur) or (data_ex.loc[ind]['Duration']>max_dur): continue
            else:
                print(ind)
                name_network = data_ex.loc[ind]['Name_network']
                array_ds_s = get_spikes_and_dendritic_spikes(data_ex.loc[ind]['Dict_spike_times_per_neuron'], data_ex.loc[ind]['Dict_dendritic_spike_times_per_neuron'])
                max_peak_time = data_ex.loc[ind]['Max_peak_time']
                instant_freq = data_ex.loc[ind]['Instant_frequency']
                instant_time = data_ex.loc[ind]['Time_array'][data_ex.loc[ind]['Indices_frequency']]
                
                array_ds_s = array_ds_s -max_peak_time
                tau_DSAP = np.diff(array_ds_s, axis=1)
#                array_ds_s = np.delete(array_ds_s, np.where(tau_DSAP>2)[0], 0)
                spikes = array_ds_s[:,1]
                tau_DSAP = np.diff(array_ds_s, axis=1)
                
                start_mask = np.where(division_array<np.min(array_ds_s))[0][-1]
                end_mask = np.where(division_array>np.max(array_ds_s))[0][0]
                division = division_array[start_mask:end_mask+1]
                means_spikes = np.zeros(len(division)-1)
                means_taus = np.zeros(len(division)-1)
                for i, start in enumerate(division[:-1]):
                    end = division[i+1]
                    mask = np.where((spikes>start)&(spikes<end))[0]
                    means_spikes[i] = np.mean(spikes[mask])
                    means_taus[i] = np.mean(tau_DSAP[mask])
                
                
                fig, ax = plt.subplots(2, 1, figsize=(21/cm, 16/cm), sharex=True)
                
                ax[0].scatter(spikes, tau_DSAP, marker='.', color='k')
                ax[0].plot(means_spikes, means_taus, marker='*', color='r')
                ax[0].set(xlabel='Spike times w.r.t. max peak [ms]', ylabel='t DS,AP [ms]')
                ax[0].grid()
                ax[1].plot(instant_time, 1000/instant_freq, marker='*', color='r')
                ax[1].set(xlabel='Time w.r.t. max peak [ms]', ylabel='Period [ms]')
                ax[1].grid()      
                plt.suptitle(f'Network {name_network[-1]}. Peak time {max_peak_time}')
                
                pdf.savefig(fig) 
                plt.close()    

def plot_all_discrete_frequencies(type_population, plot_type, net_n='Change_tauax'):
    '''
    Function to plot and analyse the instantaneous frequency
    type_population can be:
        excitatory 
        inhibitory 
    and that would determine which dataframe to load.
    plot_type could be:
        scatter_all
        plot_all
        plot_regression
        plot_mean_halves
    '''
    if type_population == 'excitatory': df = pd.read_pickle(path_to_save_figures+f'Properties_exitatory_ALL_events.pkl')   
    elif type_population == 'inhibitory': df = pd.read_pickle(path_to_save_figures+f'Properties_inhibitory_ALL_events.pkl')   
    else: return None
    if net_n=='Change_tauax': df = pd.read_pickle(path_to_save_figures+f'Properties_exitatory_ALL_events_long_6000_network_tauax_change_.pkl')
    min_dur, max_dur = [13, 53]
    selected_df = df[(df['Duration']<=max_dur) & (df['Duration']>=min_dur)]
     
    
    
    num_events = len(selected_df)
    colors = cmr.take_cmap_colors('cividis', num_events, return_fmt='hex') 
    times = np.array([])
    frequencies = np.array([])
    pdf_file_name = f'Discrete_frequencies_{type_population}_{plot_type}_{net_n}'
    with PdfPages(path_to_save_figures + pdf_file_name + '.pdf') as pdf:
        fig, ax = plt.subplots(1, 1, figsize=(21/cm, 10/cm))
        for i, event in enumerate(selected_df.index):
            print(event)
            frequencies_discrete = selected_df.loc[event]['Instant_frequency']
            time_values = selected_df.loc[event]['Time_array']
            time_discrete = time_values[selected_df.loc[event]['Indices_frequency']]
            f_half1 = np.mean(frequencies_discrete[np.where(time_discrete<0)[0]])
            f_half2 = np.mean(frequencies_discrete[np.where(time_discrete>0)[0]])
                
            times = np.append(times, time_discrete)
            frequencies = np.append(frequencies, frequencies_discrete)
            
            if plot_type == 'scatter_all': ax.scatter(time_discrete, frequencies_discrete, color=colors[i], marker='.')
            if plot_type == 'plot_all': ax.plot(time_discrete, frequencies_discrete, color=colors[i], marker='.')
            if plot_type == 'plot_mean_halves':
                ax.plot(time_discrete[np.where(time_discrete<0)[0]], np.ones(len(time_discrete[np.where(time_discrete<0)[0]]))*f_half1, color=colors[i])
                ax.plot(time_discrete[np.where(time_discrete>0)[0]], np.ones(len(time_discrete[np.where(time_discrete>0)[0]]))*f_half2, color=colors[i])
        if plot_type == 'plot_regression':     
            t = np.arange(min(times), max(times), 0.1)
            model = SVR(kernel='rbf')
            freq_predict = model.fit(times.reshape(-1, 1),frequencies).predict(t.reshape(-1, 1))
            regression = stats.linregress(times, frequencies)
            y = regression.slope*t +regression.intercept
            ax.plot(t,y, 'k')
            ax.text(10,230, f'Slope = {np.round(regression.slope,4)} [Hz/ms]')
            ax.plot(t, freq_predict, 'gray')
        
        rect = plt.Rectangle((min(times), 1/(5.2/1000)), max(times)-min(times), 1/(4.3/1000)-1/(5.2/1000), color='k', alpha=0.2, zorder=-1)
        ax.add_patch(rect)
        ax.grid(zorder= 0)
        ax.set(xlabel='Time w.r.t. highest peak [ms]', ylabel='Frequency [Hz]')
#        plt.savefig(path_to_save_figures + pdf_file_name +'.png')
        pdf.savefig(fig)      
    
    
    
def plot_correlation_freq_rateHeights(type_population, parameter, value_rates, short_event = True, name_file = 'events_long_6000_network_tauax_change_.pkl'):    
    '''
    Function to plot and correlate peak heights and period/frequency
    type_population can be:
        excitatory
        inhibitory
    parameter can be:
        frequency
        period
    value_rates can be:
        previous
        posterior
        difference : between previous and after peaks
    short_event <= 10 ripples. long event more than 10
    '''

    if type_population == 'excitatory': df = pd.read_pickle(path_to_save_figures+f'Properties_exitatory_ALL_'+name_file)   
    elif type_population == 'inhibitory': df = pd.read_pickle(path_to_save_figures+f'Properties_inhibitory_ALL_'+name_file)  
    if short_event: min_dur, max_dur = [13, 53]
    else: min_dur, max_dur = [53, 140]
    selected_df = df[(df['Duration']<=max_dur) & (df['Duration']>=min_dur)]
    num_events = len(selected_df)
    heights = np.array([])
    parameter_values = np.array([])
    times = np.array([])
    pdf_file_name = f'Scatter_corr_pop_{type_population}_{parameter}_rate_{value_rates}_shortEvent_{int(short_event)}_{name_file}'
    with PdfPages(path_to_save_figures + pdf_file_name + '.pdf') as pdf:
        fig, ax = plt.subplots(1, 1, figsize=(15/cm, 15/cm))
        for i, event in enumerate(selected_df.index):
            peak_heights = selected_df.loc[event]['Peaks_heights']
            instant_frequencies = selected_df.loc[event]['Instant_frequency']
            time_values = selected_df.loc[event]['Time_array']
            instant_times = time_values[selected_df.loc[event]['Indices_frequency']]
            if value_rates=='previous': heights_discrete = peak_heights[:-1]
            elif value_rates=='posterior': heights_discrete = peak_heights[1:]
            elif value_rates=='difference': heights_discrete = np.diff(peak_heights)
            else: return None

            if parameter=='frequency': parameter_values = np.append(parameter_values, instant_frequencies) 
            elif parameter =='period': parameter_values = np.append(parameter_values, 1000/instant_frequencies) 
            else: return None

            heights = np.append(heights, heights_discrete)
            times = np.append(times, instant_times)
            if len(np.where(instant_frequencies>250)[0])>0: print(event)
            
        mask_before = np.where(times<0)[0]
        mask_after = np.where(times>0)[0]
        ax.scatter(parameter_values[mask_before], heights[mask_before], color='indigo', label='Peaks before max', marker='.')  
        ax.scatter(parameter_values[mask_after], heights[mask_after], color='green', label='Peaks after max', marker='.') 
        ax.axvline(x=np.mean(parameter_values[mask_before]), color='indigo', linestyle='--')
        ax.axvline(x=np.mean(parameter_values[mask_after]), color='green', linestyle='--')
        regression = stats.linregress(parameter_values, heights)
        reg_before = stats.linregress(parameter_values[mask_before], heights[mask_before])
        reg_after = stats.linregress(parameter_values[mask_after], heights[mask_after])
        f = np.arange(min(parameter_values), max(parameter_values), 0.1)
        y = regression.slope*f +regression.intercept
        y_before = reg_before.slope*f +reg_before.intercept
        y_after = reg_after.slope*f +reg_after.intercept
        ax.plot(f,y, 'k')
#        ax.plot(f, y_before, 'indigo'), ax.plot(f, y_after, 'green')
        print(np.round(regression.rvalue,4), np.round(reg_before.rvalue,4), np.round(reg_after.rvalue,4))
        ax.legend()
        ax.grid(zorder= 0)
        if parameter=='frequency': ax.set(xlabel='Frequency of ripple event [Hz]')
        elif parameter=='period': ax.set(xlabel='Period of ripple event [ms]')
        if value_rates=='previous':ax.set(ylabel='Population rates before [kHz]')
        elif value_rates=='posterior':ax.set(ylabel='Population rates after [kHz]')
        elif value_rates=='difference':ax.set(ylabel='Population rates differences [kHz]')
        fig.text(0.5, 0.75, f'Correlation coefficient = {np.round(regression.rvalue,4)}') 
        plt.savefig(path_to_save_figures + pdf_file_name +'.png')
        pdf.savefig(fig)     
    
    
    

#    
import random    
array_distances = np.zeros(1000000)    
for i in np.arange(len(array_distances)):
    a = np.array([random.uniform(0, 350), random.uniform(0, 350)])
    b = np.array([random.uniform(0, 350), random.uniform(0, 350)])
    array_distances[i] = np.linalg.norm(a-b)
    
plt.figure()
plt.hist(array_distances/300, bins=100, density=True, histtype='step') 
(mu, sigma) = stats.norm.fit(array_distances/300)   
array_normal = clip(np.random.normal(loc=mu, scale=sigma, size=1000000), 0, Inf)
plt.hist(array_normal, bins=100, density=True, histtype='step') 
#     mu
#Out[103]: 182.50546181613075
#     
#     sigma
#Out[104]: 86.73284591314486
     
#distances 
#mu
#Out[108]: 0.6083515393871026     
#sigma
#Out[109]: 0.28910948637714945
#     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
    