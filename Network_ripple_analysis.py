#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 11:43:41 2022

@author: nunez
"""

'''
This file is to restore the long (6 seconds) network and do analysis on ripple events for it.
Here we should expect definition of calculation functions and plots.

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
from functions_from_Natalie import f_oscillation_analysis_transient
# ----------------------------names and folders-------------------------

path_to_save_figures = '/home/ana_nunez/Documents/BCCN_berlin/Master_thesis/Plots/'
path_networks = '/home/ana_nunez/Documents/BCCN_berlin/Master_thesis/'

#name_figures = 'network_multiply_rates_by_pop_size_cg_changing'

#name_network = 'long_baseline_no_dendritic_8'
#name_network = 'long_random_network_3'
name_network = 'long_10000_network_3'
#name_network = 'long_allneurons_event_network_3'  # For the event and all neurons the prerun is 1600 ms

# In order to restore the long network it is necessary to first create an object.
# Therefore, here I create a network of only 20 ms and on it, I restore the long one.
dur_simulation=10
network, monitors = Network_model_2(seed_num=1001, sim_dur=dur_simulation*ms, pre_run_dur=0*ms, total_neurons=1000, scale_factor=1, 
                                    dendritic_interactions=True, neurons_exc = arange(2), neurons_inh=arange(1))
#network.store(name='rand_net', filename = path_networks + name_network)

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


def find_events(n_group, pop_rate_monitor, threshold_in_sd, name_net=name_network, baseline_start=200, baseline_end=300, plot_peaks=False):
    '''
    Function to find the ripple events with in the network
    n_group: Neuron group
    pop_rate_monitor: Population rate monitor from either excitatory or inhibitory population (neuron group)
    threshold_in_sd: number of standard deviations to multiply for the baseline, in order to establish the threshold
        threshold = baseline * threshold_in_sd * sd
    baseline_start and end: in miliseconds of the signal.
    returns
        time: time array for the rate monitor [unitless]
        rate_signal: the signal from the rate monitor [unitless]
        peaks: index of the maximum peak of event     
        
    
    '''
    rate_signal = pop_rate_monitor.smooth_rate(width=1*ms)*len(n_group) / kHz
#    thr = mean(rate_signal[:int(400/dt)]) + threshold_in_sd * std(rate_signal) 
    thr = mean(rate_signal[int(baseline_start/dt):int(baseline_end/dt)]) + threshold_in_sd * std(rate_signal) 
    peaks, prop = signal.find_peaks(rate_signal, height = thr, distance = 50/dt)
    time = pop_rate_monitor.t / ms

    if plot_peaks:
        pdf_file_name = f'Events_detected_G_E_tr_{threshold_in_sd}_{name_net}'
        with PdfPages( path_to_save_figures + pdf_file_name + '.pdf') as pdf:
            fig = figure()
            plot(time, rate_signal, c='k')
            scatter(time[0] + peaks*dt, rate_signal[peaks], c='r')
            axhline(y=thr, c='gray', linestyle='--', label='Peak threshold')
            axhline(y=mean(rate_signal[int(200/dt):]), c='gray', linestyle='dotted', label='Baseline')
            legend()
            gca().set(xlabel='Time [ms]', ylabel='Rates [kHz]', xlim=(min(time),max(time)))
            savefig(path_to_save_figures + pdf_file_name +'.png')
            pdf.savefig(fig)

    return time, rate_signal, peaks

def define_event(n_group, pop_rate_monitor, threshold_in_sd, baseline_start=0, baseline_end=400, plot_peaks_bool=False):
    '''
    Function to define the start, end and therefore duration of a ripple event
    n_group: Neuron group
    pop_rate_monitor: Population rate monitor from either excitatory or inhibitory population (neuron group)
    threshold_in_sd: number of standard deviations to multiply for the baseline, in order to establish the threshold
        threshold = baseline * threshold_in_sd * sd
    returns
        
    '''
#    n_group=G_E
#    pop_rate_monitor=R_E
#    threshold_in_sd=3
#    
    time_signal, pop_rate_signal, max_peak_indexes = find_events(n_group, pop_rate_monitor, threshold_in_sd, plot_peaks=False)
    baseline = mean(pop_rate_signal[int(baseline_start/dt):int(baseline_end/dt)]) # Change BASELINE to only the mean of first 400 ms.
    event_props = {}
    simulation_props = {}
    simulation_props['Total_time'] = time_signal
    simulation_props['Total_signal'] = pop_rate_signal
    simulation_props['Max_peak_index'] = max_peak_indexes
    simulation_props['Baseline'] = baseline
#    figure()
    for i, index in enumerate(max_peak_indexes):
        print(i, index)
        event_props[i+1] = {}
        
        start_event_approx = int(index - 30/dt)
        end_event_appox = int(index + 30/dt)
        event_ranged = pop_rate_signal[start_event_approx:end_event_appox]
        time_ranged = time_signal[start_event_approx:end_event_appox]
        peaks, prop = signal.find_peaks(event_ranged, height = baseline, distance = 3/dt)
        
        index_peak_max = np.where(time_ranged[peaks] == time_ranged[index-start_event_approx])[0][0]
        find_first_peak_index = where((diff(time_ranged[peaks[:index_peak_max]]) < 6).astype(int) == 0)[0]
        find_last_peak_index = where((diff(time_ranged[peaks[index_peak_max:]]) < 6).astype(int) == 0)[0]
        
        if len(find_first_peak_index)==0: first_peak_index = where(peaks*dt > 3)[0][0]
        else: first_peak_index = find_first_peak_index[-1] + 1
        if len(find_last_peak_index)==0: last_peak_index = len(peaks) - 1
        else: last_peak_index = find_last_peak_index[0] + index_peak_max
        
#        first_peak_index = where((diff(time_ranged[peaks[:index_peak_max]]) < 6).astype(int) == 0)[0][-1] + 1
#        last_peak_index = where((diff(time_ranged[peaks[index_peak_max:]]) < 6).astype(int) == 0)[0][0] + index_peak_max 
        indexes_below_baseline_1 = where(event_ranged[:peaks[first_peak_index]] < baseline)[0]
        indexes_below_baseline_2 = where(event_ranged[peaks[index_peak_max]:] < baseline)[0] 
        
        if len(indexes_below_baseline_1)==0: start_index = 0
        else: start_index = indexes_below_baseline_1[-1] + 1
        if len(indexes_below_baseline_2)==0: end_index = len(event_ranged) - 1
        else: 
            end_index = where((indexes_below_baseline_2 + peaks[index_peak_max]) > peaks[last_peak_index])[0][0]+ peaks[last_peak_index]
#            end_index = indexes_below_baseline_2[-1] + peaks[index_peak_max] - 1
#        start_index = where(event_ranged[:peaks[first_peak_index]] < baseline)[0][-1] + 1
#        end_index = where(event_ranged[peaks[last_peak_index]:] < baseline)[0][0] + peaks[last_peak_index] - 1

        time_event = (np.arange(0, len(time_ranged[start_index:end_index])) -  (peaks[index_peak_max]-start_index )) * dt
        event = event_ranged[start_index:end_index]
#        peaks = peaks[first_peak_index:last_peak_index+1] - start_index
        event_props[i+1]['Time_array'] = time_event
        event_props[i+1]['Signal_array'] = event
        event_props[i+1]['Duration']= len(time_event)*dt
        event_props[i+1]['Index_start']= start_event_approx + start_index
        event_props[i+1]['Index_end']= start_event_approx + end_index
                
#    return event_props, simulation_props
#        figure()
#        plot(time_event, event, 'k')
#        axhline(y=baseline, c='gray', linestyle='dotted', label='Baseline')
#        scatter(time_event[peaks[first_peak_index:last_peak_index+1]-start_index], event[peaks[first_peak_index:last_peak_index+1]-start_index], c='orange')                
        if plot_peaks_bool:
            pdf_file_name = f'Net_{name_network}_Event_{i+1}_G_E_tr_{threshold_in_sd}'
            with PdfPages( path_to_save_figures + pdf_file_name + '.pdf') as pdf:
                fig = figure(figsize=(10/cm, 10/cm))
    
                plot(time_ranged, event_ranged, 'k')
                axhline(y=baseline, c='gray', linestyle='dotted', label='Baseline')
                scatter(time_ranged[peaks], event_ranged[peaks], c='orange')        
                scatter(time_ranged[index-start_event_approx], event_ranged[index-start_event_approx], c='r')
                scatter(time_ranged[start_index], event_ranged[start_index], c='b', label='Start_end')
                scatter(time_ranged[end_index], event_ranged[end_index], c='b')
                legend()
                gca().set(xlabel='Time [ms]', ylabel='Rates [kHz]', title=f'Network_{name_network[-1]}_Event_{i+1}')
    
                savefig(path_to_save_figures + pdf_file_name +'.png')
                pdf.savefig(fig)
            
    return event_props, simulation_props
        
def prop_events(n_group, pop_rate_monitor, threshold_in_sd, plot_peaks_bool=False):
    '''
    Function to determine the properties of each event such as number of peaks, 
    instantaneous frequency, mean frequency, spikes per cycle, etc.
    
    '''
       
    events_dict, simulation_dict = define_event(n_group, pop_rate_monitor, threshold_in_sd, plot_peaks_bool)
    
    for i, event in enumerate(events_dict.keys()):
    
        time = events_dict[event]['Time_array']
        signal_event = events_dict[event]['Signal_array']
        peaks, props = signal.find_peaks(signal_event, height = simulation_dict['Baseline'], distance = 3/dt, prominence=1, width=[1000,3000])
#        index_out = where(props['prominences'] < 1)[0]
        left_bases = props['left_bases']
        right_bases = props['right_bases']
        failed_left_calc = where(diff(left_bases)==0)[0]
        left_bases[failed_left_calc+1] = right_bases[failed_left_calc]
        failed_right_calc = where(diff(right_bases)==0)[0]
        right_bases[failed_right_calc] = left_bases[failed_right_calc+1]
#        if len(index_out) > 0: 
#            peaks = delete(peaks, index_out)
#            left_bases = delete(left_bases, index_out)
#            rigth_bases = delete(rigth_bases, index_out)
        
        freq = 1 / (diff(peaks)*dt/1000)   # Divided by 1000 because of ms
        indexes_freq = (diff(peaks)/2).astype(int) + peaks[:-1]
        min_dist_peak_tobase_xaxis = asarray(asmatrix(vstack((abs(peaks-left_bases), abs(peaks-right_bases)))).min(0))[0]
        start_cycles = peaks - min_dist_peak_tobase_xaxis
        end_cycles = peaks + min_dist_peak_tobase_xaxis
        spikes_percycle = np.zeros(len(peaks))
#        res_width = signal.peak_widths(signal_event, peaks, rel_height=1)
#        res_width_half = signal.peak_widths(signal_event, peaks, rel_height=0.5)
        events_dict[event]['Oscillation_events'] = {}
        for j, peak_cycle in enumerate(peaks):
            events_dict[event]['Oscillation_events']['Start_end'] = vstack((start_cycles, end_cycles))
            spikes_percycle[j] = trapz(y=signal_event[start_cycles[j]:end_cycles[j]], x=time[start_cycles[j]:end_cycles[j]], dx=dt)
            
        
        events_dict[event]['Num_peaks'] = len(peaks)
        events_dict[event]['Peaks_indexes'] = peaks
        events_dict[event]['Instant_frequency'] = freq
        events_dict[event]['Indices_frequency'] = indexes_freq
        events_dict[event]['Mean_frequency'] = mean(freq)
        events_dict[event]['Spikes_per_cycle'] = spikes_percycle
        
        
        
#        figure()
#        plot(time, signal_event, label=f'{event}')
##        plot(diff(signal_event), label=f'{event}', c='k')
#        scatter(time[peaks], signal_event[peaks], c='r')
#        scatter(time[start_cycles], signal_event[start_cycles], c='k')
#        scatter(time[end_cycles], signal_event[end_cycles], c='gray')
        
    return events_dict, simulation_dict


def plot_all_simulation(n_group, pop_rate_monitor, pop_spikes_monitor, pop_event_monitor, threshold_in_sd):
    '''
    Plot simulation over time with peak detection
    '''
    events_dict, simulation_dict = prop_events(n_group, pop_rate_monitor, threshold_in_sd, plot_peaks_bool=False)
    
    time = simulation_dict['Total_time']
    rate_signal = simulation_dict['Total_signal']
    index_peaks = simulation_dict['Max_peak_index']
    baseline = simulation_dict['Baseline']
    
    pdf_file_name = f'All_simulation_G_E_tr_{threshold_in_sd}_{name_network}'
    with PdfPages( path_to_save_figures + pdf_file_name + '.pdf') as pdf:
        fig = figure(figsize=(10/cm, 10/cm))
        plot(time, rate_signal, c='k')
        scatter(index_peaks*dt, rate_signal[index_peaks], c='r')
    #    axhline(y=thr, c='gray', linestyle='--', label='Peak threshold')
        axhline(y=baseline, c='gray', linestyle='dotted', label='Baseline')
        legend()
        gca().set(xlabel='Time [ms]', ylabel='Rates [kHz]', xlim=(0,max(time)))
        
        savefig(path_to_save_figures + pdf_file_name +'.png')
        pdf.savefig(fig)

def plot_all_events_prop(n_group, pop_rate_monitor, pop_spikes_monitor, pop_event_monitor, threshold_in_sd):
    '''
    Plot summary for one simulation.
    Plots of instantaneous freq, num_of peaks and mean freq.
    '''
    events_dict, simulation_dict = prop_events(n_group, pop_rate_monitor, threshold_in_sd, plot_peaks_bool=False)
    num_events = len(events_dict.keys())
    num_peaks = np.zeros(num_events)
    durations = np.zeros(num_events)
    mean_freq = np.zeros(num_events)
    
    colors = cmr.take_cmap_colors('viridis', num_events, return_fmt='hex') 
    pdf_file_name = f'All_events_prop_G_E_tr_{threshold_in_sd}_{name_network}'
    with PdfPages( path_to_save_figures + pdf_file_name + '.pdf') as pdf:
        fig = plt.figure(figsize=(21/cm,10/cm))
        gs0 = gridspec.GridSpec(1, 3, figure=fig, wspace=1/cm, hspace=0/cm)
        ax1 = fig.add_subplot(gs0[0])
        ax2 = fig.add_subplot(gs0[1])
        ax3 = fig.add_subplot(gs0[2])
        for i, event in enumerate(events_dict.keys()):
            num_peaks[i] = events_dict[event]['Num_peaks']
            durations[i] = events_dict[event]['Duration']
            mean_freq[i] = events_dict[event]['Mean_frequency']
            time_array = events_dict[event]['Time_array']
            freq_index = events_dict[event]['Indices_frequency']
            instant_freq = events_dict[event]['Instant_frequency']
            ax3.plot(time_array[freq_index], instant_freq, color=colors[i], marker='o')
            ax3.axhline(mean_freq[i], min(time_array), max(time_array), c=colors[i], linestyle='--')
        
        ax1.bar(arange(num_events)+1, num_peaks, color=colors)
        ax1.set_xticks(arange(num_events)+1)    
        ax1.set(xlabel='Event', ylabel='Number of peaks', title='Peaks per event')    
        ax2.bar(arange(num_events)+1, durations, color=colors)
        ax2.set_xticks(arange(num_events)+1)         
        ax2.set(xlabel='Event', ylabel='Duration [ms]', title='Duration of event')
        
        ax3.set(xlabel='Time wrt. highest peak[ms]', ylabel='Frequency[Hz]', title='Instanteneous frequency')
        
        fig.suptitle(f'Events of network {name_network[-1]}')
        
        savefig(path_to_save_figures + pdf_file_name +'.png')
        pdf.savefig(fig)

        
def get_index_neurons_from_window_of_time(monitor, time_1, time_2):
    '''
    Spike, state, or rate monitor
    time_1 and time_2 are values of ms but it is a unitless array. They define the window of time
    '''
    mask = where(np.logical_and(monitor.t/ms > time_1, monitor.t/ms < time_2))[0]
    index_neurons = monitor.i[mask]
    times_neurons = monitor.t[mask]/second
    dict_index_times = {}
    isi_values = []
    for neuron in unique(index_neurons):
        mask = where(index_neurons==neuron)
        dict_index_times[neuron] = times_neurons[mask]
        if len(times_neurons[mask])>1:
            isi_values = append(isi_values, diff(times_neurons[mask]))  
    return dict_index_times, isi_values
    
    
    

def ripple_prop(n_group, pop_rate_monitor, pop_spikes_monitor, pop_event_monitor, threshold_in_sd, plot_peaks_bool=False):
    
    '''
    With the dictionary of all events (few ripples per event). 
    In this function we check what about the spikes in each ripple oscillation
    '''
#    pop_spikes_monitor = M_E
#    pop_event_monitor = M_DS    
    pdf_file_name = f'Spiking_properties_G_E_th_{threshold_in_sd}_{name_network}'
    with PdfPages(path_to_save_figures + pdf_file_name + '.pdf') as pdf:
        events_dict, simulation_dict = prop_events(n_group, pop_rate_monitor, threshold_in_sd, plot_peaks_bool)
        total_events = len(events_dict.keys())
        fig = figure(figsize=(18/cm,23.7/cm))
        gs0 = gridspec.GridSpec(len(events_dict.keys()), 2, figure=fig, wspace=0.5/cm, hspace=0.2/cm)

        for i, event in enumerate(events_dict.keys()):
            start_time = events_dict[event]['Index_start']*dt
            end_time = events_dict[event]['Index_end']*dt
            neurons_spike, isi_spikes = get_index_neurons_from_window_of_time(pop_spikes_monitor, start_time, end_time)
            neurons_DS, isi_DS = get_index_neurons_from_window_of_time(pop_event_monitor, start_time, end_time)
    #        neurons_no_event_spike, isi_noSWR_spikes = get_index_neurons_from_window_of_time(pop_spikes_monitor, start_time-100, end_time-100)
    #        neurons_no_event_DS, isi_noSWR_DS_spikes = get_index_neurons_from_window_of_time(pop_event_monitor, start_time-100, end_time-100)
            list_neurons = []
            spikes_per_neuron = []
            for neuron in neurons_spike.keys():
                list_neurons.append(neuron)
                spikes_per_neuron.append(len(neurons_spike[neuron]))
            gssub = gs0[i,0].subgridspec(1, 2)
            plt.suptitle("GridSpec Inside GridSpec")
            ax_hist = fig.add_subplot(gssub[0])
            ax_isi = fig.add_subplot(gssub[1])
            ax_hist.bar(list_neurons, spikes_per_neuron, color='k')
            ax_isi.hist(isi_spikes*1000, color='k', range = (0,50))
            ax_hist.set(xlim=(0,900))
            ax_hist.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax_isi.set(xlim=(0,50))
            ax_isi.yaxis.tick_right()
            ax_isi.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax_isi.text(x=25,y=ax_isi.get_ylim()[1]*0.8, s=f'CV= {np.round(np.var(isi_spikes*1000) / np.mean(isi_spikes*1000),2)}')
            if i<total_events-1: 
                ax_hist.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                ax_isi.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            else:
                ax_hist.set(xlabel='Neuron index')
                ax_isi.set(xlabel='ISI at event [ms]')
            if i==0:
                ax_hist.set(title='Spikes/neuron')
                ax_isi.set(title='ISI distribution')            
                
            list_neurons_DS = []
            spikes_per_neuron_DS = []
            for neuron in neurons_DS.keys():
                list_neurons_DS.append(neuron)
                spikes_per_neuron_DS.append(len(neurons_DS[neuron]))
            gssub = gs0[i,1].subgridspec(1, 2)
            ax_hist = fig.add_subplot(gssub[0])
            ax_isi = fig.add_subplot(gssub[1])
            ax_hist.bar(list_neurons_DS, spikes_per_neuron_DS, color='k')
            ax_isi.hist(isi_DS*1000, color='k', range = (0,50))        
            ax_hist.set(xlim=(0,900))
            ax_hist.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax_isi.set(xlim=(0,50))
            ax_isi.yaxis.tick_right()
            ax_isi.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax_isi.text(x=25,y=ax_isi.get_ylim()[1]*0.8, s=f'CV= {np.round(np.var(isi_DS*1000) / np.mean(isi_DS*1000),2)}')
            if i<total_events-1: 
                ax_hist.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                ax_isi.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            else:
                ax_hist.set(xlabel='Neuron index')
                ax_isi.set(xlabel='ISI at event [ms]')
            if i==0:
                ax_hist.set(title='Spikes/neuron')
                ax_isi.set(title='ISI distribution')  
                
        plt.savefig(path_to_save_figures + pdf_file_name +'.png')
        pdf.savefig(fig)

def plot_stored_networks_full_simulation():
    '''
    Function to plot all simulation for some networks to observe if there is a baseline
    '''
    base_name = 'long_random_network'
    add_name = ['', '_2', '_3', '_4', '_5']
    
    for complement in add_name:
        network_name_complement = base_name+complement
        print(network_name_complement)
            
    
        # In order to restore the long network it is necessary to first create an object.
        # Therefore, here I create a network of only 20 ms and on it, I restore the long one.
        dur_simulation=10
        network, monitors = Network_model_2(seed_num=111, sim_dur=dur_simulation*ms, pre_run_dur=0*ms, total_neurons=1000, scale_factor=1)
        #network.store(name='rand_net', filename = path_networks + name_network)
        
        network.restore(name='rand_net', filename = path_networks + network_name_complement)
        
        # Get monitors from the network
        M_E = network.sorted_objects[-19]   
        M_I = network.sorted_objects[-18]
        M_DS = network.sorted_objects[-16]
        R_E = network.sorted_objects[-1]
        R_I = network.sorted_objects[-2]  
        State_G_ex = network.sorted_objects[2]
        State_G_in = network.sorted_objects[3]
        G_E = network.sorted_objects[0]
        G_I = network.sorted_objects[1]
        
        find_events(G_E, R_E, 3, name_net=network_name_complement, plot_peaks=True)


def plot_scatter_spikes(n_group, pop_rate_monitor, pop_spikes_monitor, pop_event_monitor, threshold_in_sd, plot_peaks_bool=False):
    '''
    Function to plot number of dendritic spikes vs number of spikes per NEURON in the network (first only excitatory neurons)
    '''
    events_dict, simulation_dict = prop_events(n_group, pop_rate_monitor, threshold_in_sd, plot_peaks_bool)
    total_events = len(events_dict.keys())
    colors = cmr.take_cmap_colors('viridis', total_events, return_fmt='hex') 

    pdf_file_name = f'Spiking_relation_G_E_th_{threshold_in_sd}_Net_{name_network[-1]}'
    with PdfPages(path_to_save_figures + pdf_file_name + '.pdf') as pdf:
        fig, ax = subplots(total_events, 1, figsize=(18/cm,23.7/cm), sharex=True, sharey=True)
        
        for i, event in enumerate(events_dict.keys()):
            start_time = events_dict[event]['Index_start']*dt
            end_time = events_dict[event]['Index_end']*dt
            neurons_spike, isi_spikes = get_index_neurons_from_window_of_time(pop_spikes_monitor, start_time, end_time)
            neurons_DS, isi_DS = get_index_neurons_from_window_of_time(pop_event_monitor, start_time, end_time)
    #        neurons_no_event_spike, isi_noSWR_spikes = get_index_neurons_from_window_of_time(pop_spikes_monitor, start_time-100, end_time-100)
    #        neurons_no_event_DS, isi_noSWR_DS_spikes = get_index_neurons_from_window_of_time(pop_event_monitor, start_time-100, end_time-100)
            array_all_spikes= np.zeros((len(n_group.i), 2))
            for index in n_group.i:
                if index in neurons_DS: array_all_spikes[index, 0] = len(neurons_DS[index])
                else: array_all_spikes[index, 0] = 0
                if index in neurons_spike: array_all_spikes[index, 1] = len(neurons_spike[index])
                else: array_all_spikes[index, 1] = 0
                    
            points = list(set(zip(array_all_spikes[:,0],array_all_spikes[:,1])))
            count = [len([x for x,y in zip(array_all_spikes[:,0],array_all_spikes[:,1]) if x==p[0] and y==p[1]]) for p in points]
            plot_x=[i[0] for i in points]
            plot_y=[i[1] for i in points]
            count=np.array(count)
            pl = ax[i].scatter(plot_x,plot_y,c=count,s=100*count**0.5,cmap='viridis', vmin=0, vmax=250)
            ax[i].set(ylabel='Number of spikes')
            ax[i].set_ylim(bottom=-1)
            plt.colorbar(pl, ax=ax[i])
        ax[i].set(xlabel='Number of dendritic spikes')
        fig.suptitle('Spikes per neuron and event')
#            ax[i].scatter(array_all_spikes[:,0], array_all_spikes[:,1], color=colors[i])
        plt.savefig(path_to_save_figures + pdf_file_name +'.png')
        pdf.savefig(fig)        
        

def currents_one_event(n_group, pop_rate_monitor, pop_spikes_monitor, pop_state_monitor, threshold_in_sd = 3):
    '''
    Function to observe how the currents in the neurons are within one event. Speciallz before and after
    '''
    if name_network != 'long_allneurons_event_network_3': return
    else:    
        events, sim = prop_events(n_group, pop_rate_monitor, threshold_in_sd, plot_peaks_bool=False)
        peaks_indexes = (events[1]['Peaks_indexes'] + events[1]['Index_start']) * dt 
        pdf_file_name = f'spikes_currents_G_{n_group.name[-2].upper()}_th_{threshold_in_sd}_{name_network}'
        with PdfPages(path_to_save_figures + pdf_file_name + '.pdf') as pdf:
            fig, ax = subplots(3, 1, figsize=(18/cm,12/cm), sharex=True)
            
            time = pop_rate_monitor.t/ms
            rate_signal = pop_rate_monitor.smooth_rate(width=1*ms)*len(n_group) / kHz
            exc_current = mean(pop_state_monitor.I_A/nA, axis=0)
            inh_current = mean(pop_state_monitor.I_G/nA, axis=0)
            
            ax[0].scatter(pop_spikes_monitor.t/ms, pop_spikes_monitor.i, marker='.', color='k')
            ax[0].set(ylabel='Neuron index', xlim=(min(time), max(time)))
            ax[0].vlines(x=peaks_indexes+time[0], ymin = ax[0].get_ylim()[0], ymax = ax[0].get_ylim()[1], colors='silver', zorder=0, linewidth=0.5)
            
            ax[1].plot(time, rate_signal, color='k')
            ax[1].set(ylabel='Rate [kHz]')
            ax[1].vlines(peaks_indexes+time[0], ymin = ax[1].get_ylim()[0], ymax = ax[1].get_ylim()[1], colors='silver', zorder=0, linewidth=0.5)
            
            ax[2].plot(time, exc_current, color='green', label='Excitatory')
            ax[2].plot(time, abs(inh_current), color='indigo', label='abs(Inhibitory)')    
            ax[2].plot(time, exc_current - abs(inh_current), color='skyblue', label='Difference')  
            ax[2].vlines(peaks_indexes+time[0], ymin = ax[2].get_ylim()[0], ymax = ax[2].get_ylim()[1], colors='silver', zorder=0, linewidth=0.5)
            ax[2].set(ylabel='Current [nA]', xlabel='Time [ms]')
            ax[2].legend()
            
            plt.savefig(path_to_save_figures + pdf_file_name +'.png')
            pdf.savefig(fig)

    
def wavelet_analysis_plot(n_group, pop_rate_monitor, threshold_in_sd, ):
    '''
    Function to evaluate the ripple event based on the frequency domain and power spectrum
    '''
    events, sim = prop_events(n_group, pop_rate_monitor, threshold_in_sd, plot_peaks_bool=False)
    total_signal = sim['Total_signal']
    peak_indexes = sim['Max_peak_index']
    
    plt.figure()
    for event in events.keys():
        start = events[event]['Index_start']*dt
        end = events[event]['Index_end']*dt
        wspec, wspec_extent, instfreq, instpower, freq_onset_inst, instcoherence, Pthr, ifreq_discr_t, ifreq_discr\
        = f_oscillation_analysis_transient(total_signal, dt=dt, baseline_window=[0, 400], target_window = [start, end], \
                                           expected_freq = 200, fmin=100, plot=False)
        plt.plot()



#rate_signal, peak_per_event = find_events(n_group=G_E, pop_rate_monitor=R_E, threshold_in_sd=4, plot_peaks=True)
#
#
#
#plot(R_E.t / ms, R_E.smooth_rate(width=1*ms)*len(G_E) / kHz, color='k')
#


