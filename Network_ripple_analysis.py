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
from sklearn.svm import SVR


from Network_model_2_two_populations import Network_model_2
from functions_from_Natalie import f_oscillation_analysis_transient
# ----------------------------names and folders-------------------------

#path_to_save_figures = '/home/ana_nunez/Documents/BCCN_berlin/Master_thesis/Plots/'
#path_networks = '/home/ana_nunez/Documents/BCCN_berlin/Master_thesis/'

path_to_save_figures = '/home/nunez/New_repo/Plots/'
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


def find_events(n_group, pop_rate_monitor, threshold_in_sd, name_net=name_network, smoothing = 0.5, baseline_start=0, baseline_end=300, plot_peaks=False, dt=dt):
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
    rate_signal = pop_rate_monitor.smooth_rate(width=smoothing*ms)*len(n_group) / kHz
#    thr = mean(rate_signal[:int(400/dt)]) + threshold_in_sd * std(rate_signal) 
    thr = mean(rate_signal[int(baseline_start/dt):int(baseline_end/dt)]) + threshold_in_sd * std(rate_signal) 
    peaks, prop = signal.find_peaks(rate_signal, height = thr, distance = 100/dt)
    time = pop_rate_monitor.t / ms

    if plot_peaks:
        pdf_file_name = f'Events_detected_G_E_tr_{threshold_in_sd}_{name_net}_windowsmooth_{smoothing}'
        with PdfPages( path_to_save_figures + pdf_file_name + '.pdf') as pdf:
            fig = figure()
            plot(time, rate_signal, c='k')
            scatter(time[0] + peaks*dt, rate_signal[peaks], c='r')
            axhline(y=thr, c='gray', linestyle='--', label='Peak threshold')
            axhline(y=mean(rate_signal[int(baseline_start/dt):]), c='gray', linestyle='dotted', label='Baseline')
            legend()
            gca().set(xlabel='Time [ms]', ylabel='Rates [kHz]', xlim=(min(time),max(time)))
            savefig(path_to_save_figures + pdf_file_name +'.png')
            savefig(path_to_save_figures + pdf_file_name +'.eps')
            pdf.savefig(fig)

    return time, rate_signal, peaks

def define_event(n_group, pop_rate_monitor, threshold_in_sd, smooth_win=0.5, baseline_start=0, baseline_end=400, plot_peaks_bool=False):
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
    time_signal, pop_rate_signal, max_peak_indexes = find_events(n_group, pop_rate_monitor, threshold_in_sd, smoothing=smooth_win, plot_peaks=False)
    baseline = mean(pop_rate_signal[int(baseline_start/dt):int(baseline_end/dt)]) # Change BASELINE to only the mean of first 400 ms.
    thr = baseline + threshold_in_sd * std(pop_rate_signal) 
    thr_2sd = baseline+2*std(pop_rate_signal)
    event_props = {}
    simulation_props = {}
    simulation_props['Total_time'] = time_signal
    simulation_props['Total_signal'] = pop_rate_signal
    simulation_props['Max_peak_index'] = max_peak_indexes
    simulation_props['Baseline'] = baseline
    simulation_props['Threshold'] = thr
    simulation_props['Threshold_2sd'] = thr_2sd
    pdf_file_name = f'Net_{name_network}_allevents_G_{n_group.name[-2].upper()}_tr_{threshold_in_sd}_smoothwin_{smooth_win}'
    with PdfPages( path_to_save_figures + pdf_file_name + '.pdf') as pdf:    
        for i, index in enumerate(max_peak_indexes):
            print(i, index)
            event_props[i+1] = {}
            
            start_event_approx = int(index - 70/dt)
            end_event_appox = int(index + 70/dt)
            event_ranged = pop_rate_signal[start_event_approx:end_event_appox]
            time_ranged = time_signal[start_event_approx:end_event_appox]
    #        peaks, props = signal.find_peaks(event_ranged, height = baseline, distance = 4/dt, prominence=1, width=[1000,4000])
            peaks, prop = signal.find_peaks(event_ranged, height = thr_2sd, distance = 3/dt, prominence=0.5)
            peaks_2, prop_2 = signal.find_peaks(event_ranged, height = baseline, distance = 3/dt, prominence=0.5)
            index_peak_max = np.where(time_ranged[peaks] == time_ranged[index-start_event_approx])[0][0]
            outliers = np.where(diff(peaks)>8000)[0]
            if sum(outliers<index_peak_max) > 0: peaks = peaks[outliers[outliers<index_peak_max][-1]+1:]
            index_peak_max = np.where(time_ranged[peaks] == time_ranged[index-start_event_approx])[0][0]
            outliers_2 = np.where(diff(peaks)>8000)[0]
            if sum(outliers_2>=index_peak_max) > 0: peaks = peaks[:outliers_2[0]+1]
            if len(peaks)<2: 
                del event_props[i+1]
                continue
            first_peak_event_index = peaks[0]
            pre_peak = np.where(first_peak_event_index==peaks_2)[0][0]
            last_peak_event_index = peaks[-1]
            post_peak = np.where(last_peak_event_index==peaks_2)[0][0]
            if pre_peak == 0: start_index_event = np.argmin(event_ranged[:first_peak_event_index])
            else: start_index_event = np.argmin(event_ranged[peaks_2[pre_peak-1]:first_peak_event_index]) + peaks_2[pre_peak-1]
            if len(peaks_2)==post_peak+1: end_index_event = np.argmin(event_ranged[last_peak_event_index:]) + last_peak_event_index
            else: end_index_event = np.argmin(event_ranged[last_peak_event_index:peaks_2[post_peak+1]]) + last_peak_event_index
            
            index_peak_max = np.where(time_ranged[peaks] == time_ranged[index-start_event_approx])[0][0]
            time_event = (np.arange(0, len(time_ranged[start_index_event:end_index_event])) -  (peaks[index_peak_max]-start_index_event )) * dt
            event = event_ranged[start_index_event:end_index_event]
            
            event_props[i+1]['Max_peak_time'] = np.round(index*dt,3)
            event_props[i+1]['Time_array'] = time_event
            event_props[i+1]['Signal_array'] = event
            event_props[i+1]['Duration']= len(time_event)*dt
            event_props[i+1]['Index_start']= start_event_approx + start_index_event
            event_props[i+1]['Index_end']= start_event_approx + end_index_event
    
            if plot_peaks_bool:
    
                fig = figure(figsize=(21/cm, 10/cm))
    
                plot(time_ranged, event_ranged, 'k')
                axhline(y=baseline, c='gray', linestyle='dotted', label='Baseline')
                axhline(y=thr_2sd, c='gray', linestyle='dashed', label='Threshold 2 sd')
                scatter(time_ranged[peaks[first_peak_event_index:last_peak_event_index+1]], event_ranged[peaks[first_peak_event_index:last_peak_event_index+1]], c='orange')        
                scatter(time_ranged[peaks], event_ranged[peaks], c='orange')
                scatter(time_ranged[index-start_event_approx], event_ranged[index-start_event_approx], c='r')
                scatter(time_ranged[start_index_event], event_ranged[start_index_event], c='b', label='Start_end')
                scatter(time_ranged[end_index_event], event_ranged[end_index_event], c='b')
                legend()
                gca().set(xlabel='Time [ms]', ylabel='Rates [kHz]', title=f'Network_{name_network[-1]}_Event_{i+1}')
    
#                savefig(path_to_save_figures + pdf_file_name +'.png')
                pdf.savefig(fig)
                plt.close(fig)
            
    return event_props, simulation_props

def prop_events(n_group, pop_rate_monitor, threshold_in_sd, plot_peaks_bool=False):
    '''
    Function to determine the properties of each event such as number of peaks, 
    instantaneous frequency, mean frequency, spikes per cycle, etc.
    
    '''
       
    events_dict, simulation_dict = define_event(n_group, pop_rate_monitor, threshold_in_sd, plot_peaks_bool=True)
    
    for i, event in enumerate(events_dict.keys()):
#        if len(events_dict[event])==0: 
#            del events_dict[event]
#            continue
        print(event)
        time = events_dict[event]['Time_array']
        signal_event = events_dict[event]['Signal_array']
        baseline = simulation_dict['Baseline']
        threshold = simulation_dict['Threshold']
        threshold_2sd = simulation_dict['Threshold_2sd']
        peaks, props = signal.find_peaks(signal_event, height = threshold_2sd, distance = 3/dt, prominence=0.5)
#        peaks, props = signal.find_peaks(signal_event, height = baseline, distance = 4/dt)
#        index_out = where(props['prominences'] < 1)[0]
        left_bases = props['left_bases']
        right_bases = props['right_bases']
        failed_left_calc = where(diff(left_bases)==0)[0]
        left_bases[failed_left_calc+1] = right_bases[failed_left_calc]
        failed_right_calc = where(diff(right_bases)==0)[0]
        right_bases[failed_right_calc] = left_bases[failed_right_calc+1]

        
        freq = 1 / (diff(peaks)*dt/1000)   # Divided by 1000 because of ms
        peak_heights = props['peak_heights']
        indexes_freq = (diff(peaks)/2).astype(int) + peaks[:-1]
        min_dist_peak_tobase_xaxis = asarray(asmatrix(vstack((abs(peaks-left_bases), abs(peaks-right_bases)))).min(0))[0]
        start_cycles = peaks - min_dist_peak_tobase_xaxis
        end_cycles = peaks + min_dist_peak_tobase_xaxis
        spikes_percycle = np.zeros(len(peaks))
        long_slope = stats.linregress(time[indexes_freq], freq).slope
        mask = np.where((time[indexes_freq]<20) & (time[indexes_freq]>-20))[0]
        short_time = time[indexes_freq][mask]
        short_freq = freq[mask] 
        short_slope = stats.linregress(short_time, short_freq).slope
#        res_width = signal.peak_widths(signal_event, peaks, rel_height=1)
#        res_width_half = signal.peak_widths(signal_event, peaks, rel_height=0.5)
        events_dict[event]['Oscillation_events'] = {}
        for j, peak_cycle in enumerate(peaks):
            events_dict[event]['Oscillation_events']['Start_end'] = vstack((start_cycles, end_cycles))
            spikes_percycle[j] = trapz(y=signal_event[start_cycles[j]:end_cycles[j]], x=time[start_cycles[j]:end_cycles[j]], dx=dt)
            
        
        events_dict[event]['Num_peaks'] = len(peaks)
        events_dict[event]['Peaks_indexes'] = peaks
        events_dict[event]['Peaks_heights'] = peak_heights
        events_dict[event]['Instant_frequency'] = freq
        events_dict[event]['Indices_frequency'] = indexes_freq
        events_dict[event]['Mean_frequency'] = mean(freq)
        events_dict[event]['Spikes_per_cycle'] = spikes_percycle
        events_dict[event]['Short_freq_time_slope'] = short_slope
        events_dict[event]['Long_freq_time_slope'] = long_slope

        if plot_peaks_bool:
            pdf_file_name = f'Net_{name_network}_Event_{event}_G_{n_group.name[-2].upper()}_tr_{threshold_in_sd}_cut'
            with PdfPages( path_to_save_figures + pdf_file_name + '.pdf') as pdf:
                fig = figure(figsize=(10/cm, 10/cm))
    
                plot(time, signal_event, 'k')
                axhline(y=baseline, c='gray', linestyle='dotted', label='Baseline')
                axhline(y=threshold, c='gray', linestyle='dashed', label='Threshold')
                scatter(time[peaks], signal_event[peaks], c='orange')        
#                scatter(time[index-start_event_approx], event_ranged[index-start_event_approx], c='r')
#                scatter(time[start_index], event_ranged[start_index], c='b', label='Start_end')
#                scatter(time_ranged[end_index], event_ranged[end_index], c='b')
                legend()
                gca().set(xlabel='Time [ms]', ylabel='Rates [kHz]', title=f'Network_{name_network[-1]}_Event_{i+1}_cut')
    
                savefig(path_to_save_figures + pdf_file_name +'.png')
                pdf.savefig(fig)        
                
    return events_dict, simulation_dict

# MIGHT NOT BE USED ----------------------------------------------start------------------------------------------
#def plot_all_simulation(n_group, pop_rate_monitor, pop_spikes_monitor, pop_event_monitor, threshold_in_sd):
#    '''
#    Plot simulation over time with peak detection
#    '''
#    events_dict, simulation_dict = prop_events(n_group, pop_rate_monitor, threshold_in_sd, plot_peaks_bool=False)
#    
#    time = simulation_dict['Total_time']
#    rate_signal = simulation_dict['Total_signal']
#    index_peaks = simulation_dict['Max_peak_index']
#    baseline = simulation_dict['Baseline']
#    
#    pdf_file_name = f'All_simulation_G_{n_group.name[-2].upper()}_tr_{threshold_in_sd}_{name_network}'
#    with PdfPages( path_to_save_figures + pdf_file_name + '.pdf') as pdf:
#        fig = figure(figsize=(10/cm, 10/cm))
#        plot(time, rate_signal, c='k')
#        scatter(index_peaks*dt, rate_signal[index_peaks], c='r')
#    #    axhline(y=thr, c='gray', linestyle='--', label='Peak threshold')
#        axhline(y=baseline, c='gray', linestyle='dotted', label='Baseline')
#        legend()
#        gca().set(xlabel='Time [ms]', ylabel='Rates [kHz]', xlim=(0,max(time)))
#        
#        savefig(path_to_save_figures + pdf_file_name +'.png')
#        pdf.savefig(fig)

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
    pdf_file_name = f'All_events_prop_G_{n_group.name[-2].upper()}_tr_{threshold_in_sd}_{name_network}'
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
    
    
    
# MIGHT NOT BE USED ----------------------------------------------start------------------------------------------

#def ripple_prop(n_group, pop_rate_monitor, pop_spikes_monitor, pop_event_monitor, threshold_in_sd, plot_peaks_bool=False):
#    
#    '''
#    With the dictionary of all events (few ripples per event). 
#    In this function we check what about the spikes in each ripple oscillation
#    '''
##    pop_spikes_monitor = M_E
##    pop_event_monitor = M_DS    
#    pdf_file_name = f'Spiking_properties_G_{n_group.name[-2].upper()}_th_{threshold_in_sd}_{name_network}'
#    with PdfPages(path_to_save_figures + pdf_file_name + '.pdf') as pdf:
#        events_dict, simulation_dict = prop_events(n_group, pop_rate_monitor, threshold_in_sd, plot_peaks_bool)
#        total_events = len(events_dict.keys())
#        fig = figure(figsize=(18/cm,23.7/cm))
#        gs0 = gridspec.GridSpec(len(events_dict.keys()), 2, figure=fig, wspace=0.5/cm, hspace=0.2/cm)
#
#        for i, event in enumerate(events_dict.keys()):
#            start_time = events_dict[event]['Index_start']*dt
#            end_time = events_dict[event]['Index_end']*dt
#            neurons_spike, isi_spikes = get_index_neurons_from_window_of_time(pop_spikes_monitor, start_time, end_time)
#            neurons_DS, isi_DS = get_index_neurons_from_window_of_time(pop_event_monitor, start_time, end_time)
#    #        neurons_no_event_spike, isi_noSWR_spikes = get_index_neurons_from_window_of_time(pop_spikes_monitor, start_time-100, end_time-100)
#    #        neurons_no_event_DS, isi_noSWR_DS_spikes = get_index_neurons_from_window_of_time(pop_event_monitor, start_time-100, end_time-100)
#            list_neurons = []
#            spikes_per_neuron = []
#            for neuron in neurons_spike.keys():
#                list_neurons.append(neuron)
#                spikes_per_neuron.append(len(neurons_spike[neuron]))
#            gssub = gs0[i,0].subgridspec(1, 2)
#            plt.suptitle("GridSpec Inside GridSpec")
#            ax_hist = fig.add_subplot(gssub[0])
#            ax_isi = fig.add_subplot(gssub[1])
#            ax_hist.bar(list_neurons, spikes_per_neuron, color='k')
#            ax_isi.hist(isi_spikes*1000, color='k', range = (0,50))
#            ax_hist.set(xlim=(0,900))
#            ax_hist.yaxis.set_major_locator(MaxNLocator(integer=True))
#            ax_isi.set(xlim=(0,50))
#            ax_isi.yaxis.tick_right()
#            ax_isi.yaxis.set_major_locator(MaxNLocator(integer=True))
#            ax_isi.text(x=25,y=ax_isi.get_ylim()[1]*0.8, s=f'CV= {np.round(np.var(isi_spikes*1000) / np.mean(isi_spikes*1000),2)}')
#            if i<total_events-1: 
#                ax_hist.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
#                ax_isi.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
#            else:
#                ax_hist.set(xlabel='Neuron index')
#                ax_isi.set(xlabel='ISI at event [ms]')
#            if i==0:
#                ax_hist.set(title='Spikes/neuron')
#                ax_isi.set(title='ISI distribution')            
#                
#            list_neurons_DS = []
#            spikes_per_neuron_DS = []
#            for neuron in neurons_DS.keys():
#                list_neurons_DS.append(neuron)
#                spikes_per_neuron_DS.append(len(neurons_DS[neuron]))
#            gssub = gs0[i,1].subgridspec(1, 2)
#            ax_hist = fig.add_subplot(gssub[0])
#            ax_isi = fig.add_subplot(gssub[1])
#            ax_hist.bar(list_neurons_DS, spikes_per_neuron_DS, color='k')
#            ax_isi.hist(isi_DS*1000, color='k', range = (0,50))        
#            ax_hist.set(xlim=(0,900))
#            ax_hist.yaxis.set_major_locator(MaxNLocator(integer=True))
#            ax_isi.set(xlim=(0,50))
#            ax_isi.yaxis.tick_right()
#            ax_isi.yaxis.set_major_locator(MaxNLocator(integer=True))
#            ax_isi.text(x=25,y=ax_isi.get_ylim()[1]*0.8, s=f'CV= {np.round(np.var(isi_DS*1000) / np.mean(isi_DS*1000),2)}')
#            if i<total_events-1: 
#                ax_hist.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
#                ax_isi.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
#            else:
#                ax_hist.set(xlabel='Neuron index')
#                ax_isi.set(xlabel='ISI at event [ms]')
#            if i==0:
#                ax_hist.set(title='Spikes/neuron')
#                ax_isi.set(title='ISI distribution')  
#                
#        plt.savefig(path_to_save_figures + pdf_file_name +'.png')
#        pdf.savefig(fig)

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
    num_neurons = len(n_group.i)
    array_all_spikes_allevents = np.zeros((2,2))
    pdf_file_name = f'Spiking_relation_AVERAGED_events_G_{n_group.name[-2].upper()}_th_{threshold_in_sd}_Net_{name_network[-1]}'
    with PdfPages(path_to_save_figures + pdf_file_name + '.pdf') as pdf:
        fig, ax = subplots(1, 1, figsize=(18/cm,23.7/cm), sharex=True, sharey=True)
        
        for j, event in enumerate(events_dict.keys()):
            start_time = events_dict[event]['Index_start']*dt
            end_time = events_dict[event]['Index_end']*dt
            neurons_spike, isi_spikes = get_index_neurons_from_window_of_time(pop_spikes_monitor, start_time, end_time)
            neurons_DS, isi_DS = get_index_neurons_from_window_of_time(pop_event_monitor, start_time, end_time)
    #        neurons_no_event_spike, isi_noSWR_spikes = get_index_neurons_from_window_of_time(pop_spikes_monitor, start_time-100, end_time-100)
    #        neurons_no_event_DS, isi_noSWR_DS_spikes = get_index_neurons_from_window_of_time(pop_event_monitor, start_time-100, end_time-100)
            array_all_spikes= np.zeros((num_neurons, 2))
            for index in np.arange(num_neurons):
                if index in neurons_DS: array_all_spikes[index, 0] = len(neurons_DS[index])
                else: array_all_spikes[index, 0] = 0
                if index in neurons_spike: array_all_spikes[index, 1] = len(neurons_spike[index])
                else: array_all_spikes[index, 1] = 0
            array_all_spikes_allevents = np.append(array_all_spikes_allevents, array_all_spikes, axis=0) 
            
        array_all_spikes_allevents = array_all_spikes_allevents[2:,:]
#        res = stats.linregress(array_all_spikes_allevents[0], array_all_spikes_allevents[1])
#        x_val = np.arange(25)
#        y_val = res.slope * x_val + res.intercept 
        points = list(set(zip(array_all_spikes_allevents[:,0],array_all_spikes_allevents[:,1])))
        count = [len([x for x,y in zip(array_all_spikes_allevents[:,0],array_all_spikes_allevents[:,1]) if x==p[0] and y==p[1]]) for p in points] 
        plot_x=[k[0] for k in points]
        plot_y=[k[1] for k in points]
        mask_spikes = np.where(np.array(plot_y)>=2)[0]
#        res2 = stats.linregress(plot_x, plot_y)
#        y_val = res2.slope * x_val + res2.intercept 
        print(f'Number of neurons that spike 2 or more times per event {np.sum(count[mask_spikes])}')
        count=np.array(count)/total_events        
        mask = np.where(count>0)[0]
        pl = ax.scatter(np.array(plot_x)[mask],np.array(plot_y)[mask],c=np.log(count[mask]),s=100*count[mask]**0.5,cmap='viridis')
#        ax.plot(x_val, y_val, color='k')
        ax.set(ylabel='Number of spikes')
        ax.set_ylim(bottom=-1)
        plt.colorbar(pl, ax=ax,label='Log number of neurons per event')
        ax.set(xlabel='Number of dendritic spikes')
        fig.suptitle('Spikes behavior. Averaged per event.')
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

def cells_information_one_event(n_group, pop_rate_monitor, pop_spikes_monitor, pop_state_monitor,dendritic_monitor, threshold_in_sd = 3):
    '''
    Function to observe how the voltages, currents and dendritic spikes appear in the neurons within one event. Speciallz before and after
    '''
    if name_network != 'long_allneurons_event_network_3': return
    else:    
        events, sim = prop_events(n_group, pop_rate_monitor, threshold_in_sd, plot_peaks_bool=False)
        peaks_indexes = (events[1]['Peaks_indexes'] + events[1]['Index_start']) * dt 
        pdf_file_name = f'Cells_activity_G_{n_group.name[-2].upper()}_th_{threshold_in_sd}_{name_network}'
        with PdfPages(path_to_save_figures + pdf_file_name + '.pdf') as pdf:
#            num_neurons = len(n_group.i)
            neuron_index= np.arange(30)
            time = pop_rate_monitor.t/ms
            rate_signal = pop_rate_monitor.smooth_rate(width=0.5*ms)*len(n_group) / kHz
            for index in neuron_index:
                fig, ax = subplots(2, 1, figsize=(18/cm,12/cm), sharex=True)
                voltage = pop_state_monitor.v[index,:]/mV
                exc_current = pop_state_monitor.I_A[index,:]/nA
                inh_current = pop_state_monitor.I_G[index,:]/nA
                dendritic_spikes_times = dendritic_monitor.t[np.where(dendritic_monitor.i==index)[0]]/ms
                spike_times = pop_spikes_monitor.t[np.where(pop_spikes_monitor.i==index)[0]]/ms
                
                ax[0].plot(time, voltage, color='k')
                ax2 = ax[0].twinx()
                ax2.plot(time, rate_signal, color='silver', alpha=0.8, label='Population rate')
                lims = ax[0].get_ylim()
                if n_group.name[-2].upper() =='E': ax[0].vlines(x=dendritic_spikes_times,  ymin = lims[0], ymax = lims[1], colors='steelblue', linewidth=0.5, label='Dendritic spike', linestyle='--')
                ax[0].vlines(x=spike_times, ymin = lims[0], ymax = lims[1], colors='blue', linewidth=0.5, label='Spike')
                ax[0].set(ylabel='Voltage [mV]', title=f'Neuron {index}')
                ax[0].legend(loc=2)
                ax2.set(ylabel='Rate [Hz]')
                
                
                ax[1].plot(time, exc_current, color='green', label='Excitatory')
                ax[1].plot(time, inh_current, color='indigo', label='Inhibitory')
                ax2 = ax[1].twinx()
                ax2.plot(time, rate_signal, color='silver', alpha=0.8, label='Population rate')
                lims = ax[1].get_ylim()
                if n_group.name[-2].upper() =='E': ax[1].vlines(x=dendritic_spikes_times,  ymin = lims[0], ymax = lims[1], colors='steelblue', linewidth=0.5, linestyle='--')
                ax[1].vlines(x=spike_times, ymin = lims[0], ymax = lims[1], colors='blue', linewidth=0.5)
                ax[1].set(ylabel='Current [nA]',  xlabel='Time [ms]', xlim=(1700,1800))
                ax[1].legend(loc=2)
                ax2.set(ylabel='Rate [Hz]')
                ax2.legend(loc=1)
                
                pdf.savefig(fig)
                plt.close(fig)


def get_wavelet_information(n_group, pop_rate_monitor, threshold_in_sd):
    '''
    Function to evaluate the ripple event based on the frequency domain and power spectrum
    '''
    events, sim = prop_events(n_group, pop_rate_monitor, threshold_in_sd, plot_peaks_bool=False)
#    total_signal = sim['Total_signal']
    total_signal = pop_rate_monitor.smooth_rate(width=0.1*ms)*len(n_group) / kHz # Smothing the signal by 0.1 ms
    total_time = sim['Total_time']
    peak_indexes = sim['Max_peak_index']
    dict_wavelet = {}
    print(f'Num_events: {len(events.keys())}')
    for i, event in enumerate(events.keys()):
        print(f'Event: {event}')
        dict_wavelet[event] = {}
        start = events[event]['Index_start']
        end = events[event]['Index_end']
#        time_event = events[event]['Time_array']

        wspec, wspec_extent, instfreq, instpower, freq_onset_inst, instcoherence, Pthr, ifreq_discr_t, ifreq_discr\
        = f_oscillation_analysis_transient(total_signal, dt=dt, baseline_window=[0, 400], target_window = [start*dt, end*dt], \
                                           expected_freq = 200, fmin=100, plot=False)
        
        dict_wavelet[event]['Max_peak_time'] = total_time[peak_indexes[i]]
        dict_wavelet[event]['Time_event_from_alltime'] = total_time[start:end]
        dict_wavelet[event]['Instantaneous_frequency'] = instfreq
        dict_wavelet[event]['Instantaneous_power'] = instpower
        dict_wavelet[event]['Discrete_frequency'] = ifreq_discr
        dict_wavelet[event]['Discrete_frequency_time'] = ifreq_discr_t
        dict_wavelet[event]['wspec'] = wspec
        regression = stats.linregress(ifreq_discr_t, ifreq_discr)
        dict_wavelet[event]['slope_intercept'] = [regression.slope, regression.intercept]  # Hz/ms slope
        
   
    return dict_wavelet

    
def discrete_freq_analysis(dict_information, n_group, wavelet = True):
    '''
    Function to plot and analyse the instantaneous frequency
    dict_information could come from wavelet or from simple period discrete frequency analysis
    '''
    num_events = len(dict_information.keys())
    colors = cmr.take_cmap_colors('cividis', num_events, return_fmt='hex') 
    if wavelet: pdf_file_name = f'Discrete_frequencies_all_G_{n_group.name[-2].upper()}_th_{threshold_in_sd}_{name_network}_wavelet'
    pdf_file_name = f'Discrete_frequencies_all_G_{n_group.name[-2].upper()}_th_{threshold_in_sd}_{name_network}'
    with PdfPages(path_to_save_figures + pdf_file_name + '.pdf') as pdf:
#        fig, ax = plt.subplots(num_events, 1, figsize=(21/cm, 21/cm), sharex=True, sharey=True)
        for i, event in enumerate(dict_information.keys()):
            fig, ax = plt.subplots(1, 1, figsize=(21/cm, 21/cm))
            if wavelet:
                max_peak_time = dict_wavelet_information[event]['Max_peak_time']
                frequencies_discrete = dict_wavelet_information[event]['Discrete_frequency']
                time_discrete = dict_wavelet_information[event]['Discrete_frequency_time'] - max_peak_time
                slope = dict_wavelet_information[event]['slope_intercept'][0]
                intercept = dict_wavelet_information[event]['slope_intercept'][1]
                time_values = np.arange(dict_wavelet_information[event]['Discrete_frequency_time'][0], dict_wavelet_information[event]['Discrete_frequency_time'][-1]+dt, dt)
                regression_values = time_values * slope + intercept
            else:
                max_peak_time = dict_information[event]['Max_peak_time']
                frequencies_discrete = dict_information[event]['Instant_frequency']
                time_values = dict_information[event]['Time_array']
                time_discrete = time_values[dict_information[event]['Indices_frequency']]
                res = stats.linregress(time_discrete, frequencies_discrete)
                slope = res.slope
                intercept = res.intercept
                regression_values = time_values * slope + intercept
                
            ax.grid(zorder= 0)
            ax.scatter(time_discrete, frequencies_discrete, color=colors[i], marker='.', label=f'Event {event}')
            if wavelet: ax.plot(time_values-max_peak_time, regression_values, color=colors[i], linestyle='-')
            else: ax.plot(time_values, regression_values, color=colors[i], linestyle='-')
            ax.legend()
            
            ax.set(xlabel='Time w.r.t. highest peak', ylabel = 'Frequency [Hz]')
            fig.subplots_adjust(right=0.8)
            pdf.savefig(fig)        
        
        plt.close('all')

def plot_all_discrete_frequencies(dict_information, n_group, wavelet=True, short_event = True, threshold_in_sd=3):
    '''
    Function to plot and analyse the instantaneous frequency
    if short_event is true. Then we only take the frequencies between -20 to 20 ms from the max peak
    '''
    num_events = len(dict_information.keys())
    colors = cmr.take_cmap_colors('cividis', num_events, return_fmt='hex') 
    times = np.array([])
    frequencies = np.array([])
    if short_event: pdf_file_name = f'Discrete_frequencies_SUMMARY_G_{n_group.name[-2].upper()}_th_{threshold_in_sd}_{name_network}_short_event'
    else: pdf_file_name = f'Discrete_frequencies_SUMMARY_G_{n_group.name[-2].upper()}_th_{threshold_in_sd}_{name_network}'
    with PdfPages(path_to_save_figures + pdf_file_name + '.pdf') as pdf:
        fig, ax = plt.subplots(1, 1, figsize=(21/cm, 10/cm))
        for i, event in enumerate(dict_information.keys()):
            if wavelet:
                max_peak_time = dict_information[event]['Max_peak_time']
                frequencies_discrete = dict_information[event]['Discrete_frequency']
                time_discrete = dict_information[event]['Discrete_frequency_time'] - max_peak_time     
            else:
                frequencies_discrete = dict_information[event]['Instant_frequency']
                time_values = dict_information[event]['Time_array']
                time_discrete = time_values[dict_information[event]['Indices_frequency']]
                
            times = np.append(times, time_discrete)
            frequencies = np.append(frequencies, frequencies_discrete)

            ax.scatter(time_discrete, frequencies_discrete, color=colors[i], marker='.')
        if short_event:
            mask = np.where((times<20) & (times>-20))[0]     
            frequencies = frequencies[mask]
            times = times[mask]
#        from sklearn.svm import SVC
        t = np.arange(min(times), max(times), 0.1)
#        model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
        model = SVR(kernel='rbf')
        freq_predict = model.fit(times.reshape(-1, 1),frequencies).predict(t.reshape(-1, 1))
        regression = stats.linregress(times, frequencies)
#        t = np.arange(min(times), max(times), 0.1)
        y = regression.slope*t +regression.intercept
        ax.plot(t,y, 'k')
        ax.plot(t, freq_predict, 'gray')
        ax.grid(zorder= 0)
        ax.set(xlabel='Time w.r.t. highest peak [ms]', ylabel='Frequency [Hz]')
        fig.text(0.7, 0.75, f'Slope= {np.round(regression.slope,4)} [Hz/ms]') 
        plt.savefig(path_to_save_figures + pdf_file_name +'.png')
        pdf.savefig(fig)     

def get_and_plot_correlation_freq_rates_height(dict_information, n_group, previous = False, threshold_in_sd=3):
    '''Function to plot the correlation between calculater frequencies and the height of the peak (previous or posterior)
    to the 1/period calculated.
    dict information is e.g. events_dict, simulation_dict = prop_events(G_E, R_E, 3, plot_peaks_bool=False)
    '''
    num_events = len(dict_information.keys())
    colors = cmr.take_cmap_colors('cividis', num_events, return_fmt='hex') 
    heights = np.array([])
    frequencies = np.array([])
    times = np.array([])
    if previous: pdf_file_name = f'Scatter_frequency_previous_ratepeaks_G_{n_group.name[-2].upper()}_th_{threshold_in_sd}_{name_network}'
    else: pdf_file_name = f'Scatter_frequency_posterior_ratepeaks_G_{n_group.name[-2].upper()}_th_{threshold_in_sd}_{name_network}'
    with PdfPages(path_to_save_figures + pdf_file_name + '.pdf') as pdf:
        fig, ax = plt.subplots(1, 1, figsize=(15/cm, 15/cm))
        for i, event in enumerate(dict_information.keys()):
            frequencies_discrete = dict_information[event]['Instant_frequency']
            time_values = dict_information[event]['Time_array']
            time_discrete = time_values[dict_information[event]['Indices_frequency']]

            if previous: heights_discrete = dict_information[event]['Peaks_heights'][:-1]
            else: heights_discrete = dict_information[event]['Peaks_heights'][1:]

            heights = np.append(heights, heights_discrete)
            frequencies = np.append(frequencies, frequencies_discrete)
            times = np.append(times, time_discrete)
            if len(np.where(frequencies_discrete>250)[0])>0: print(event)
#            ax.scatter(frequencies_discrete, heights_discrete, color=colors[i], marker='.')
        mask_before = np.where(times<0)[0]
        mask_after = np.where(times>0)[0]
        ax.scatter(frequencies[mask_before], heights[mask_before], color='indigo', label='Peaks before max', marker='.')  
        ax.scatter(frequencies[mask_after], heights[mask_after], color='green', label='Peaks after max', marker='.') 
        ylim_val = ax.get_ylim()
        ax.axvline(x=np.mean(frequencies[mask_before]), color='indigo', linestyle='--')
        ax.axvline(x=np.mean(frequencies[mask_after]), color='green', linestyle='--')
#        mask = np.where(frequencies<250)[0]
        regression = stats.linregress(frequencies, heights)
        f = np.arange(min(frequencies), max(frequencies), 0.1)
        y = regression.slope*f +regression.intercept
        ax.plot(f,y, 'k')
        ax.legend()
        ax.grid(zorder= 0)
        ax.set(xlabel='Frequency of ripple event [Hz]', ylabel='Population rates [kHz]')
        fig.text(0.5, 0.75, f'Correlation coefficient = {np.round(regression.rvalue,4)}') 
        plt.savefig(path_to_save_figures + pdf_file_name +'.png')
        pdf.savefig(fig)     
        
# Using the csv dataframes with properties of events--------------------------------------------------------------
def plot_correlation_per_event( n_group, previous = False):
    '''Function to plot the correlation per event between calculated periods and the height of the peak (previous or posterior)
    to the period calculated.
    '''
    data_ex = pd.read_pickle('/home/nunez/New_repo/Plots_populations_comparison/'+f'Properties_events_ex_{name_network}.pkl')
    if previous: pdf_file_name = f'Scatter_frequency_previous_ratepeaks_G_{n_group.name[-2].upper()}_{name_network}_perEvent'
    else: pdf_file_name = f'Scatter_frequency_posterior_ratepeaks_G_{n_group.name[-2].upper()}_{name_network}_perEvent'
    with PdfPages(path_to_save_figures + pdf_file_name + '.pdf') as pdf:        
        for evs in data_ex.index:
            fig, ax = plt.subplots(1, 2, figsize=(21/cm, 12/cm), sharey=True)
            
            periods = 1000/data_ex.loc[evs]['Instant_frequency']
            periods_times = data_ex.loc[evs]['Time_array'][data_ex.loc[evs]['Indices_frequency']]
            peak_heights = data_ex.loc[evs]['Peaks_heights']
            
            mask_before = np.where(periods_times<0)[0]
            mask_after = np.where(periods_times>0)[0]
            
            ax[0].plot(periods_times, periods, marker='*', color='k')
            ax[0].axvline(x=0, color='gray', linestyle='--')
            ax[0].set(xlabel='Time w.r.t. max peak [ms]', ylabel='Period [ms]')
            
            if previous: peak_heights = peak_heights[:-1]
            else: peak_heights = peak_heights[1:]
            
            if len(mask_before)>0:
                corr_before = stats.linregress(peak_heights[mask_before], periods[mask_before]).rvalue
                ax[1].scatter(peak_heights[mask_before], periods[mask_before], color='indigo', label='Peaks before max', marker='.')
            if len(mask_after)>0:
                corr_after = stats.linregress(peak_heights[mask_after], periods[mask_after]).rvalue
                ax[1].scatter(peak_heights[mask_after], periods[mask_after], color='green', label='Peaks after max', marker='.')
            xlim_, ylim_ = ax[1].get_xlim(), ax[1].get_ylim()
            if len(mask_before)>0: ax[1].text(np.diff(xlim_)*0.7+xlim_[0],np.diff(ylim_)*0.7+ylim_[0], f'\u03C1 : {np.round(corr_before,4)}', color='indigo' )
            if len(mask_after)>0:ax[1].text(np.diff(xlim_)*0.7+xlim_[0],np.diff(ylim_)*0.6+ylim_[0], f'\u03C1 : {np.round(corr_after,4)}', color='green' )

            ax[1].legend()
            ax[1].set(xlabel='Population rate cycle [kHz]')

            pdf.savefig(fig) 
            plt.close('all')
            
#def check_average_shapes
#-------------------------------------------------------------------------------------------
    
def spikes_per_cycle(dict_information, n_group, pop_spikes_monitor, long_event=False):
    '''
    Fucntion to check how many spikes there are per cycle in event
    '''
    if long_event: event = 31
    else: event = 18
    pdf_file_name = f'Spikes_percell_percycle_G_{n_group.name[-2].upper()}_{name_network}_event_{event}'
    with PdfPages(path_to_save_figures + pdf_file_name + '.pdf') as pdf:
        fig, ax = plt.subplots(1, 2, figsize=(21/cm, 10/cm))
#        signal_array = dict_information[event]['Signal_array']
        time_start = dict_information[event]['Index_start'] * dt
        time_end = dict_information[event]['Index_end'] * dt
#        time_array = np.round(np.arange(time_start, time_end, 0.001),3)[:-1]
        spikes_times = pop_spikes_monitor.t/ms
        spikes_index = pop_spikes_monitor.i
        mask = np.where((spikes_times >=time_start) & (spikes_times <=time_end))[0]
        spikes_per_cycle = np.round(dict_information[event]['Spikes_per_cycle'],0)
        unique_index, counts_spikes = np.unique(spikes_index[mask], return_counts=True)
        
        ax[0].bar(np.arange(len(spikes_per_cycle))+1, spikes_per_cycle, color='gray')
        ax[0].set(xlabel='Cycle in event', ylabel='Number of cells spiking')
        ax[1].hist(counts_spikes, bins=np.arange(0.5,counts_spikes.max()+1, 1), histtype='step', color='k')
        ax[1].set(xlabel='Number of spikes per cell in whole event', ylabel='Amount of neurons')
        
        fig.tight_layout()
        plt.show()
        plt.savefig(path_to_save_figures + pdf_file_name +'.png')
        pdf.savefig(fig)     
    
    
    
    
    

def plot_histogram_slopes(dict_information, n_group, short_event=True):
    '''
    dict_information could be the dictorionary from wavelet analysis or events_dict
    short event only take the slopes from -20 to 20 ms
    '''
    slopes = np.array([])    
    if short_event: 
        pdf_file_name = f'Histogram_slopes_G_{n_group.name[-2].upper()}_{name_network}_short_event'
        type_slope = 'Short_freq_time_slope'
    else: 
        pdf_file_name = f'Histogram_slopes_G_{n_group.name[-2].upper()}_th_{threshold_in_sd}_{name_network}'
        type_slope = 'Long_freq_time_slope'
    for event in dict_information.keys():
        slopes = np.append(slopes, dict_information[event][type_slope])    
    with PdfPages(path_to_save_figures + pdf_file_name + '.pdf') as pdf:
        fig, ax = plt.subplots(1, 1, figsize=(21/cm, 10/cm))
        mean_val = np.mean(slopes)
        median_val = np.median(slopes)
        ax.hist(slopes, bins=100, histtype='step', color='k')
        ax.set(xlabel='Slopes [Hz/ms]')
        lims = ax.get_ylim()
        ax.axvline(x=mean_val, ymin=lims[0], ymax=lims[1], color='r', label=f'Mean value {np.round(mean_val, 4)}')
        ax.axvline(x=median_val, ymin=lims[0], ymax=lims[1], color='orange', label=f'Median value {np.round(median_val, 4)}')
        ax.legend()
        
        plt.savefig(path_to_save_figures + pdf_file_name +'.png')
        pdf.savefig(fig) 
    
    
    
# Store and restore long networks---------------------------------------------------------

def store_long_networks(time_simulation = 50000, neu_ext = False, neu_inh = False):
    
    seeds = [222,1001,6016, 770, 8, 769]
    network_num = [2,3,6,7,8,9]
    
    for i, net_num in enumerate(network_num):
        print(i, net_num)
        name_network = f'long_50000_network_{net_num}'

        network, monitors = Network_model_2(seed_num=seeds[i], sim_dur=time_simulation*ms, pre_run_dur=0*ms, total_neurons=1000, 
                                    scale_factor=1, dendritic_interactions=True, neurons_exc = neu_ext, neurons_inh = neu_inh)
        network.store(name='rand_net', filename = path_networks + name_network)
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
        
        time, signal, peaks = find_events(G_E, R_E, 3, name_net=name_network, baseline_start=0, baseline_end=300, plot_peaks=True)

def restore_net_and_dataframe():
    name_network = 'long_50000_network_2'
    #name_network = 'long_allneurons_event_network_3'  # For the event and all neurons the prerun is 1600 ms
    
    # In order to restore the long network it is necessary to first create an object.
    # Therefore, here I create a network of only 20 ms and on it, I restore the long one.
    dur_simulation=10
    network, monitors = Network_model_2(seed_num=1001, sim_dur=dur_simulation*ms, pre_run_dur=0*ms, total_neurons=1000, 
                                        scale_factor=1, dendritic_interactions=True, neurons_exc = False, neurons_inh = False)
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
    
    

#    
#    pdf_file_name_2 = f'Wavelet_AVERAGE_G_{n_group.name[-2].upper()}_th_{threshold_in_sd}_{name_network}'
#    with PdfPages(path_to_save_figures + pdf_file_name_2 + '.pdf') as pdf:
#        fig, ax = plt.subplots(2, 1, figsize=(21/cm, 5/cm), sharey=True, sharex=True)
#        ax[0].grid(zorder= 0)
#        ax[0].scatter(dict_sts['Time_all_max_power'], dict_sts['Mean_frequencies_nan'], c = dict_sts['Mean_powers_nan'], 
#          cmap=colormap, norm=normalize, marker='.', zorder=1, label = 'All data')
#        ax[1].grid(zorder= 0)
#        ax[1].scatter(dict_sts['Time_all_max_power'], dict_sts['Mean_frequencies'], c = dict_sts['Mean_powers'], 
#          cmap=colormap, norm=normalize, marker='.', zorder=1, label = 'Cut data')
#        fig.subplots_adjust(right=0.8)
#        cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
#        fig.colorbar(p, cax=cbar_ax, label='Power')
#
#        fig.text(0.06, 0.5, 'Frequency [Hz]', ha='center', va='center', rotation='vertical')  
#        fig.text(0.85, 0.5, 'Network rate [Hz]', ha='center', va='center', rotation='270')  
#        plt.savefig(path_to_save_figures + pdf_file_name_2 +'.png')
#        pdf.savefig(fig)              
#        

    
    
    

#def wavelet_analysis_all_events(n_group, pop_rate_monitor, threshold_in_sd, ):
#    '''
#    Function to evaluate the ripple event based on the frequency domain and power spectrum
#    '''
#    events, sim = prop_events(n_group, pop_rate_monitor, threshold_in_sd, plot_peaks_bool=False)
#    total_signal = sim['Total_signal']
#    total_time = sim['Total_time']
#    peak_indexes = sim['Max_peak_index']
#    dict_wavelet = {}
#
#    for i, event in enumerate(events.keys()):
#        dict_wavelet[event] = {}
#        start = events[event]['Index_start']
#        end = events[event]['Index_end']
#        time_event = events[event]['Time_array']
#
#        wspec, wspec_extent, instfreq, instpower, freq_onset_inst, instcoherence, Pthr, ifreq_discr_t, ifreq_discr\
#        = f_oscillation_analysis_transient(total_signal, dt=dt, baseline_window=[0, 400], target_window = [start*dt, end*dt], \
#                                           expected_freq = 200, fmin=100, plot=False)
#        
#        time_powers = np.arange(- np.argmax(instpower)*dt, (instpower.shape - np.argmax(instpower))*dt, dt)
#        dict_wavelet[event]['Time_array'] = time_event
#        dict_wavelet[event]['Time_array_max_power'] = np.around(time_powers, 3)
#        dict_wavelet[event]['Instantaneous_frequency'] = instfreq
#        dict_wavelet[event]['Instantaneous_power'] = instpower
#        dict_wavelet[event]['wspec'] = wspec
##            
#        dict_wavelet[event]['Time_inst_freq_discrete'] = ifreq_discr_t
#        dict_wavelet[event]['inst_freq_discrete'] = ifreq_discr
#   
#    return dict_wavelet
#
#def adjust_dict_wavelet(n_group, pop_rate_monitor, threshold_in_sd):
#    '''
#    Function to create all wavelet frequency arrays of the same size in order to average with respect to the max power
#    In order to do so, arrays of nan's will be concatenated
#    
#    return 
#        dict_wavelet_events modified
#        dict with mean_values and max power
#    '''    
#    
#    dict_wavelet_events = wavelet_analysis_all_events(n_group, pop_rate_monitor, threshold_in_sd)
#    events, sim = prop_events(n_group, pop_rate_monitor, threshold_in_sd, plot_peaks_bool=False)
#    dict_sts = {}
#    min_time, max_time, max_power = 0, 0, 0
#    
#    # Get maximum values for normalization and standarization
#    for event in dict_wavelet_events.keys():
#        time_powers = dict_wavelet_events[event]['Time_array_max_power']
#        powers = dict_wavelet_events[event]['Instantaneous_power']
#        frequencies = dict_wavelet_events[event]['Instantaneous_frequency']
#        min_time, max_time = min(min_time, time_powers[0]), max(max_time, time_powers[-1])
#        max_power = max(max_power, max(powers))
#        index_to_delete = np.where(frequencies[:len(frequencies)//2]==100)[0]
#        adjusted_frequencies = np.zeros(len(frequencies))
#        if len(index_to_delete) > 0: 
#            adjusted_frequencies[index_to_delete] = np.nan * np.ones(len(index_to_delete))
#            adjusted_frequencies[len(index_to_delete):] = frequencies[len(index_to_delete):]
#        else:
#            adjusted_frequencies = frequencies
#        dict_wavelet_events[event]['Instantaneous_frequency'] = adjusted_frequencies
#        dict_wavelet_events[event]['Time_delay_with_peak'] = time_powers[np.argmax(events[event]['Signal_array'])]
#        
#    
#    time_all = np.around(np.arange(min_time, max_time+dt, dt), decimals=3)
#    freq_matrix, powers_matrix = np.zeros((len(dict_wavelet_events.keys()), len(time_all))), np.zeros((len(dict_wavelet_events.keys()), len(time_all)))
#    
#    for event in dict_wavelet_events.keys():
#        frequencies = dict_wavelet_events[event]['Instantaneous_frequency']
#        powers = dict_wavelet_events[event]['Instantaneous_power']
#        wspec = dict_wavelet_events[event]['wspec']
#        
#        indexes_1 = where(time_all < dict_wavelet_events[event]['Time_array_max_power'][0])[0]
#        indexes_2 = where(dict_wavelet_events[event]['Time_array_max_power'][-1] <  time_all)[0]
#        
#        if len(indexes_1) > 0:
#            frequencies = np.append(np.nan * np.ones (len(indexes_1)), frequencies) 
#            powers = np.append(np.nan * np.ones (len(indexes_1)), powers)
#            wspec = np.append(np.nan * np.ones((350,len(indexes_1))), wspec, axis=1)
#        if len(indexes_2) > 0:
#            frequencies = np.append(frequencies, np.nan * np.ones (len(indexes_2))) 
#            powers = np.append(powers, np.nan * np.ones (len(indexes_2))) 
#            wspec = np.append(wspec, np.nan * np.ones((350,len(indexes_2))), axis=1)    
#            
#        freq_matrix[event-1, :] = frequencies
#        powers_matrix[event-1, :] = powers
#
#        dict_wavelet_events[event]['Time_array_max_power'] = time_all
#        dict_wavelet_events[event]['Instantaneous_frequency'] = frequencies
#        dict_wavelet_events[event]['Instantaneous_power'] = powers
#        dict_wavelet_events[event]['wspec'] = wspec
#        
#
#    dict_sts['Min_time'] = min_time
#    dict_sts['Max_time'] = max_time
#    dict_sts['Max_power'] = max_power
#    dict_sts['Time_all_max_power'] = time_all
#    dict_sts['Mean_frequencies_nan'] = np.nanmean(freq_matrix, axis=0)
#    dict_sts['Mean_frequencies'] = np.mean(freq_matrix, axis=0)
#    dict_sts['Mean_powers_nan'] = np.nanmean(powers_matrix, axis=0)
#    dict_sts['Mean_powers'] = np.mean(powers_matrix, axis=0)
#
#    
#    return dict_wavelet_events, dict_sts    
#        
#
#def wavelet_plot_all_events(n_group, pop_rate_monitor, threshold_in_sd):
#    '''
#    Function to plot the wavelet of the ripple events in a signal
#    '''
#    dict_wavelet_events_adjust, dict_sts = adjust_dict_wavelet(n_group, pop_rate_monitor, threshold_in_sd)
#    events, sim = prop_events(n_group, pop_rate_monitor, threshold_in_sd, plot_peaks_bool=False)
#    
#    colormap = plt.cm.viridis 
#    normalize = matplotlib.colors.Normalize(vmin=0, vmax=int(dict_sts['Max_power']) + 1.)
#    
#    pdf_file_name = f'Wavelet_G_{n_group.name[-2].upper()}_th_{threshold_in_sd}_{name_network}'
#    with PdfPages(path_to_save_figures + pdf_file_name + '.pdf') as pdf:
#        fig, ax = plt.subplots(len(events.keys()), 1, figsize=(21/cm, 12/cm), sharey=True, sharex=True)
#        for i, event in enumerate(events.keys()):
#            frequencies = dict_wavelet_events_adjust[event]['Instantaneous_frequency']
#            powers = dict_wavelet_events_adjust[event]['Instantaneous_power']
#            time_array_max_power = dict_wavelet_events_adjust[event]['Time_array_max_power']
#            signal_array = events[event]['Signal_array']
#            time_signal = events[event]['Time_array']
#            peak_delay = dict_wavelet_events_adjust[event]['Time_delay_with_peak']
#
#            ax[i].grid(zorder= 0)
#            p = ax[i].scatter(time_array_max_power, frequencies, c = powers, cmap=colormap, norm=normalize, marker='.', zorder=1)
#            ax[i].text(x= 40, y = 170, s=f'Event {event}', zorder=3)
#            ax2 = ax[i].twinx()
#            ax2.plot(time_signal+peak_delay, signal_array, 'k', zorder=2)
##            ax[i].set(ylabel='Frequency [Hz]')
#        fig.subplots_adjust(right=0.8)
#        cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
#        fig.colorbar(p, cax=cbar_ax, label='Power')
#
#        fig.text(0.06, 0.5, 'Frequency [Hz]', ha='center', va='center', rotation='vertical')  
#        fig.text(0.85, 0.5, 'Network rate [Hz]', ha='center', va='center', rotation='270')  
#        plt.savefig(path_to_save_figures + pdf_file_name +'.png')
#        pdf.savefig(fig)        
#    
#    pdf_file_name_2 = f'Wavelet_AVERAGE_G_{n_group.name[-2].upper()}_th_{threshold_in_sd}_{name_network}'
#    with PdfPages(path_to_save_figures + pdf_file_name_2 + '.pdf') as pdf:
#        fig, ax = plt.subplots(2, 1, figsize=(21/cm, 5/cm), sharey=True, sharex=True)
#        ax[0].grid(zorder= 0)
#        ax[0].scatter(dict_sts['Time_all_max_power'], dict_sts['Mean_frequencies_nan'], c = dict_sts['Mean_powers_nan'], 
#          cmap=colormap, norm=normalize, marker='.', zorder=1, label = 'All data')
#        ax[1].grid(zorder= 0)
#        ax[1].scatter(dict_sts['Time_all_max_power'], dict_sts['Mean_frequencies'], c = dict_sts['Mean_powers'], 
#          cmap=colormap, norm=normalize, marker='.', zorder=1, label = 'Cut data')
#        fig.subplots_adjust(right=0.8)
#        cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
#        fig.colorbar(p, cax=cbar_ax, label='Power')
#
#        fig.text(0.06, 0.5, 'Frequency [Hz]', ha='center', va='center', rotation='vertical')  
#        fig.text(0.85, 0.5, 'Network rate [Hz]', ha='center', va='center', rotation='270')  
#        plt.savefig(path_to_save_figures + pdf_file_name_2 +'.png')
#        pdf.savefig(fig)              
#        
        

#rate_signal, peak_per_event = find_events(n_group=G_E, pop_rate_monitor=R_E, threshold_in_sd=4, plot_peaks=True)
#
#
#
#plot(R_E.t / ms, R_E.smooth_rate(width=1*ms)*len(G_E) / kHz, color='k')
#


