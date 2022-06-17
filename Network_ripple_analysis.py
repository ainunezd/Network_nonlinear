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

from Network_model_2_two_populations import Network_model_2

# ----------------------------names and folders-------------------------

path_to_save_figures = '/home/nunez/Network_nonlinear/Plots/'
path_networks = '/home/nunez/Network_nonlinear/stored_networks/'

#name_figures = 'network_multiply_rates_by_pop_size_cg_changing'
name_network = 'long_random_network'

# In order to restore the long network it is necessary to first create an object.
# Therefore, here I create a network of only 20 ms and on it, I restore the long one.
dur_simulation=20
network, monitors = Network_model_2(sim_dur=dur_simulation*ms, pre_run_dur=0*ms, total_neurons=1000, scale_factor=1)
#network.store(name='rand_net', filename = path_networks + name_network)

network.restore(name='rand_net', filename = path_networks + 'long_random_network')

# Get monitors from the network
M_E = network.sorted_objects[-18]
M_I = network.sorted_objects[-17]
R_E = network.sorted_objects[-1]
R_I = network.sorted_objects[-2]  
State_G_ex = network.sorted_objects[2]
State_G_in = network.sorted_objects[3]
G_E = network.sorted_objects[0]
G_I = network.sorted_objects[1]

dt = 0.001


def find_events(n_group, pop_rate_monitor, threshold_in_sd, plot_peaks=True):
    '''
    Function to find the ripple events with in the network
    n_group: Neuron group
    pop_rate_monitor: Population rate monitor from either excitatory or inhibitory population (neuron group)
    threshold_in_sd: number of standard deviations to multiply for the baseline, in order to establish the threshold
        threshold = baseline * threshold_in_sd * sd
    returns
        time: time array for the rate monitor [unitless]
        rate_signal: the signal from the rate monitor [unitless]
        peaks: index of the maximum peak of event     
        
    
    '''

    rate_signal = pop_rate_monitor.smooth_rate(width=1*ms)*len(n_group) / kHz
    thr = mean(rate_signal) + threshold_in_sd * std(rate_signal) 
    peaks, prop = signal.find_peaks(rate_signal, height = thr, distance = 50/dt)
    time = pop_rate_monitor.t / ms
    
    if plot_peaks:
        figure()
        plot(time, rate_signal, c='k')
        scatter(peaks*dt, rate_signal[peaks], c='r')
        axhline(y=thr, c='gray', linestyle='--', label='Peak threshold')
        axhline(y=mean(rate_signal), c='gray', linestyle='dotted', label='Baseline')
        legend()
        gca().set(xlabel='Time [ms]', ylabel='Rates [kHz]', xlim=(0,max(time)))

    return time, rate_signal, peaks

def define_event(n_group, pop_rate_monitor, threshold_in_sd, plot_peaks_bool=False):
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
#    threshold_in_sd=4
    
    time_signal, pop_rate_signal, max_peak_indexes = find_events(n_group, pop_rate_monitor, threshold_in_sd, plot_peaks=plot_peaks_bool)
    baseline = mean(pop_rate_signal)
    event_props = {}
    simulation_props = {}
    simulation_props['Total_time'] = time_signal
    simulation_props['Total_signal'] = pop_rate_signal
    simulation_props['Max_peak_index'] = max_peak_indexes
    simulation_props['Baseline'] = baseline
#    figure()
    for i, index in enumerate(max_peak_indexes):
        event_props[i+1] = {}
        
        start_event_approx = int(index - 50/dt)
        end_event_appox = int(index + 50/dt)
        event_ranged = pop_rate_signal[start_event_approx:end_event_appox]
        time_ranged = time_signal[start_event_approx:end_event_appox]
        peaks, prop = signal.find_peaks(event_ranged, height = baseline, distance = 3/dt)
        
        index_peak_max = np.where(time_ranged[peaks] == time_ranged[index-start_event_approx])[0][0]
        find_first_peak_index = where((diff(time_ranged[peaks[:index_peak_max]]) < 6).astype(int) == 0)[0]
        find_last_peak_index = where((diff(time_ranged[peaks[index_peak_max:]]) < 6).astype(int) == 0)[0]
        
        if len(find_first_peak_index)==0: first_peak_index = 0
        else: first_peak_index = find_first_peak_index[-1] + 1
        if len(find_last_peak_index)==0: last_peak_index = index_peak_max
        else: last_peak_index = find_last_peak_index[0] + index_peak_max
        
#        first_peak_index = where((diff(time_ranged[peaks[:index_peak_max]]) < 6).astype(int) == 0)[0][-1] + 1
#        last_peak_index = where((diff(time_ranged[peaks[index_peak_max:]]) < 6).astype(int) == 0)[0][0] + index_peak_max 
        start_index = where(event_ranged[:peaks[first_peak_index]] < baseline)[0][-1] + 1
        end_index = where(event_ranged[peaks[last_peak_index]:] < baseline)[0][0] + peaks[last_peak_index] - 1

        time_event = (np.arange(0, len(time_ranged[start_index:end_index])) -  (peaks[index_peak_max]-start_index )) * dt
        event = event_ranged[start_index:end_index]
#        peaks = peaks[first_peak_index:last_peak_index+1] - start_index
        event_props[i+1]['Time_array'] = time_event
        event_props[i+1]['Signal_array'] = event
        event_props[i+1]['Duration']= len(time_event)*dt
        event_props[i+1]['Index_start']= start_event_approx + start_index
        event_props[i+1]['Index_end']= start_event_approx + end_index
                
    return event_props, simulation_props
        
#        plot(time_event, event, 'k')
#        axhline(y=baseline, c='gray', linestyle='dotted', label='Baseline')
#        scatter(time_event[peaks], event[peaks], c='orange')                
        
#        plot(time_ranged, event_ranged, 'k')
#        axhline(y=baseline, c='gray', linestyle='dotted', label='Baseline')
#        scatter(time_ranged[peaks], event_ranged[peaks], c='orange')        
#        scatter(time_ranged[index-start_event_approx], event_ranged[index-start_event_approx], c='r')
#        scatter(time_ranged[start_index], event_ranged[start_index], c='b')
#        scatter(time_ranged[end_index], event_ranged[end_index], c='b')

        
        
def prop_events(n_group, pop_rate_monitor, threshold_in_sd, plot_peaks_bool=False):
    '''
    Function to determine the properties of each event such as number of peaks, 
    instantaneous frequency, mean frequency, spikes per cycle, etc.
    
    '''
       
    events_dict, simulation_dict = define_event(n_group, pop_rate_monitor, threshold_in_sd, plot_peaks_bool)
    
    for i, event in enumerate(events_dict.keys()):
    
        time = events_dict[event]['Time_array']
        signal_event = events_dict[event]['Signal_array']
        peaks, props = signal.find_peaks(signal_event, height = baseline, distance = 3/dt, prominence=1, width=[1000,3000])
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
        

def ripple_prop(n_group, pop_rate_monitor, pop_spikes_monitor, threshold_in_sd, plot_peaks_bool=False):
    '''
    With the dictionary of all events (few ripples per event). 
    In this function we check what about the spikes in each ripple oscillation
    '''
    pop_spikes_monitor = M_E
    events_dict, simulation_dict = prop_events(n_group, pop_rate_monitor, threshold_in_sd, plot_peaks_bool)
    
    for event in events_dict.keys():
        start
        time = events_dict[event]['Time_array']
        signal_event = events_dict[event]['Signal_array']
        Oscilla
    
    




#rate_signal, peak_per_event = find_events(n_group=G_E, pop_rate_monitor=R_E, threshold_in_sd=4, plot_peaks=True)
#
#
#
#plot(R_E.t / ms, R_E.smooth_rate(width=1*ms)*len(G_E) / kHz, color='k')
#

