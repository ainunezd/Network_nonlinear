#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 12:56:26 2022

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
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from sklearn.svm import SVR


from Network_model_2_two_populations import Network_model_2
from Populations_analysis import create_dataframe_all_long_networks, modify_population_dataframes


path_to_save_figures = '/home/ana_nunez/Documents/BCCN_berlin/Master_thesis/Plots/refractory_exploration/'
path_networks = '/home/ana_nunez/Documents/BCCN_berlin/Master_thesis/'

dt=0.01
cm=2.54

create_dataframe_all_long_networks(net_name = 'long_50000_network_tauax_change_distrib_t_ref_ds_', networks_names =['1_5ms_3', '2ms_3', '2_5ms_3', '3ms_3', '3_5ms_3'], path_to_save_figures=path_to_save_figures)
modify_population_dataframes(general_name_networks = 'long_50000_network_tauax_change_distrib_t_ref_ds_', networks_names =['1_5ms_3', '2ms_3', '2_5ms_3', '3ms_3', '3_5ms_3'], path_to_save_figures=path_to_save_figures)
df_exc_tref_ds = pd.read_pickle(path_to_save_figures + 'Properties_exitatory_ALL_events_selected_long_50000_network_tauax_change_distrib_t_ref_ds_.pkl')

#create_dataframe_all_long_networks(net_name = 'long_50000_network_tauax_change_distrib_one_neuron_tRefEx_', networks_names =['2ms_3', '2_5ms_3', '3ms_3'], path_to_save_figures=path_to_save_figures)
#modify_population_dataframes(general_name_networks = 'long_50000_network_tauax_change_distrib_one_neuron_tRefEx_', networks_names =['2ms_3', '2_5ms_3', '3ms_3'], path_to_save_figures=path_to_save_figures)
df_exc_tRefEx = pd.read_pickle(path_to_save_figures + 'Properties_exitatory_ALL_events_selected_long_50000_network_tauax_change_distrib_one_neuron_tRefEx_.pkl')


def plot_events_statistics_refrac_exploration(population='Excitatory', df_population=df_exc_tref_ds, net_name = 'long_50000_network_tauax_change_distrib_t_ref_ds_', 
                                              networks_names =['1_5ms_3', '2ms_3', '2_5ms_3', '3ms_3', '3_5ms_3']):
    '''
    Funtion to compare the different simulations and values for tref_ds 1.5 2 and 2.5 ms to observe differences
    '''
#    if population=='Excitatory': df_population = df_exc_tref_ds
#    elif population=='Inhibitory': df_population = pd.read_pickle(path_to_save_figures + name_dataframe_inh)
#    else: return None
    
    pdf_file_name = f'All_events_histograms_overview_{population}_' + net_name
    with PdfPages(path_to_save_figures + pdf_file_name + '.pdf') as pdf:        
        fig, ax = plt.subplots(len(networks_names), 4, figsize=(21/cm, 15/cm))
        for j, net_value in enumerate(networks_names):
            print(net_value)
            df_net = df_population[df_population.Name_network == net_name+net_value]
            values_duration = df_net['Duration'].values
            values_numpeaks = df_net['Num_peaks'].values
            values_freq = df_net['Mean_frequency'].values
            values_maxpop_rate = np.zeros(len(values_duration))
            for k, index in enumerate(df_net.index):
                values_maxpop_rate[k] = np.max(df_net.loc[index]['Peaks_heights'])
        
                
            ax[j,0].hist(values_duration, bins=50, histtype='step', color='k')
            ax[j,0].set(xlim=(0,120))
            ax[j,1].hist(values_numpeaks, bins=50, histtype='step', color='k')
            ax[j,1].set(xlim=(0,25))
            ax[j,2].hist(values_maxpop_rate, bins=50, histtype='step', color='k')
            ax[j,2].set(xlim=(0,155))
            ax[j,3].hist(values_freq, bins=50, histtype='step', color='k')
            ax[j,3].set(xlim=(185,220))
            ax[j,3].axvline(x=np.mean(values_freq), color='gray', linestyle='--')
            print(f'Mean frequency: {np.round(np.mean(values_freq),2)}')
        ax[j,0].set(xlabel='Duration of event [ms]')
        ax[j,1].set(xlabel='Peaks/event')
        ax[j,2].set(xlabel='Max peak height [kHz]')
        ax[j,3].set(xlabel='Mean frequency [Hz]')
        
        fig.tight_layout()
        pdf.savefig(fig) 
        
def plot_discrete_frequencies_refrac_exploration(population='Excitatory', df_population=df_exc_tref_ds, net_name = 'long_50000_network_tauax_change_distrib_t_ref_ds_', 
                                                 networks_names =['1_5ms_3', '2ms_3', '2_5ms_3', '3ms_3', '3_5ms_3'], plot_type='plot_all'):
    '''
    Function to plot and analyse the instantaneous frequency
    type_population can be:
        Excitatory 
        Inhibitory 
    and that would determine which dataframe to load.
    plot_type could be:
        scatter_all
        plot_all
        plot_regression
    '''
    pdf_file_name = f'All_events_frequencies_{population}_{plot_type}_' + net_name
    with PdfPages(path_to_save_figures + pdf_file_name + '.pdf') as pdf:        
        fig, ax = plt.subplots(1, len(networks_names), figsize=(21/cm, 10/cm), sharex=True, sharey=True)
        for j, net_value in enumerate(networks_names):
            print(net_value)
            df_net = df_population[df_population.Name_network == net_name+net_value]
            num_events = len(df_net)
            colors = cmr.take_cmap_colors('cividis', num_events, return_fmt='hex') 
            times = np.array([])
            frequencies = np.array([])
            for k, event in enumerate(df_net.index):
                frequencies_discrete = df_net.loc[event]['Instant_frequency']
                time_values = df_net.loc[event]['Time_array']
                time_discrete = time_values[df_net.loc[event]['Indices_frequency']]
                times = np.append(times, time_discrete)
                frequencies = np.append(frequencies, frequencies_discrete)
            
                if plot_type == 'scatter_all': ax[j].scatter(time_discrete, frequencies_discrete, color=colors[k], marker='.')
                if plot_type == 'plot_all': ax[j].plot(time_discrete, frequencies_discrete, color=colors[k], marker='.')
            
            if plot_type == 'plot_regression':     
                t = np.arange(min(times), max(times), 0.1)
                model = SVR(kernel='rbf')
                freq_predict = model.fit(times.reshape(-1, 1),frequencies).predict(t.reshape(-1, 1))
                regression = stats.linregress(times, frequencies)
                y = regression.slope*t +regression.intercept
                ax[j].plot(t,y, 'k')
                ax[j].text(10,200, f'Slope: {np.round(regression.slope,4)} [Hz/ms] \n p value: {np.round(regression.pvalue,4)}')
                ax[j].plot(t, freq_predict, 'gray')

            ax[j].grid(zorder= 0)
            
        ax[len(networks_names)//2].set(xlabel='Time w.r.t. highest peak [ms]')
        ax[0].set(ylabel='Frequency [Hz]')
        pdf.savefig(fig)  
            


 
    




