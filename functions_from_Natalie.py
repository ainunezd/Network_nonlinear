#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 10:37:09 2022

@author: schieferstein
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal 
import scipy

pi = np.pi
nan=np.nan
infty = np.infty

def f_oscillation_analysis_transient(signal, dt, fmax= 350, df=1, baseline_window=[50,150], target_window = [], Pthr=nan, \
                                     sig = None, expected_freq = 200, t_pad = 20, \
                                     fmin= 30, stationary=False, plot=False):
  '''
  INPUT
    signal: time series with signal to be analyzed
    dt: [ms] time step
    fmax: [Hz] waveletspectrogram will be computed for frequency range [0, fmax] with resolution df
    df: [Hz] resolution of wavelet spectrogram in frequency space
    baseline_window: [ms], list len 2, beginning and end of time window to use as baseline activity to determine threshold for power and amplitude
    target_window: [ms], list len 2, beginning and end of time window for which we want to analyse instantaneous frequency 
    Pthr: user-defined power threshold, if nan it will be determined based on the activity during the baseline_window
    sig: [ms] width of gaussian window to use for the waveletspectrogram, if None, sig will be defined based on the expected mean frequency.
              lower sig -> finer temporal resolution of waveletspec but coarser frequency resolution and vice versa
    expected_freq: [Hz] expected mean frequency of signal, used to infer optimal sig 
    t_pad: [ms] padding of baseline and target window to avoid boundary effects in the instantaneous frequency estimates
    fmin: [Hz] minimum for instantaneous frequency 
    stationary: bool, does the signal exhibit persistent oscillations? 
    plot: bool, produce result plot  
  
  OUTPUT
    wspec: 2D array (freq x time), instantaneous power in frequencies [0, fmax] over time
    wspec_extent: freq and time boundaries of wspec to be used for plotting with imshow
    instfreq: [Hz], instantaneous frequency (freq of maximal power, that is larger than fmin over time)
    instpower: power associated to the instfreq
    freq_onset_inst: onset frequency at beginning of ripple event
    instcoherence: instantaneous coherence associated to inst. freq.
    Pthr: power threshold
    ifreq_discr_t: time points associated to discrete inst. freq estimates 
    ifreq_discr: [Hz] discrete estimate of inst. frequency based on peak-to-peak distances in signal
  '''
  
  print('Oscillation analysis for transient stimulation...')    
  # --- initialization ---------------------------------------------------------
  wspec, wspec_extent, instfreq, instpower, instcoherence, freq_onset_inst = [],[],[],[],[],nan
  freq = np.arange(0, fmax, df)
  Tsim = signal.size*dt
  if not len(target_window): # take inst freq of the full simulation
    target_window = [baseline_window[1], Tsim]
  # --- pad the baseline and target window to avoid boundary effects
  target_window_pad = [np.max([0,target_window[0]-t_pad]), np.min([Tsim,target_window[1]+t_pad])]
  baseline_window_pad = [np.max([0,baseline_window[0]-t_pad]), np.min([Tsim,baseline_window[1]+t_pad])]

  # --- setting the wavelet window width ---------------------------------------------------------
  if not sig: # no temporal window width given
    sig = 1/expected_freq # period of the expected onset freq in sec, 3sig~ Width of Gaussian Window in Time
    sig_frq = np.sqrt(pi/2)/sig # was bigger (=248) before, maybe make sig smaller
  
  # --- determine significant power threshold ---------------------------------------------------------
  if not stationary and np.isnan(Pthr):
    # --- establish baseline power threshold
    start = int(baseline_window_pad[0]/dt)
    end = int(baseline_window_pad[1]/dt)
    signal0 = signal[start:end]
      
    wspec0 = waveletspec(signal0, freq, dt/1000, sig = sig, fmin=0)[0]
    wspec0_extent = (baseline_window_pad[0], baseline_window_pad[1], 0, fmax)
    
    # take the average power at 0 ONLY from the baseline window (which is in the middle of the signal I used)
    power0_mean = np.mean(wspec0[0,int((baseline_window[0]-baseline_window_pad[0])/dt):int((baseline_window[1]-baseline_window_pad[0])/dt)])
    power0_std = np.std(wspec0[0,int((baseline_window[0]-baseline_window_pad[0])/dt):int((baseline_window[1]-baseline_window_pad[0])/dt)])
    # define threshold power
    Pthr = power0_mean + 2*power0_std # maybe experiment here
    print('Pthr: ', Pthr)
  else: 
    wspec0 = []
    
  # --- actual analysis ---------------------------------------------------------
  # pad to avoid boundary effects
  start = int(target_window[0]/dt)
  end = int(target_window[1]/dt)
  start_pad = int(target_window_pad[0]/dt)
  end_pad = int(target_window_pad[1]/dt)
  
  # --- discrete
  if not stationary:
    ampl_min = np.mean(signal[int(baseline_window[0]/dt):int(baseline_window[1]/dt)]) \
            + 4*np.std(signal[int(baseline_window[0]/dt):int(baseline_window[1]/dt)])
  else:
    ampl_min = np.mean(signal)
  ifreq_discr_t, ifreq_discr = get_instfreq_discrete(signal[start : end], dt, freq_av=expected_freq, ampl_min=ampl_min)
  ifreq_discr_t += target_window[0]
    
  # --- continuous
  signal_target_pad = signal[start_pad : end_pad]
  
  wspec, instfreq, instpower = waveletspec(signal_target_pad, freq, dt/1000, sig = sig, fmin=fmin)
  # keep only the target window
  wspec_extent = (target_window[0], target_window[1], 0, fmax)
  wspec, instfreq, instpower \
  = wspec[:,start-start_pad:end-start_pad], instfreq[start-start_pad:end-start_pad], \
    instpower[start-start_pad:end-start_pad]
  
  if plot:
      t = np.arange(signal.size)*dt
      t_target = np.arange(target_window[0], target_window[1], dt)
      fig, ax = plt.subplots(2, figsize=(10,6), sharex=True)
      cbar_ax = fig.add_axes([0.92, 0.55, 0.025, 0.3])
      norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max(wspec))
      if len(wspec0):
        ax[0].imshow(wspec0, extent = wspec0_extent, origin='lower', aspect='auto', norm=norm)
      im = ax[0].imshow(wspec, extent = wspec_extent, origin='lower', aspect='auto', norm=norm)
      ax[0].plot(ifreq_discr_t, ifreq_discr, 'wo', label='inst.freq. (discrete)')
      ax[0].autoscale(False)
      # identify snippets of significant inst freq (to avoid that continuous line is plotted through the gaps)
      ix = np.where(instpower>=Pthr)[0]
      jumps = np.where(np.diff(ix)>1)[0]
      jumps = np.concatenate([[-1], jumps, [ix.size-1]])
      for j in range(jumps.size-1):
        if not j:
          ax[0].plot(t_target[ix[jumps[j]+1] : ix[jumps[j+1]]+1], instfreq[ix[jumps[j]+1] : ix[jumps[j+1]]+1], 'w', label='inst.freq. (continuous)')
        else:
                    ax[0].plot(t_target[ix[jumps[j]+1] : ix[jumps[j+1]]+1], instfreq[ix[jumps[j]+1] : ix[jumps[j+1]]+1], 'w')
      cb = fig.colorbar(im, cax=cbar_ax, label='power [signal-unit*sec]')
      cb.ax.plot([0,1e5],[Pthr]*2, 'r', lw=2)
      ax[0].legend(loc='best')
      ax[1].plot(t, signal)  
      ax[1].set_xlabel('time [ms]')
      ax[0].set_ylabel('freq [Hz]')
      ax[1].set_ylabel('signal')

  instcoherence = np.sqrt(instpower/wspec[0,:])
            
  print('[done]')
  return wspec, wspec_extent, instfreq, instpower, freq_onset_inst, instcoherence, Pthr, ifreq_discr_t, ifreq_discr


def getVtraces(Vrec, Vthr=-52, volt_min=None, volt_max=None, vbins=50, density=False, dv=None):
  ''' make histogram of voltages over time
  Vrec: [mV, but without Brian unit] record of all membrane potentials, axis 0: time, 1: neurons
  Vthr: [mV] spike threshold 
  volt_min, volt_max: [mV] minimum and maximum for voltage histogram 
  vbins: number of bins for voltage histogram 
  density: bool, whether to show histogram normalized as density 
  dv: [mV] resolution of voltage histogram
  
  OUTPUT
    Vdistr: 2D array (volt x time), histogram of membrane potentials over time
    vbinedges: [mV], binedges of voltage histogram, used for plotting as shown below
  '''
  print('Extracting voltage histogram...', end="")
  if not volt_min:
    volt_min = np.floor(np.min(Vrec.flatten()))-1 
  if not volt_max:
    volt_max = np.ceil(np.max(Vrec.flatten())) +1
    if volt_max<Vthr:
      volt_max=Vthr+2
  if dv:
    vbinedges = np.arange(volt_min, volt_max+dv, dv)
    vbins=len(vbinedges)-1
  else:
    vbinedges = np.linspace(volt_min, volt_max, vbins+1, endpoint=True)
  Vdistr = np.zeros((vbins,Vrec.shape[0]))
  for t in range(Vrec.shape[0]):
    volt = Vrec[t,:] # voltages in all history/neuron bins
    Vdistr[:,t] = np.histogram(volt,  vbinedges, density=density)[0]
  Vdistr[np.isnan(Vdistr)] = 0
  print('[done]')
  return Vdistr, vbinedges


def plot_vhist(ax, ax_cb, Vthr, Vdistr, vbinedges, v_recordwindow, t0=0, \
               show_colorbar=True, cbar_label='$p(V,t)$', cbar_max_relative=3, cbar_max=None):
  '''
  just to demonstrate how voltage histogram can be plotted 
  v_recordwindow = [begining of recording time of voltages, end of recording time of voltages] ( in ms )
  '''
  if not cbar_max:
    cbar_max = np.max(Vdistr)*cbar_max_relative
  im = ax.imshow(Vdistr, origin='lower', extent=(v_recordwindow[0]-t0, v_recordwindow[-1]-t0, vbinedges[0], vbinedges[-1]), \
                 norm=matplotlib.colors.PowerNorm(gamma=.2, vmax=cbar_max), aspect='auto', cmap='binary', interpolation=None) 
  gridline(Vthr, ax, 'y')
  ax.set_ylabel('V [mV]')
  ax.set_xlim([v_recordwindow[0]-t0, v_recordwindow[-1]-t0])
  
  if show_colorbar:
    if np.max(Vdistr) > cbar_max:
      plt.colorbar(im, cax=ax_cb,  extend='max', ticks=np.round(np.floor(np.linspace(0, cbar_max, 3)*10)/10, decimals=1)).set_label( label=cbar_label, labelpad=-.1)
    else:
      plt.colorbar(im, cax=ax_cb, ticks=np.round(np.floor(np.linspace(0, cbar_max, 3)*10)/10, decimals=1)).set_label(label=cbar_label, labelpad=-.1)
  return ax, ax_cb

#%% helper functions called above 
  
def gabor(x, w, sig= 1/np.sqrt(2), x0=0):
    '''
    sig: 
    w: frequency (beachte: 2pi faktor ist hier im gabor wavelet enthalten)
    '''    
    return 1/(sig*np.sqrt(2*pi))*np.exp(-(x-x0)**2/(2*sig**2))*np.exp(-1j*2*pi*w*(x-x0))  
  
def waveletspec(signal, freq, dt, wavelet=gabor, sig=None, fmin=30):
    ''' creates wavelet spectrogram (cf. Donoso IFA paper)
    signal: signal to be analysed
    freq: [Hz] array of frequencies to be tested
    dt: [sec] time step of discrete signal measurements t (in seconds)
    wavelet: so far just gabor used, has to take arguments time, freq, center x0
    fmin: [Hz] for instantaneous freq, only look at freqs > fmin
    '''
    if not sig: # no temporal window width given
      expPer = 1/np.mean(freq) # expected period of the signal
      sig = expPer # choose default time window such that it contains ~3 cycles of the signal: 3sig~ Width of Gaussian Window in Time
      sig_frq = np.sqrt(pi/2)/sig # just for reference: the std in freq domain (note: was bigger (=248) before, maybe make sig smaller?)
    t = np.arange(signal.size)*dt
    t0 = np.mean(t) # center wavelet array w.r.t time array
    F = np.zeros((freq.size, signal.size))
    for i in range(freq.size):
        w = freq[i]
        g = wavelet(t, w, sig=sig, x0=t0)
        F[i,:] = np.abs(scipy.signal.fftconvolve(signal,g,mode='same')*dt)
    ifmin = np.where(freq>=fmin)[0][0] # index of first frequency point larger than fmin
    imax = np.argmax(F[ifmin:,:], axis=0) # at each point in time, find frequency with maximal power
    instfreq = freq[ifmin+imax] # instantaneous frequency estimate, consider only freqs > fmin
    instpower = F[ifmin+imax,np.arange(0,F.shape[1])]
#    plt.figure()
#    plt.imshow(F, origin='lower', aspect='auto', extent=[t[0], t[-1], freq[0], freq[-1]])
#    plt.plot(t, instfreq, 'w')
    return F, instfreq, instpower

def get_instfreq_discrete(signal, dt, freq_av=180, ampl_min=0):
  '''
  dt: [ms]
  freq_av: [Hz] initial guess for frequencies that we are looking for
  ampl_min: minimal signal amplitude to be considered a significant peak
  '''
  print('InstFreq discrete...', end='')
  T = len(signal)*dt 
  mindiff = 1000/(freq_av+200)/dt # minimal distance between peaks
  maxpeaks = (freq_av+100)*T/1000
  period = 1000/freq_av # expected period in ms  #period = 1/self.freq_av*1000 
  hw = period/dt/10
  
  peak_idx = findPeaks(signal, maxpeaks = maxpeaks, minabs=ampl_min, mindiff = mindiff, halfwidth = hw)[0] 

  ifreq_discr_t = (peak_idx[:-1]+np.diff(peak_idx)/2)*dt # mid points between signal peaks
  ifreq_discr = 1/(np.diff(peak_idx)*dt/1000) #  inst freq as 1/peak distance
  print('[done]')
  return ifreq_discr_t, ifreq_discr

def findPeaks(f, maxpeaks=1e10, minabs = -1e10, mindiff = 2, halfwidth = 1):
    ''' finds local maxima of array f (excluding boundaries)
    maxpeaks: maximal number of peaks to be found
    minabs: minimal absolute value of peaks
    mindiff: minimal distance between peaks (in step units (dimensionless)), if 2 peaks are too close, just the latter one is included
    halfwidth: minimal halfwidth of one peak (in step units)
    '''
    sortidx = np.flipud(np.argsort(f))
    peak_idx = np.array([]).astype(int)
    peakcounter = 0
    hw = int(halfwidth)
    if hw> mindiff/2:
      raise ValueError('Halfwidth cannot be bigger than half the minimal peak distance!')
    for i in sortidx:
      if (i-1>=0) and (i+1<f.size): # exclude boundary values
        if f[i]>=np.max(np.append(f[i-hw:i+hw+1], minabs)):
#        if f[i]>=np.max((f[i-1], f[i+1], minabs)): # local maximum, >= also possible, possible extension to the function to choose this
          peak_idx = np.append(peak_idx,i)
          peakcounter += 1
          if peakcounter > maxpeaks-1:
            break
    if peakcounter:
      peak_idx = np.sort(peak_idx)
      too_close = np.where(np.diff(peak_idx) < mindiff)[0]
      keep = np.array(list(set(np.arange(peak_idx.size)) - set(too_close) - set(too_close+1))).astype(int) # at first take out BOTH peaks that are too close to each other
      # kick out peaks that are too close to each other, keep larger one:
#      print(keep)
      for i in too_close:
        ix_keep = i + np.argmax([ f[peak_idx[i]], f[peak_idx[i+1]] ])
        if ix_keep not in keep:
          keep = np.append(keep, ix_keep)
#        print(keep)
      peak_idx = peak_idx[keep]
      peak_idx = np.sort(peak_idx)
      if len(np.unique(peak_idx)) != len(peak_idx):
        raise ValueError('some peaks still counted twice!')
      peakcounter = peak_idx.size
    else: 
      print('No peaks detected!')
    return peak_idx, peakcounter
  
def gridline(coords, ax, axis='y', zorder=-10):
  if np.isscalar(coords):
    coords = list([coords])
  for c in coords:
    if axis=='y':
      if type(ax) in [list, np.ndarray]:
        for axi in ax:
          axi.axhline(c, lw=.5, linestyle=':', color='gray', zorder=zorder)
      else:
        ax.axhline(c, lw=.5, linestyle=':', color='gray', zorder=zorder)
    else:
      if type(ax) in [list, np.ndarray]:
        for axi in ax:
          axi.axvline(c, lw=.5, linestyle=':', color='gray', zorder=zorder)
      else:
        ax.axvline(c, lw=.5, linestyle=':', color='gray', zorder=zorder)
  return

#%% Example usage 
  
# create test signal
#dt = 0.01
#f = 200
#t = np.arange(0, 500, dt)
#signal = 50*(1 + np.sin(2*pi*f*t/1000)) # sine wave 
#baseline_window = [200, 300]
#signal[int(baseline_window[0]/dt): int(baseline_window[1]/dt)] = 10 # add baseline period
#signal[40000: 45000] = 10 # add another baseline period
#signal = signal + np.random.normal(scale=5, size=t.size) # add noise
#signal[signal<0] = 0

#wspec, wspec_extent, instfreq, instpower, freq_onset_inst, instcoherence, Pthr, ifreq_discr_t, ifreq_discr\
#= f_oscillation_analysis_transient(signal, dt, baseline_window=baseline_window, target_window = [t[0], t[-1]], \
#                                   expected_freq = f, fmin=70, plot=True)

#wspec, wspec_extent, instfreq, instpower, freq_onset_inst, instcoherence, Pthr, ifreq_discr_t, ifreq_discr\
#= f_oscillation_analysis_transient(rate_signal, dt=0.001, baseline_window=[0, 400], target_window = [3920, 4000], \
#                                   expected_freq = 200, fmin=70, plot=True)