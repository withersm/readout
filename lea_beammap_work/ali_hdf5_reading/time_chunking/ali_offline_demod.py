import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.formatter.useoffset'] = False
import matplotlib.pyplot as plt
import scipy

from scipy.signal import sawtooth, square, savgol_filter
import pandas as pd
import glob as gl
import os
import cmath

from scipy.signal import sawtooth, square,find_peaks
from scipy import spatial
#import lambdafit as lf
from scipy.interpolate import CubicSpline,interp1d
import h5py

from tqdm import tqdm as tqdm_terminal
from tqdm.notebook import trange, tqdm_notebook
from scipy.signal.windows import hann

from scipy.fft import fft, ifft, fftfreq
from copy import deepcopy
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import curve_fit

def test_import():
    print('Package import working.')
    return 0

#Functions for generating fake data
def generate_science_signal(sig_type = 'sin', n_analog = 400000, freq=0.5, phase=0, envelope = 0, t_length=1, plot = True):

    t = np.linspace(0,t_length,n_analog*t_length)
    
    if sig_type == 'sin':
        sig = np.sin(2*np.pi*freq*t+phase)*np.exp(-envelope*t)
    elif sig_type == 'cos':
        sig = np.cos(2*np.pi*freq*t+phase)*np.exp(-envelope*t)
    elif sig_type == 'square':
        sig = square(2*np.pi*freq*t+phase)*np.exp(-envelope*t)
    elif sig_type == 'sawtooth':
        sig = sawtooth(2*np.pi*freq*t+phase)*np.exp(-envelope*t)        
    
    if plot == True:
        plt.plot(t, sig)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (arb.)')
        plt.title('Example Science Signal')
        plt.show()
        
    return t, sig

def scale_science_signal(t, signal, scale_factor = 0.1, plot=True):
    sig = signal*scale_factor
    
    if plot == True:
        plt.plot(t,sig)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (arb)')
        plt.title('Scaled Example Science Signal')
        plt.show()
        
    return t, sig

def apply_gaussian_noise(t, signal, noise_scale = 0.01, plot=True):
    white_noise = noise_scale * np.random.normal(0,1,len(t))
    
    sig = signal+white_noise
    
    if plot == True:
        plt.plot(t,sig)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (arb)')
        plt.title('Example Science Signal with White Noise')
        plt.show()
    
    return t, sig

def generate_flux_ramp(t, n_Phi0=4, f_sawtooth=40, plot = True, plot_len = None):
    flux_ramp = n_Phi0*(0.5+0.5*sawtooth(t*2*np.pi*f_sawtooth)) #generate with the same number of points as the science signal so they can be modulated together
    
    if plot == True:
        plt.plot(t[0:plot_len],flux_ramp[0:plot_len])
        plt.xlabel('Time (s)')
        plt.ylabel('$\Phi_0$')
        plt.title('Flux Ramp')
        plt.show()
    
    return t, flux_ramp

#Functions for simulating flux ramp modulation
def sinusoidal_modulation(t, sig, flux_ramp, phase_offset = 0, f0 = 4.5e9, fpp = 150e3, plot = True, plot_len = None):
    squid_resp_ramp_and_sig = f0 + fpp/2 * np.sin(2*np.pi*(flux_ramp - sig) - phase_offset) #squid response to flux ramp and science signal
    squid_resp_ramp_only = f0 + fpp/2 * np.sin(2*np.pi*(flux_ramp) - phase_offset) #squid response to flux ramp only (for baseline purposes)
    
    if plot == True:
        plt.plot(t[0:plot_len]*1e3,squid_resp_ramp_and_sig[0:plot_len]/1e9,label='SQUID Response; Signal')
        plt.plot(t[0:plot_len]*1e3,squid_resp_ramp_only[0:plot_len]/1e9,label='SQUID Response; No Signal')
        plt.xlabel('Time (ms)')
        plt.ylabel('Frequency (GHz)')
        plt.title('Small Segment of Flux Ramp Modulated Signal')
        plt.legend(loc='upper right')
        
    return t, squid_resp_ramp_and_sig, squid_resp_ramp_only

def realistic_squid_modulation(t, sig, flux_ramp, phase_offset = 0, f0 = 4.5e9, fpp = 150e3 , squid_lambda = 0.3, plot = True, plot_len = None):
    Phis = np.linspace(0,1,50)
    f0s = lf.f0ofphie(phie=2*np.pi*Phis, f2=f0, P = fpp, lamb=squid_lambda)
    f0s = f0s - np.max(f0s) + f0 + fpp/2 # center it on f0
    lookup_curve = CubicSpline(x=Phis, y=f0s, bc_type='periodic')
    squid_resp_ramp_and_sig = lookup_curve((flux_ramp - sig) - phase_offset/(2*np.pi)) #yes signal
    squid_resp_ramp_only = lookup_curve((flux_ramp) - phase_offset/(2*np.pi)) #no signal; just SQUID response to flux ramp

    if plot == True:
        plt.plot(t[0:plot_len]*1e3,squid_resp_ramp_and_sig[0:plot_len]/1e9,label='SQUID Response; Signal')
        plt.plot(t[0:plot_len]*1e3,squid_resp_ramp_only[0:plot_len]/1e9,label='SQUID Response; No Signal')
        plt.xlabel('Time (ms)')
        plt.ylabel('Frequency (GHz)')
        plt.title('Small Segment of Flux Ramp Modulated Signal')
        plt.legend(loc='upper right')
    
    return t, squid_resp_ramp_and_sig, squid_resp_ramp_only

def sample_squid(t, sig, flux_ramp, sample_rate = 4000, plot = True, plot_len = None):
    division_factor = int(len(t) // (sample_rate*t[len(t)-1]))
    #print(division_factor)
    
    sampled_t = t[0:len(t):division_factor]
    #print(len(sampled_t))
    sampled_signal = sig[0:len(sig):division_factor]
    #print(len(sampled_signal))
    
    if plot == True:
        fig, ax1 = plt.subplots(1)
        ax2 = plt.twinx(ax1)
        
        if plot_len != None:
            ax1.plot(t[0:plot_len*division_factor],sig[0:plot_len*division_factor])
            ax1.plot(sampled_t[0:plot_len],sampled_signal[0:plot_len],'.')
            ax2.plot(t[0:plot_len*division_factor],flux_ramp[0:plot_len*division_factor],'r-')
            plt.show()
        else:
            ax1.plot(t[0:plot_len],sig[0:plot_len])
            ax1.plot(sampled_t[0:plot_len],sampled_signal[0:plot_len],'.')
            ax2.plot(t[0:plot_len],flux_ramp[0:plot_len],'r-')
            plt.show()
        
    return sampled_t, sampled_signal
    
    
        
#Functions for reading, processing, and demodulating real data
def read_data(filename,channels='all',start_channel=0,stop_channel=1000):
    if channels == 'all':
        file = h5py.File(filename, 'r')
        adc_i = np.array(file['time_ordered_data']['adc_i'])
        adc_i = np.delete(adc_i, slice(0,22), 0)
        adc_q = file['time_ordered_data']['adc_q']
        adc_q = np.delete(adc_q, slice(0,22), 0)
        t = np.array(file['time_ordered_data']['timestamp'])  
    elif channels == 'some':
        start_channel += 23 #eliminate the first 23 empty channels in hdf5 -> makes channel numbering match resonator numbering
        stop_channel += 23 + 1 #eliminate the first 23 empty channels in hdf5 -> makes channel numbering match resonator numbering; +1 forces python to include the stop_channel
        file = h5py.File(filename, 'r')
        adc_i = np.array(file['time_ordered_data']['adc_i'][start_channel:stop_channel]) 
        adc_q = np.array(file['time_ordered_data']['adc_q'][start_channel:stop_channel]) 
        t = np.array(file['time_ordered_data']['timestamp'])  
    
    return t, adc_i, adc_q

def read_data_live(filename,channels):
    channels += 22 #elimiate the first 23 empty channels in hdf5
    print(channels)
    file = h5py.File(filename, 'r', libver='latest', swmr=True)
    adc_i = np.array([file['time_ordered_data']['adc_i'][channel,:] for channel in channels])
    #adc_i = np.array(file['time_ordered_data']['adc_i'][channels]) 
    adc_q = np.array([file['time_ordered_data']['adc_q'][channel,:] for channel in channels])
    #adc_q = np.array(file['time_ordered_data']['adc_q'][channels]) 
    t = np.array(file['time_ordered_data']['timestamp'])  

    return t, adc_i, adc_q

def read_eta(eta_file):
    data = pd.read_csv(eta_file,sep=',')
    freq = data['f0'].to_numpy(float)
    eta = data['eta'].to_numpy(complex)
    return freq, eta


def apply_correction(eta_array, adc_i, adc_q):
    #make eta_array a column vector of complex values
    complex_data = adc_i + 1j*adc_q
    corrected_complex_data = eta_array * complex_data
    #print(corrected_complex_data[22:25])
    
    return corrected_complex_data

def extract_science_signal(corrected_complex_data):
    #demodulate each channel, return an array with a row of demod data for each channel
    pass


"""

def demodulate(t, sig, n_Phi0, f_sawtooth, plot = True, plot_len = None):
    #print(len(sig))
    #print(t[len(t)-1])
    chunksize = len(sig) / t[len(t)-1] / f_sawtooth
    n_chunks = int(len(t)//chunksize)
    
    #print(n_chunks)

    #sig -= (max(sig)+min(sig))/2
    
    #print(2*np.pi*n_Phi0*f_sawtooth)
    
    slow_t = np.full(shape=n_chunks, dtype=float, fill_value=np.nan)
    slow_TOD = np.full(shape=n_chunks, dtype=float, fill_value=np.nan)
    for ichunk in range(n_chunks):
        #print(ichunk)
        """"""
        if ichunk == 10:#ichunk == 4 or ichunk ==5 or ichunk == 9 or ichunk == 10:
            print('at 4')
            continue
        """"""
        start = int(ichunk*chunksize)
        stop = int((ichunk+1)*chunksize)
        #print(len(sampled_signal[start:stop]))
        #print(stop)
        num = np.sum(sig[start:stop]*np.sin(2*np.pi*n_Phi0*f_sawtooth*(t[start:stop]-t[start])))
        den = np.sum(sig[start:stop]*np.cos(2*np.pi*n_Phi0*f_sawtooth*(t[start:stop]-t[start])))
        slow_TOD[ichunk] = np.arctan2(num, den)
        #print(slow_TOD[ichunk])
        slow_t[ichunk] = t[(start+stop)//2]
        #print(ichunk)
        #print(slow_t[ichunk])
        #print(slow_TOD)
    
    
    #slow_t = slow_t[~np.isnan(slow_TOD)]
    #slow_TOD = np.unwrap(slow_TOD[~np.isnan(slow_TOD)])
    slow_TOD /= 2*np.pi # convert to Phi0
    #slow_TOD -= np.average(slow_TOD) # DC subtract
    #print(np.isnan(slow_TOD))
    
    
    if plot == True:
        plt.plot(slow_t,slow_TOD,'.')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (arb.)')
        #plt.legend(loc='upper right')
        plt.title('Reconstructed Signal')
        plt.show()
        
    return slow_t, slow_TOD
"""


def demodulate(t, sig, n_Phi0, n, f_sawtooth, fs=512e6/(2**20)):
    t = np.arange(sig.shape[0])/fs
    period=1/f_sawtooth
    n_chunks=int(t[-1]/period) 
    chunksize_org=len(sig) / t[len(t)-1] / f_sawtooth
    #chunksize_org = len(t)/n_chunks
    chunksize = int(chunksize_org)
    chunksize_left=chunksize-2*n
    resets = np.arange(n_chunks)/f_sawtooth
    #eset_inds=np.array([find_nearest_idx(t,reset) for reset in resets])
    reset_inds=np.arange(n_chunks)*chunksize_org
    #reset_inds=np.rint(reset_inds)
    reset_inds=reset_inds.astype(int)
    inds_2d=np.mgrid[0:chunksize_left,0:n_chunks][0].T
    inds_2d=inds_2d.astype(int)
    inds_2d+=reset_inds.reshape((-1,1))
    inds_2d+=n
    t_2d=t[inds_2d]
    sig_2d=sig[inds_2d]
    
    """
    window = hann(sig_2d[1,:].size)[np.newaxis]
    print(window)
    window_T = window.T
    print(window_T)
    
    sig_2d=np.matmul(sig_2d,window_T)
    print(sig_2d)
    """
    t_resets = resets.reshape((-1,1))*np.ones(shape=t_2d.shape)
    t_2d = t_2d-t_resets
    
    num = np.sum(sig_2d*np.sin(2*np.pi*n_Phi0*f_sawtooth*(t_2d)),axis=1)
    den = np.sum(sig_2d*np.cos(2*np.pi*n_Phi0*f_sawtooth*(t_2d)),axis=1)
    
    slow_TOD = np.arctan2(num, den)
    slow_t = np.arange(n_chunks)/f_sawtooth+0.5/f_sawtooth
    slow_TOD /= 2*np.pi
    return slow_t, slow_TOD, reset_inds

   

    
    
    
    
"""

def demodulate(t, sig, n_Phi0, n, f_sawtooth, fs=512e6/(2**20),plot=False):
    #chunksize = len(sig) / t[len(t)-1] / f_sawtooth
    #n_chunks = int(len(t)//chunksize)
    #fs = 488.28125#1/np.nanmedian(np.diff(t))
    #fs=1/np.nanmedian(np.diff(t))
    t = np.arange(sig.shape[0])/fs
    period=1/f_sawtooth
    n_chunks=int(t[-1]/period)  
    slow_t = []
    slow_TOD = []
    for ichunk in range(n_chunks):
        t_start =period*ichunk+n/fs
        t_stop = period*(ichunk+1)-n/fs
        sig_chunk=sig[np.where((t>t_start)&(t<t_stop))]
        t_chunk=t[np.where((t>t_start)&(t<t_stop))]
        t_reset=ichunk/f_sawtooth
        if t_chunk.shape[0]==0:
            continue
        t_diff=np.diff(t_chunk)
        t_diff=np.insert(t_diff, 0, 0)
        #print (t_chunk.shape)
        #print (t_diff.shape)
        #t_chunk_sel=t_chunk[(t_diff >= 1./fs*0.8) & (t_diff <= 1./fs*1.2)]
        #sig_chunk_sel=sig_chunk[(t_diff >= 1./fs*0.8) & (t_diff <=1./fs*1.2)]
        if t_chunk.shape[0]<10:
            continue
        else:
            num = np.sum(sig_chunk*np.sin(2*np.pi*n_Phi0*f_sawtooth*(t_chunk-t_reset)))
            den = np.sum(sig_chunk*np.cos(2*np.pi*n_Phi0*f_sawtooth*(t_chunk-t_reset)))
            slow_TOD.append(np.arctan2(num, den))
            slow_t.append((t_start+t_stop)/2)
    
    
    #slow_t = slow_t[~np.isnan(slow_TOD)]
    #slow_TOD = np.unwrap(slow_TOD[~np.isnan(slow_TOD)])
    slow_t=np.array(slow_t)
    slow_TOD=np.array(slow_TOD)
    slow_TOD /= 2*np.pi # convert to Phi0
    #slow_TOD -= np.average(slow_TOD) # DC subtract
    #print(np.isnan(slow_TOD))
    
    
    if plot == True:
        plt.plot(slow_t,slow_TOD,'.')
        plt.vlines(ts_start,0,0.4)
        print (range(ichunk))
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (arb.)')
        #plt.legend(loc='upper right')
        plt.title('Reconstructed Signal')
        plt.show()
    
        
    return slow_t, slow_TOD

"""


"""
#inital before testing with real data
def demodulate(t, sig, n_Phi0, f_sawtooth, plot = True, plot_len = None):
    chunksize = len(sig) / t[len(t)-1] / f_sawtooth
    n_chunks = int(len(t)//chunksize)
    
    print(n_chunks)
    
    #print(2*np.pi*n_Phi0*f_sawtooth)
    
    slow_t = np.full(shape=n_chunks, dtype=float, fill_value=np.nan)
    slow_TOD = np.full(shape=n_chunks, dtype=float, fill_value=np.nan)
    for ichunk in range(n_chunks):
        start = int(ichunk*chunksize)
        stop = int((ichunk+1)*chunksize)
        #print(len(sampled_signal[start:stop]))
        #print(stop)
        num = np.sum(sig[start:stop]*np.sin(2*np.pi*n_Phi0*f_sawtooth*t[start:stop]))
        den = np.sum(sig[start:stop]*np.cos(2*np.pi*n_Phi0*f_sawtooth*t[start:stop]))
        slow_TOD[ichunk] = np.arctan2(num, den)
        #print(slow_TOD[ichunk])
        slow_t[ichunk] = t[(start+stop)//2]
    slow_TOD = np.unwrap(slow_TOD) # unwrap
    slow_TOD /= 2*np.pi # convert to Phi0
    slow_TOD -= np.average(slow_TOD) # DC subtract
    
    if plot == True:
        plt.plot(slow_t,slow_TOD,'.')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (arb.)')
        plt.legend(loc='upper right')
        plt.title('Reconstructed Signal')
        plt.show()
        
    return slow_t, slow_TOD
"""   


def demodulate_for_iv(t, sig, n_Phi0, n, f_sawtooth, fs=512e6/(2**20)):
    #chunksize = len(sig) / t[len(t)-1] / f_sawtooth
    #n_chunks = int(len(t)//chunksize)
    period=1/f_sawtooth
    n_chunks=int(t[-1]/period)  
    slow_t = []
    slow_TOD = []
    for ichunk in range(n_chunks):
        t_start =period*ichunk+n/fs
        t_stop = period*(ichunk+1)-n/fs
        sig_chunk=sig[np.where((t>t_start)&(t<t_stop))]
        t_chunk=t[np.where((t>t_start)&(t<t_stop))]
        if t_chunk.shape[0]==0:
            continue
        t_diff=np.diff(t_chunk)
        t_diff=np.insert(t_diff, 0, 0)
        #print (t_chunk.shape)
        #print (t_diff.shape)
        t_chunk_sel=t_chunk[(t_diff >= 1./fs*0.8) & (t_diff <= 1./fs*1.2)]
        sig_chunk_sel=sig_chunk[(t_diff >= 1./fs*0.8) & (t_diff <=1./fs*1.2)]
        if t_chunk_sel.shape[0]<10:
            continue
        else:
            num = np.sum(sig_chunk_sel*np.sin(2*np.pi*n_Phi0*f_sawtooth*(t_chunk_sel-t_chunk_sel[0])))
            den = np.sum(sig_chunk_sel*np.cos(2*np.pi*n_Phi0*f_sawtooth*(t_chunk_sel-t_chunk_sel[0])))
            slow_TOD.append(np.arctan2(num, den))
            slow_t.append((t_start+t_stop)/2)
    
    
    #slow_t = slow_t[~np.isnan(slow_TOD)]
    #slow_TOD = np.unwrap(slow_TOD[~np.isnan(slow_TOD)])
    slow_t=np.array(slow_t)
    slow_TOD=np.array(slow_TOD)
    slow_TOD /= 2*np.pi # convert to Phi0
    #slow_TOD -= np.average(slow_TOD) # DC subtract
    #print(np.isnan(slow_TOD))
    
    """
    if plot == True:
        plt.plot(slow_t,slow_TOD,'.')
        plt.vlines(ts_start,0,0.4)
        print (range(ichunk))
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (arb.)')
        #plt.legend(loc='upper right')
        plt.title('Reconstructed Signal')
        plt.show()
    """
        
    return slow_t, slow_TOD



def linear_model(x,m,b):
    return m*x+b

def demodulate_with_fft(t,sig,start_index,f_fr,phase_units='rad',correct_phase_jumps=False,phase_jump_threshold=0,plot_demod = False,plot_demod_title=None,intermediate_plotting_limits=[None,None],plot_chunking_process = False,plot_fft = False,plot_fft_no_dc = False,plot_limited_fft = False,plot_fit = False,plot_vectors = False):
    print(f'shape sig: {sig.shape}')
    print(f'len t: {len(t)}')

    if intermediate_plotting_limits[0] == None:
        intermediate_plotting_limits[0] = t[0]
    if intermediate_plotting_limits[1] == None:
        intermediate_plotting_limits[1] = t[-1]
    
    
    if plot_demod == True:
        fig_demod, ax_demod = plt.subplots(1)
        ax_demod.set_ylim([-0.05,1])
            
        #establish array for storing phase
        phase_array = np.array([])
        

    
    
    if plot_chunking_process == True:
        #print(sig[ch])
        fig1, ax1 = plt.subplots(1)
        ax1.plot(t,sig,'.-')
        ax1.set_xlabel('$t$ (s)')
        ax1.set_ylabel('Resonator Position (arb.)')
        #ax1.set_title(f'Ch. {ch}')
        fig1.show()

    #begin at start of first fr
    t_fr_start = t[start_index:]
    t_fr_start = t_fr_start - t_fr_start[0]
    sig_fr_start = sig[start_index:]

    #fig_test, ax_test = plt.subplots(1)
    #ax_test.plot(t,sig)
    #ax_test.plot(t[start_index],sig[start_index],'*')

        
    if plot_chunking_process == True:
        ax1.plot(t_fr_start,sig_fr_start,'.-')
        ax1.set_xlabel('$t$ (s)')
        ax1.set_ylabel('Resonator Position (arb.)')
        #ax1.set_title(f'Ch. {ch}')

    #interpolate the data
    print(f'len t_fr_start: {len(t_fr_start)}')
    print(f'len sig_fr_start: {len(sig_fr_start)}')
    interpolation = interp1d(t_fr_start, sig_fr_start,fill_value='extrapolate')

        
    t_final = round((t_fr_start[-1])/(1/f_fr))*(1/f_fr) #make final time the latest time that fits an integer number of flux ramps; will interpolate to here
    t_start = t_fr_start[0]
    t_elapsed = t_final - t_start
    n_reset_periods = t_elapsed * f_fr #will be replaced when we have the fr reference
        
    t_interp = np.linspace(t_start, t_final, round(n_reset_periods)*1024) #interpolate so that every chunk has 1024 points; will make fft faster        
    sig_interp = interpolation(t_interp)

    
              
    if plot_chunking_process == True:
        fig2, ax2 = plt.subplots(1)
        ax2.plot(t_interp, sig_interp,'.')
        ax2.set_xlabel('$t$ (s)')
        ax2.set_ylabel('Resonator Position (arb.)')
        #ax2.set_title(f'Ch. {ch}')
        
    t_chunked = np.reshape(t_interp,(int(len(t_interp)/1024), 1024))
    sig_chunked = np.reshape(sig_interp,(int(len(sig_interp)/1024), 1024))
    sig_average = sig_chunked.mean(axis=1, keepdims=True)
    sig_chunked = sig_chunked - sig_average

    reset_indices_interp_space = [(row+1)*sig_chunked.shape[1] for row in range(len(sig_chunked))]

    reset_indices = [find_nearest_idx(sig, entry) for entry in reset_indices_interp_space]

    if plot_chunking_process == True:
        fig3, ax3 = plt.subplots(1)
        for chunk in range(len(t_chunked)):
            if np.median(t_chunked[chunk]) >= intermediate_plotting_limits[0] and np.median(t_chunked[chunk]) <= intermediate_plotting_limits[1]:
                ax3.plot(t_chunked[chunk],sig_chunked[chunk],'.-')
                ax3.set_xlabel('$t$ (s)')
                ax3.set_ylabel('Resonator Position (arb.)')
                #ax3.set_title(f'Ch. {ch}; Chunk {ch}; t_demod = {np.median(t_chunked[chunk])}')
                
    #reset_mask = np.append(np.zeros(50),np.ones(1024-50*2))
    #reset_mask = np.append(reset_mask,np.zeros(50))
      
    #sig_chunked = np.array([row*reset_mask for row in sig_chunked])
                
        
    t_increase = (t_chunked[0,-1] + np.median(np.diff(t_chunked[0])))
    t_new = t_chunked + t_increase
               
    t_zero_padded = np.hstack((t_chunked, t_new))        
    sig_zero_padded = np.hstack((sig_chunked, np.zeros(sig_chunked.shape)))
        
    if plot_chunking_process == True and t[0] >= intermediate_plotting_limits[0] and t[0] <= intermediate_plotting_limits[1]:
        fig4, ax4 = plt.subplots(1)
        for chunk in range(len(t_zero_padded)):
            #fft_fit = ifft(sig_fft) #only waste computation time on computing the fit if we actually want to plot it; otherwise we'll just use the fft above
            ax4.plot(t_zero_padded[chunk],sig_zero_padded[chunk],'.')
            ax4.set_xlabel('$t$ (s)')
            ax4.set_ylabel('Resonator Position (arb.)')
            #ax4.set_title(f'Ch. {ch}; Chunk {ch}; t_demod = {np.median(t_chunked[chunk])}')
                        
    sig_fft = fft(sig_zero_padded)   
    freq_fft = fftfreq(len(t_zero_padded[0]),np.median(np.diff(t_zero_padded[0])))
        
    if plot_fft == True:
            
        for chunk in range(len(t_zero_padded)):
            if np.median(t_chunked[chunk]) >= intermediate_plotting_limits[0] and np.median(t_chunked[chunk]) <= intermediate_plotting_limits[1]:
                fig5, ax5 = plt.subplots(1)
                ax5.stem(freq_fft,np.abs(sig_fft[chunk]))
                #ax5.set_title(f'Full FFT; Ch. {ch}; Chunk {chunk}; t_demod = {np.median(t_chunked[chunk])}')  
                ax5.set_xlabel('$f$ (Hz.)')
                ax5.set_ylabel('FFT Power (arb.)')
                
        
    #remove dc from fft:
    sig_fft_no_dc = deepcopy(sig_fft)
    sig_fft_no_dc[:,0] = 0
        
    if plot_fft_no_dc == True:
            
        for chunk in range(len(t_zero_padded)):
            if np.median(t_chunked[chunk]) >= intermediate_plotting_limits[0] and np.median(t_chunked[chunk]) <= intermediate_plotting_limits[1]:
                fig6, ax6 = plt.subplots(1)
                ax6.stem(freq_fft,np.abs(sig_fft_no_dc[chunk]))
                #ax6.set_title(f'No DC FFT; Ch. {ch}; Chunk {chunk}; t_demod = {np.median(t_chunked[chunk])}')  
                ax6.set_xlabel('$f$ (Hz.)')
                ax6.set_ylabel('FFT Power (arb.)')
        
        
    #keep largest fourier component
    sig_fft_reduced = deepcopy(sig_fft_no_dc)
        
    primary_bins = [np.argpartition(np.abs(row), -4)[-4:] for row in sig_fft_reduced] #identify the four greatest bins (two on each side of the fft)
    primary_bins = [np.sort(row) for row in primary_bins]
    mask = np.zeros_like(sig_fft_reduced)
    [np.put(mask[row],primary_bins[row],1) for row in range(len(mask))]
        
                       
    sig_fft_reduced = sig_fft_reduced * mask
        
    #
    if plot_limited_fft == True:
            
        for chunk in range(len(t_zero_padded)):
            if np.median(t_chunked[chunk]) >= intermediate_plotting_limits[0] and np.median(t_chunked[chunk]) <= intermediate_plotting_limits[1]:
                fig7, ax7 = plt.subplots(1)
                ax7.stem(freq_fft,np.abs(sig_fft_reduced[chunk]))  
                #ax7.set_title(f'Reduced FFT; Ch. {ch}; Chunk {chunk}; t_demod = {np.median(t_chunked[chunk])}')  
                ax7.set_xlabel('$f$ (Hz.)')
                ax7.set_ylabel('FFT Power (arb.)')
        
    if plot_fit == True:
           
        for chunk in range(len(t_zero_padded)):
            if np.median(t_chunked[chunk]) >= intermediate_plotting_limits[0] and np.median(t_chunked[chunk]) <= intermediate_plotting_limits[1]:
                limited_fit = ifft(sig_fft_reduced)
                fig8, ax8 = plt.subplots(1)
                ax8.plot(t_zero_padded[chunk], sig_zero_padded[chunk], '.')
                ax8.plot(t_zero_padded[chunk], limited_fit[chunk], '-')
                ax8.set_xlabel('$t$ (s)')
                ax8.set_ylabel('Resonator Position (arb.)')
                #ax8.set_title(f'Ch. {ch}; Chunk {chunk}; t_demod = {np.median(t_chunked[chunk])}')
                
        
    #find phase for each chunk          
    R1 = [np.real(sig_fft_reduced[row][primary_bins[row][0]]) if primary_bins[row][0] >= primary_bins[row][1] else np.real(sig_fft_reduced[row][primary_bins[row][1]]) for row in range(len(sig_fft_reduced))]
    R2 = [np.real(sig_fft_reduced[row][primary_bins[row][1]]) if primary_bins[row][0] >= primary_bins[row][1] else np.real(sig_fft_reduced[row][primary_bins[row][0]]) for row in range(len(sig_fft_reduced))]

    I1 = [np.imag(sig_fft_reduced[row][primary_bins[row][0]]) if primary_bins[row][0] >= primary_bins[row][1] else np.imag(sig_fft_reduced[row][primary_bins[row][1]]) for row in range(len(sig_fft_reduced))]
    I2 = [np.imag(sig_fft_reduced[row][primary_bins[row][1]]) if primary_bins[row][0] >= primary_bins[row][1] else np.imag(sig_fft_reduced[row][primary_bins[row][0]]) for row in range(len(sig_fft_reduced))]
        
        
                
        
    #note change here from unvectorized code: sets ange1 to 0 if Re=0 and Im=0
    angle1 = [np.angle(R1[row]+1j*I1[row]) if I1[row] > 0 else 2*np.pi + np.angle(R1[row]+1j*I1[row]) if I1[row] < 0 else 0 if I1[row] == 0 and R1[row] > 0 else np.pi if I1[row] == 0 and R1[row] < 0 else 0 for row in range(len(R1))]
    
    #note change here from unvectorized code: sets angle2 to 0 if Re=0 and Im=0
    angle2 = [np.angle(R2[row]+1j*I2[row]) if I2[row] > 0 else 2*np.pi + np.angle(R2[row]+1j*I2[row]) if I2[row] < 0 else 0 if I2[row] == 0 and R2[row] > 0 else np.pi if I2[row] == 0 and R2[row] < 0 else 0 for row in range(len(R2))]
        
        
    if plot_vectors == True:
            
        summed_vectors = [(R1[row]+I2[row])+1j*(I1[row]-R2[row]) if R2[row] > 0 and I2[row] > 0 and I1[row] < 0 else (R1[row]-I2[row])+1j*(I1[row]+R2[row]) if angle1[row] > angle2[row] else (R1[row]+I2[row])+1j*(I1[row]-R2[row]) if angle2[row] > angle1[row] else (R1[row]+R2[row])+1j*(I1[row]+I2[row]) for row in range(len(angle1))]
        
            
        for chunk in range(len(R1)):
            if np.median(t_chunked[chunk]) >= intermediate_plotting_limits[0] and np.median(t_chunked[chunk]) <= intermediate_plotting_limits[1]:
                fig9, ax9 = plt.subplots(1)
                ax9.set_aspect('equal', adjustable='box')
                vec1 = [0+1j*0, R1[chunk]+1j*I1[chunk]]
                vec2 = [0+1j*0, R2[chunk]+1j*I2[chunk]]
                summed = [0+1j*0, summed_vectors[chunk]]

                ax9.plot(np.real(vec1), np.imag(vec1),'-',label='vec1')
                ax9.plot(np.real(vec2), np.imag(vec2),'-',label='vec2')
                ax9.plot(np.real(summed), np.imag(summed),'-',label='summed')
                ax9.set_xlabel('Real')
                ax9.set_ylabel('Imaginary')
                #ax9.set_title(f'Ch. {ch}; Chunk {chunk}; t_demod = {np.median(t_chunked[chunk])}')
                ax9.legend()
        
        
    #note change here form unvectorized code: sets interpolated_phase to the phase of the sum of the two angles if they exactly align from the beginning
    interpolated_phase = [np.arctan2((I1[row]-R2[row]),(R1[row]+I2[row])) if R2[row] > 0 and I2[row] > 0 and I1[row] < 0 else np.arctan2((I1[row]+R2[row]),(R1[row]-I2[row])) if angle1[row] > angle2[row] else np.arctan2((I1[row]-R2[row]),(R1[row]+I2[row])) if angle2[row] > angle1[row] else np.arctan2((I1[row]+I2[row]),(R1[row]+R2[row])) for row in range(len(angle1))]
    interpolated_phase = np.unwrap(interpolated_phase)
        
    if phase_units == 'nPhi0':
        interpolated_phase = [entry / (2*np.pi) for entry in interpolated_phase]
        axis_label = 'Phase ($n_{\\Phi_0}$)'
    elif phase_units == 'deg':
        interpolated_phase = [entry * 180 / np.pi for entry in interpolated_phase]
        axis_label = 'Phase (deg.)'
    elif phase_units == 'rad':
        axis_label = 'Phase (rad.)'
    else:
        raise('Invalid units requested. Please use "nPhi0", "rad", or "deg".')
            
    demod_t = np.array([np.median(row) for row in t_chunked])
        
    if correct_phase_jumps == True:
        phase_array_diff = np.diff(interpolated_phase)
        discontinuity_index = np.argwhere(phase_array_diff > phase_jump_threshold)
        discontinuity_index = discontinuity_index.reshape(1,len(discontinuity_index))[0]
               
        def linear_model(x,m,b):
            return m*x+b
            
        if len(discontinuity_index) != 0:
                
                           
            phase_split = np.split(interpolated_phase, discontinuity_index+1)
            t_split = np.split(demod_t, discontinuity_index+1)

                
            phase_split_corrected = np.array([])
            t_split_corrected = np.array([])
            for i in range(len(phase_split)):
                    
                if len(phase_split[i]) >= 4:
                        

                    t_drop_final = t_split[i]
                        
                    phase_drop_final = phase_split[i]        
                    
                        

                    linear_fit = curve_fit(linear_model, t_drop_final, phase_drop_final, check_finite=False)

                    fitted_line = linear_model(t_drop_final,linear_fit[0][0],linear_fit[0][1])

                    phase_remove_accumulation = phase_drop_final - fitted_line
                        
                else:
                    t_drop_final = t_split[i]
                    phase_drop_final = phase_split[i]
                    phase_remove_accumulation = phase_split[i]
                        
                phase_split_corrected = np.append(phase_split_corrected, phase_remove_accumulation)
                t_split_corrected = np.append(t_split_corrected, t_drop_final)
                    
            demod_t = t_split_corrected
            interpolated_phase = phase_split_corrected
  
    #if chunk_count == 0:
    #    demod_data = np.array(interpolated_phase)            
    #else:
    #    demod_data = np.vstack([demod_data, np.array(interpolated_phase)])
            
      
                
    def linear_model(x,m,b):
        return m*x+b                    
   
  
        

    linear_fit = curve_fit(linear_model, demod_t, interpolated_phase, check_finite=False)

    fitted_line = linear_model(demod_t,linear_fit[0][0],linear_fit[0][1])

    phase_remove_accumulation = interpolated_phase - fitted_line
    
    
    
    if plot_demod == True:
        ax_demod.plot(demod_t, interpolated_phase-np.average(interpolated_phase)+0.1*chunk_count,'-')
        ax_demod.set_ylabel(axis_label)
        ax_demod.set_xlabel('$t$ (s)')            
        
    #chunk_count += 1
    
    return demod_t, phase_remove_accumulation, reset_indices#demod_data

def find_start_idx_internal_fr(t, sig, n_packets):

    fig1, ax1 = plt.subplots(1)
    ax1.plot(t, sig,'.-')
    
    
    data = sig #fix this later; made it easier to copy paste working code
    n_packets = int(n_packets)

    derivative = np.diff(data)
    ax1.plot(t[:-1],derivative)
    dd = np.diff(derivative)
    ddd = np.diff(dd)
    reset = np.argmax(dd[0:n_packets-1])
    ax1.plot(t[:-2],dd)
    #ax1.plot(t[:-3],ddd)

    #ax1.plot(t[reset],sig[reset],'*')
    #d=np.diff(data)
    #dd=np.diff(d)
    #reset=np.argmax(dd)#+2

    peaks = find_peaks(derivative[0:n_packets+int(n_packets/2)],height=0)[0]

    peak_widths = [np.abs(t[peak_i-1] - t[peak_i+1]) for peak_i in peaks]

    try:
        narrowest_peak = np.argmin(peak_widths)

        #reset_new = np.argwhere(t == t[peaks[narrowest_peak]])
        reset_new = np.argmax(dd[0:int(n_packets + n_packets / 2)])


        #print(peaks)
        #peaks_d = np.diff(peaks[0])
        
        #reset_new = np.argmin(peaks_d)
        #ax1.plot(t[peaks[0]],dd[peaks[0]],'o')
        ax1.plot(t[peaks], derivative[peaks],'o')
        ax1.plot(t[reset_new],sig[reset_new],'*')

        return reset_new

    except:
        return np.nan


def demodulate_with_fft_internal_fr(t, sig, n_packets, correct_phase_jumps = False, phase_jump_threshold=0.4):#, start_idx):


    #start by chunking blindly
    end_idx_blind = int(len(sig)/32) * 32
    

    t_blind_chunk = np.reshape(t[:end_idx_blind],(int(len(t)/32),32))
    data_blind_chunk = np.reshape(sig[:end_idx_blind],(int(len(sig)/32),32))

    fig, ax = plt.subplots(1)
    for i in range(len(t_blind_chunk)):
        ax.plot(t_blind_chunk[i], data_blind_chunk[i])

    data_blind_avg = np.average(data_blind_chunk,axis=0)


    fig_a, ax_a = plt.subplots(1)
    ax.plot(t_blind_chunk[0],data_blind_avg)


    
    
    
    
    fig1, ax1 = plt.subplots(1)
    ax1.plot(t, sig,'.-')
    
    
    data = sig #fix this later; made it easier to copy paste working code
    n_packets = int(n_packets)

    derivative = np.diff(data)
    ax1.plot(t[:-1],derivative)
    dd = np.diff(derivative)
    ddd = np.diff(dd)
    reset = np.argmax(dd[0:n_packets-1])
    ax1.plot(t[:-2],dd)
    #ax1.plot(t[:-3],ddd)

    ax1.plot(t[reset],sig[reset],'*')
    #d=np.diff(data)
    #dd=np.diff(d)
    #reset=np.argmax(dd)#+2

    peaks = find_peaks(derivative[0:n_packets-1],height=0.5)[0]

    peak_widths = [np.abs(t[peak_i-1] - t[peak_i+1]) for peak_i in peaks]

    narrowest_peak = np.argmin(peak_widths)

    reset_new = np.argwhere(t == t[peaks[narrowest_peak]])


    #print(peaks)
    #peaks_d = np.diff(peaks[0])
    
    #reset_new = np.argmin(peaks_d)
    #ax1.plot(t[peaks[0]],dd[peaks[0]],'o')
    ax1.plot(t[reset_new],sig[reset_new],'*')

    #data_drop = data[reset+1:]
    data_drop = data[reset:]
    #data_drop = data[start_idx:]
    end = int(len(data_drop)/n_packets)*n_packets
    #input(f'end: {end}')
    data_drop = data_drop[:end]

    #input(f'data drop: {len(data_drop)}')


    t_drop = t[reset:]
    t_drop = t_drop[:end]

    ax1.plot(t_drop,data_drop,'--')

    #input(f't drop: {len(t_drop)}')

    data_matrix = np.reshape(data_drop, (int(len(data_drop)/32),32))

    data_matrix = np.array([row - np.average(row) for row in data_matrix])
    
    t_chunked = np.reshape(t_drop,(int(len(t_drop)/32), 32))

    #input(t_chunked)

    #fig_chunked, ax_chunked = plt.subplots(1)
    #for i in range(len(data_matrix)):
    #    ax_chunked.plot(t_chunked[i],np.array(range(len(data_matrix[0,:])))+i*len(data_matrix[0,:]),data_matrix[i,:])
        
    zeros_to_append = np.zeros_like(data_matrix)

    data_matrix_zero_pad = np.hstack([data_matrix,zeros_to_append])

    t_increase = (t_chunked[0,-1] + np.median(np.diff(t_chunked[0])))
    t_new = t_chunked + t_increase
               
    t_zero_padded = np.hstack((t_chunked, t_new))  


    sig_fft = fft(data_matrix_zero_pad)   
    freq_fft = fftfreq(len(t_zero_padded[0]),np.median(np.diff(t_zero_padded[0])))

    sig_fft_no_dc = deepcopy(sig_fft)
    sig_fft_no_dc[:,0] = 0

    sig_fft_reduced = deepcopy(sig_fft_no_dc)
        
    primary_bins = [np.argpartition(np.abs(row), -4)[-4:] for row in sig_fft_reduced] #identify the four greatest bins (two on each side of the fft)
    primary_bins = [np.sort(row) for row in primary_bins]
    mask = np.zeros_like(sig_fft_reduced)
    [np.put(mask[row],primary_bins[row],1) for row in range(len(mask))]
        
                       
    sig_fft_reduced = sig_fft_reduced * mask

    R1 = [np.real(sig_fft_reduced[row][primary_bins[row][0]]) if primary_bins[row][0] >= primary_bins[row][1] else np.real(sig_fft_reduced[row][primary_bins[row][1]]) for row in range(len(sig_fft_reduced))]
    R2 = [np.real(sig_fft_reduced[row][primary_bins[row][1]]) if primary_bins[row][0] >= primary_bins[row][1] else np.real(sig_fft_reduced[row][primary_bins[row][0]]) for row in range(len(sig_fft_reduced))]

    I1 = [np.imag(sig_fft_reduced[row][primary_bins[row][0]]) if primary_bins[row][0] >= primary_bins[row][1] else np.imag(sig_fft_reduced[row][primary_bins[row][1]]) for row in range(len(sig_fft_reduced))]
    I2 = [np.imag(sig_fft_reduced[row][primary_bins[row][1]]) if primary_bins[row][0] >= primary_bins[row][1] else np.imag(sig_fft_reduced[row][primary_bins[row][0]]) for row in range(len(sig_fft_reduced))]
        
    #for i in range(len(R1)):
    #    fig_test, ax_test = plt.subplots(1)
    #    ax_test.plot(data_matrix_zero_pad[i],'.-')
    #    ax_test.plot(ifft(sig_fft_reduced[i]),'-')    
                
        
    #note change here from unvectorized code: sets ange1 to 0 if Re=0 and Im=0
    angle1 = [np.angle(R1[row]+1j*I1[row]) if I1[row] > 0 else 2*np.pi + np.angle(R1[row]+1j*I1[row]) if I1[row] < 0 else 0 if I1[row] == 0 and R1[row] > 0 else np.pi if I1[row] == 0 and R1[row] < 0 else 0 for row in range(len(R1))]
    
    #note change here from unvectorized code: sets angle2 to 0 if Re=0 and Im=0
    angle2 = [np.angle(R2[row]+1j*I2[row]) if I2[row] > 0 else 2*np.pi + np.angle(R2[row]+1j*I2[row]) if I2[row] < 0 else 0 if I2[row] == 0 and R2[row] > 0 else np.pi if I2[row] == 0 and R2[row] < 0 else 0 for row in range(len(R2))]
        
    interpolated_phase = [np.arctan2((I1[row]-R2[row]),(R1[row]+I2[row])) if R2[row] > 0 and I2[row] > 0 and I1[row] < 0 else np.arctan2((I1[row]+R2[row]),(R1[row]-I2[row])) if angle1[row] > angle2[row] else np.arctan2((I1[row]-R2[row]),(R1[row]+I2[row])) if angle2[row] > angle1[row] else np.arctan2((I1[row]+I2[row]),(R1[row]+R2[row])) for row in range(len(angle1))]
    interpolated_phase = np.unwrap(interpolated_phase)

    demod_t = np.array([np.median(row) for row in t_chunked])

    if correct_phase_jumps == True:
        phase_array_diff = np.diff(interpolated_phase)
        discontinuity_index = np.argwhere(phase_array_diff > phase_jump_threshold)
        discontinuity_index = discontinuity_index.reshape(1,len(discontinuity_index))[0]
               
        def linear_model(x,m,b):
            return m*x+b
            
        if len(discontinuity_index) != 0:
                
                           
            phase_split = np.split(interpolated_phase, discontinuity_index+1)
            t_split = np.split(demod_t, discontinuity_index+1)

                
            phase_split_corrected = np.array([])
            t_split_corrected = np.array([])
            for i in range(len(phase_split)):
                    
                if len(phase_split[i]) >= 4:
                        

                    t_drop_final = t_split[i]
                        
                    phase_drop_final = phase_split[i]        
                    
                        

                    linear_fit = curve_fit(linear_model, t_drop_final, phase_drop_final, check_finite=False)

                    fitted_line = linear_model(t_drop_final,linear_fit[0][0],linear_fit[0][1])

                    phase_remove_accumulation = phase_drop_final - fitted_line
                        
                else:
                    t_drop_final = t_split[i]
                    phase_drop_final = phase_split[i]
                    phase_remove_accumulation = phase_split[i]
                        
                phase_split_corrected = np.append(phase_split_corrected, phase_remove_accumulation)
                t_split_corrected = np.append(t_split_corrected, t_drop_final)
                    
            demod_t = t_split_corrected
            interpolated_phase = phase_split_corrected
      
    return demod_t, interpolated_phase#, reset_indices


def demodulate_with_template(t, sig, n_packets, correct_phase_jumps = False, phase_jump_threshold=0.4):#, start_idx):

    data = sig #fix this later; made it easier to copy paste working code
    n_packets = int(n_packets)

    derivative = np.diff(data)
    dd = np.diff(derivative)
    reset = np.argmax(derivative[0:n_packets-1])

    #d=np.diff(data)
    #dd=np.diff(d)
    #reset=np.argmax(dd)#+2

    #data_drop = data[reset+1:]
    data_drop = data[reset:]
    #data_drop = data[start_idx:]
    end = int(len(data_drop)/n_packets)*n_packets
    #input(f'end: {end}')
    data_drop = data_drop[:end]

    #input(f'data drop: {len(data_drop)}')


    t_drop = t[reset:]
    t_drop = t_drop[:end]

    #input(f't drop: {len(t_drop)}')

    data_matrix = np.reshape(data_drop, (int(len(data_drop)/32),32))

    data_matrix = np.array([row - np.average(row) for row in data_matrix])
    
    t_chunked = np.reshape(t_drop,(int(len(t_drop)/32), 32))
    
    
    subtract = np.array([data_matrix[i,:] - data_matrix[i-1,:] for i in np.arange(1,len(data_matrix),1)])

    #print(subtract)

    derivative_matrix = np.diff(data_matrix)

    #print(subtract.shape)
    #print(derivative_matrix.shape)

    phi_matrix_naive = subtract[:,1:] / derivative_matrix[:-1,:]

    #phi_naive = np.array([np.average(row) for row in phi_matrix_naive])


    phi_matrix_weighted = subtract[:,1:] * derivative_matrix[:-1,:]

    phi_weighted = np.array([np.average(row) for row in phi_matrix_weighted])

    phi_weighted_average = np.average(phi_weighted)
    phi_weighted_std = np.std(phi_weighted)

    phi_masked = np.array([entry if entry >= phi_weighted_average-3*phi_weighted_std and entry <= phi_weighted_average+3*phi_weighted_std else np.nan for entry in phi_weighted])


    demod_t = np.array([np.median(row) for row in t_chunked])

    return demod_t, phi_masked



def get_max_len_ind(ll):
    lens=[len(l) for l in ll]
    return np.argmax(lens)
def get_max_len_list(ll):
    lens=[len(l) for l in ll]
    return ll[np.argmax(lens)]
def get_dt(times):
    dt=np.median(np.diff(times))
    return dt
def mea_nphi0(times,delta_fs_ch,reset_freq,plot=False):
    peaks_neg,_=find_peaks(0-delta_fs_ch, distance=10,width=2.5)
    t_peaks=times[peaks_neg]
    t_peaks.sort()
    peaks_neg.sort()
    t_diff=np.diff(t_peaks)
    t_diff_2d=np.vstack((t_diff,np.zeros(t_diff.shape[0]))).T
    tree = spatial.cKDTree(t_diff_2d)
    idups_raw = tree.query_ball_tree(tree, 2./488.)
    if plot==True:
        plt.plot(times-times[0],delta_fs_ch)
        plt.scatter(times[peaks_neg]-times[0],delta_fs_ch[peaks_neg],color='r')
        plt.xlim(0,1)
        plt.show()
    nphi0=1/(np.median(t_diff[idups_raw[get_max_len_ind(idups_raw)]])*reset_freq)
    return nphi0

def find_n_phi0(time,data_cal,f_sawtooth,plot=True):
    """
    helper function which finds n_phi0 for all channels in dataset and returns the median
    calls mea_nphi0
    """
    n_phi0_array = np.array([])
    for i in range(len(data_cal)):
        try:
            n_phi0 = mea_nphi0(time,data_cal[i,:],f_sawtooth,plot=plot)
            n_phi0_array = np.append(n_phi0_array, n_phi0)
        except:
            continue
            #print ('bad channel', i)

    n_phi0 = np.median(n_phi0_array)
    print (n_phi0)
    return n_phi0

def mea_reset_t0(times,delta_fs_ch,reset_freq,plot=False):
    d=np.diff(delta_fs_ch)
    dd=np.diff(d)
    ind=np.argmax(dd)+2
    #print(f'index: {ind}')
    #print(f'len times: {len(times)}')
    t_init=times[ind]-times[0]-int((times[ind]-times[0])*reset_freq)/reset_freq
    ind_init= find_nearest_idx(times-times[0],t_init)
    if plot==True:
        plt.plot(times-times[0],delta_fs_ch,alpha=0.05)
        plt.scatter(times[ind_init]-times[0],delta_fs_ch[ind_init],color='r')
        plt.xlim(0,1)
        plt.show()
    return t_init



def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def find_file(p,filetype):
    #filetype could be target,eta,freqs,test,ALICPT_RDF
    if filetype=='target':
        filenames=filename=gl.glob(p+'%s_*'%filetype)
        filename = max(filenames, key=os.path.getctime)
    else:
        filename=gl.glob(p+'%s_*'%filetype)[0]
    return filename

# Cable Delay Removal

def measure_delay_test(test_sweeps,freq_list,plot=False):
    """
    Finds a segment without resonators for cable delay measurement.
    """
    delays=[]
    for i in range(test_sweeps.shape[1]):
        test_sweep=test_sweeps[:,i,:]
        freq_start=np.min(test_sweep[0,:].real)
        freq_stop=np.max(test_sweep[0,:].real)
        if np.any((freq_list > freq_start) & (freq_list < freq_stop)):continue
        #need to use :100 because there's a amp shift within each target sweep segment at the center  
        m,b = np.polyfit(test_sweep[0,:100].real, [cmath.phase(x) for x in test_sweep[1,:100]], 1)
        tau=0-m/(2*np.pi)
        delays.append(tau)
    delays=np.array(delays)
    delays_sel=delays[np.where(delays>0)]
    if plot==True:
        plt.figure(figsize=(11,8))
        plt.hist(delays_sel*1e9,bins=np.linspace(0,100,50))
        plt.xlabel('Cable Delay (ns)')
        plt.ylabel('Number of Segments')
    return delays_sel


def measure_delay_test_given_freq(test_sweeps,fmin,fmax,plot=False):
    """
    Manually enter frequency region for cable delay measurement.
    First 100 points currently used because of system stability issue. Might remove in the future if it gets better.
    """
    delays=[]
    for i in range(test_sweeps.shape[1]):
        test_sweep=test_sweeps[:,i,:]
        freq_start=np.min(test_sweep[0,:].real)
        freq_stop=np.max(test_sweep[0,:].real)
        if freq_start>fmin and freq_stop<fmax:
            m,b = np.polyfit(test_sweep[0,:100].real, [cmath.phase(x) for x in test_sweep[1,:100]], 1)
            tau=-m/(2*np.pi)
            delays.append(tau)
    delays=np.array(delays)
    delays_sel=delays[np.where(delays>0)]
    if plot==True:
        plt.figure(figsize=(11,8))
        plt.hist(delays_sel*1e9,bins=np.linspace(0,100,50))
        plt.xlabel('Cable Delay (ns)')
        plt.ylabel('Number of Segments')
    return delays_sel

def remove_delay(target_sweeps,delay,channels='all',start_channel=0,stop_channel=1000):
    target_sweeps_rm=target_sweeps.copy()
    
    if channels == 'all':
        loop_range = range(target_sweeps.shape[1])
    elif channels == 'some':
        loop_range = range(stop_channel-start_channel)
    
    for i in loop_range:
        target_sweep=target_sweeps[:,i,:]
        freqs=target_sweep[0,:]
        delay_fac=np.exp(1j*delay*2*np.pi*freqs)
        target_sweeps_rm[1,i,:]=target_sweeps[1,i,:]*delay_fac
            
    return target_sweeps_rm

def remove_delay_live(target_sweeps,delay,channels=[0]):
    target_sweeps_rm=target_sweeps.copy()

    for i in channels:
        target_sweep=target_sweeps[:,i,:]
        freqs=target_sweep[0,:]
        delay_fac=np.exp(1j*delay*2*np.pi*freqs)
        target_sweeps_rm[1,i,:]=target_sweeps[1,i,:]*delay_fac
            
    return target_sweeps_rm

def remove_delay_timestream(stream,f0s,delay):
    stream_rm=np.zeros((f0s.shape[0],stream.shape[1]),dtype = 'complex_')
    for i in range(f0s.shape[0]):
        #print(delay)
        #print(f0s)
        delay_fac=np.exp(1j*2*np.pi*f0s[i]*delay)
        #print(delay_fac)
        stream_rm[i,:]=stream[i]*delay_fac
    return stream_rm


#functions for resonator circle fitting

def calculate_M(target_sweep_one):
    I=target_sweep_one[1,:].real
    Q=target_sweep_one[1,:].imag
    z=I**2+Q**2
    M=np.zeros((4,4))
    M[0,0]=np.sum(z**2.)
    M[1,1]=np.sum(I**2.)
    M[2,2]=np.sum(Q**2.)
    M[3,3]=I.shape[0]
    M[0,1]=np.sum(I*z)
    M[1,0]=np.sum(I*z)
    M[0,2]=np.sum(Q*z)
    M[2,0]=np.sum(Q*z)
    M[0,3]=np.sum(z)
    M[3,0]=np.sum(z)
    M[1,2]=np.sum(I*Q)
    M[2,1]=np.sum(I*Q)
    M[1,3]=np.sum(I)
    M[3,1]=np.sum(I)
    M[2,3]=np.sum(Q)
    M[3,2]=np.sum(Q)
    return M


def measure_circle(target_sweep,f0):
    """ Finds the resonator circle's center

    Args:
        target_sweep: target S21 sweep around one resonator, col0 is freq, col1 is complex S21
        tau: cable delay in s

    Returns:
        center and radius on the complex plane, and initial phase
    """
    M=calculate_M(target_sweep)
    B = np.array([[0,0,0,-2],[0,1,0,0],[0,0,1,0],[-2,0,0,0]])
    VX=scipy.linalg.eig(M, B)
    X=VX[0]
    V=VX[1]
    C=np.sort(X,0)
    IX=np.argsort(X,0)
    Values = V[:,IX[1]]
    xc = -Values[1]/(2*Values[0])
    yc = -Values[2]/(2*Values[0])
    R = (xc**2+yc**2-Values[3]/Values[0])**0.5
    ind=find_nearest_idx(target_sweep[0,:], f0)
    phi_0=cmath.phase(target_sweep[1,ind]-xc-1j*yc)
    return np.array([xc+1j*yc,R,phi_0])  

def measure_circle_allch(target_sweeps,f0s,channels='all',start_channel=0,stop_channel=1000):
    cals=[]
    
    if channels == 'all':
        loop_range = range(target_sweeps.shape[1])
    elif channels == 'some':
        loop_range = range(stop_channel - start_channel)
    
    for i in loop_range:
        sweep=target_sweeps[:,i,:]
        cal=measure_circle(sweep,f0s[i])
        cals.append(cal)
            
    cals=np.vstack(cals)

    print(f'shape_cals: {cals.shape}')
    return cals  


def measure_circle_live(target_sweeps,f0s,channels=[0]):
    #loop_count = 0
    cals=[]
    
    for i in channels:
        sweep=target_sweeps[:,i,:]
        cal=measure_circle(sweep,f0s[i])
        cals.append(cal)
            
    cals=np.vstack(cals)
        
        #if loop_count == 0:
        #    cals = cal
        #else:
        #    cals = np.append(cals, cal)
        #loop_count += 1
    
    cals=np.vstack(cals)
    return cals

def get_phase(timestreams,calibration):
    """ Finds the resonator circle's center
    
    Args:
        timestream : timestream data in I and Q for all resonators
        phi_init: initial phase at f0
        calibration: col0 center, col1 radius, col2 initial phase

    Returns:
        phase of timestream in radian
    """
    phases=[]
    for i in range(calibration.shape[0]):
        timestream=timestreams[i]
        center = calibration[i,0]
        phi_init=calibration[i,2]
        timestream_origin=timestream-center
        timestream_rot=timestream_origin*np.exp(-1j*phi_init)
        phase=[cmath.phase(x) for x in timestream_rot]
        #phase-=phi_init
        phase=np.array(phase)
        phases.append(np.unwrap(phase.real))
    phases=np.vstack(phases)
    
    return phases


#plotting
def plot_s21(lo_data,labels = None):
    plt.figure(figsize=(14,8))
    count = 0
    for current_data in lo_data:
        ftones = np.concatenate(current_data[0])
        sweep_Z = np.concatenate(current_data[1])
        
        mag = 20* np.log10(np.abs(sweep_Z))
        
        if labels != None:
            plt.plot(ftones/1e9, mag.real, '-',label=labels[count], alpha=1)
        else:
            plt.plot(ftones/1e9, mag.real, '-', alpha=1)
        
        count += 1
    plt.grid()
    plt.xlabel('Frequency (GHz)',fontsize=16)
    plt.ylabel('$|S_{21}|$',fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if labels != None:
        plt.legend(fontsize=16)
    plt.show()
            
def plot_timestream(time, data, start_time = None, end_time = None, channel_nums = [0]):
    
    plt.figure(figsize=(11,8))
    for current_channel in channel_nums:
        plt.plot(time-time[0], data[current_channel], label = current_channel)
    
    plt.xlabel('Time (s)',fontsize=16)
    plt.ylabel('Phase (rad.)',fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim([start_time,end_time])    
    plt.legend(fontsize=16)
    plt.show()

def find_freqs_cable_delay_subtraction(initial_lo_sweep,target,n_freq):
   
    tests=initial_lo_sweep
    x=[]
    y=[]
    for i in range(tests.shape[1]):
        test_sweep=tests[:,i,:]
        x=np.append(x,test_sweep[0,:].real)
        y=np.append(y,20*np.log10(np.abs(test_sweep[1,:])))
    finite=np.asarray(np.where(np.isfinite(y) == True))
    yfinite=y[finite].flatten()
    xfinite=x[finite].flatten()
    idmax=np.where(yfinite == np.max(yfinite))
   
    # idgt=index array of the (x,y) values > target*ymax (id=index, gt=greater than)
    idgt=np.asarray(np.where(yfinite > (target*yfinite[idmax][0]))).flatten()
    # frequency range for cable delay
    f_start=xfinite[idgt[0]]
    f_end=xfinite[idgt[0]+n_freq]


    #fig_test, ax_test = plt.subplots(1)
    #ax_test.plot(xfinite,yfinite)
    #ax_test.plot(x[idgt[0]:idgt[0]+n_freq], y[idgt[0]:idgt[0]+n_freq])
   
    return f_start, f_end


def full_demod_process_live(ts_fr, Is_fr, Qs_fr, fs, f_sawtooth, use_channels, length, correct_phase_jumps, phase_jump_threshold, delays, calibration, tone_freqs, start_time):

    #read last 10s of data
    #truncate = int(length * fs)

    

    print(f'ts_fr[0]: {ts_fr[0]}')


    ts_len = len(ts_fr)
    Is_len = Is_fr.shape[1]
    Qs_len = Qs_fr.shape[1]

    print(f'len ts: {ts_len}')
    #print(f'len Is_fr: {Is_len}')
    #print(f'len Qs_fr: {Qs_len}')

    #Make all arrays the same length
    mlen = int(min([ts_len, Is_len, Qs_len]))

    #print(f'mlen: {mlen}')

    ts_trunc = ts_fr[:mlen]
    Is_trunc = Is_fr[:][:mlen]
    Qs_trunc = Qs_fr[:][:mlen]

    #print(f'len(ts_trunc): {len(ts_trunc)}')
    #print(f'len(Is_trunc): {Is_trunc.shape[1]}')

    #fix ts
    #ts_trunc = np.array([time + ts_trunc[0] - start_time for time in np.arange(Is_trunc.shape[1])/fs])
    ts_trunc = np.arange(Is_trunc.shape[1])/fs

    #print(f'len ts_trunc after fix: {len(ts_trunc)}')

    #remove cable delay
    IQ_stream_rm=remove_delay_timestream(Is_trunc+1j*Qs_trunc,tone_freqs,np.median(delays))
    #print(f'shape IQ_stream_rm: {IQ_stream_rm.shape}')

    #apply calibration
    data_cal=get_phase(IQ_stream_rm,calibration)
    #print(f'shape data_cal: {data_cal.shape}')


    #find t0 for all channels
    t0_array = np.array([])
    t_start = 0
    t_stop = length
    for current_channel in range(len(data_cal)):
        t0 = mea_reset_t0(ts_trunc[488*t_start:488*t_stop],data_cal[current_channel,488*t_start:488*t_stop],f_sawtooth,plot=False)
        #ts_freq = 1/np.nanmedian(np.diff(ts_fr))
        #t0 = mea_reset_t0(ts_fr[ts_freq*t_start:ts_freq*t_stop],data_cal[current_channel,ts_freq*t_start:ts_freq*t_stop],f_sawtooth,plot=False)
        t0_array = np.append(t0_array,t0)

    t0_med = np.nanmedian(t0_array)
    start_idx = find_nearest_idx(ts_trunc-ts_trunc[0], t0_med)
    #print(f'start_idx: {start_idx}')

    #demodulate requested channels
    counter = 0
    #print(f'use_channels: {use_channels}')
    for channel in use_channels:
        print(f'channel: {channel}')
        t_demod, data_demod, reset_indices = demodulate_with_fft(t=ts_trunc,
                                                                sig=data_cal[channel],
                                                                start_index=start_idx,                                                                      
                                                                f_fr=f_sawtooth,
                                                                phase_units='nPhi0',
                                                                correct_phase_jumps=correct_phase_jumps,
                                                                phase_jump_threshold=phase_jump_threshold,
                                                                plot_demod = False,
                                                                plot_demod_title=None,
                                                                intermediate_plotting_limits=[None,None],
                                                                plot_chunking_process = False,
                                                                plot_fft = False,
                                                                plot_fft_no_dc = False,
                                                                plot_limited_fft = False,
                                                                plot_fit = False,
                                                                plot_vectors = False)
        if counter == 0:
            data_demod_stacked = data_demod
        else:
            data_demod_stacked = np.vstack([data_demod_stacked, data_demod])
    
        counter += 1


    #return demoded data
    #print(f'len t_demod: {len(t_demod)}')
    #print(f'shape data_demod_stacked: {data_demod_stacked.shape}')

    return t_demod, data_demod_stacked




def full_demod_process(ts_file, f_sawtooth, method = 'fft', correct_phase_jumps = False, phase_jump_threshold = 0.4, n=0, channels='all',start_channel=0,stop_channel=1000,tone_init_path = '/home/matt/alicpt_data/tone_initializations', ts_path = '/home/matt/alicpt_data/time_streams', display_mode = 'terminal'):
    #n is number of points blanked before and after fr reset; only used when method='simple'
    #unpack data -> eventually change so that you give the ts data path and the function finds the associated tone initialization files

    print('using full_demod_process')

    init_freq = ts_file.split('_')[3]
    print(init_freq)
    init_time = ts_file.split('_')[4]
    print(init_time)
    init_directory = f'{tone_init_path}/fcenter_{init_freq}_{init_time}/'
    print(init_directory)
    
    initial_lo_sweep_path = find_file(init_directory, 'lo_sweep_initial')
    targeted_lo_sweep_path = find_file(init_directory, 'lo_sweep_targeted_2')
    tone_freqs_path = find_file(init_directory, 'freq_list_lo_sweep_targeted_1')
    ts_path = f'{ts_path}/{ts_file}'    
    
    initial_lo_sweep=np.load(initial_lo_sweep_path) #find initial lo sweep file
    targeted_lo_sweep=np.load(targeted_lo_sweep_path) #find targeted sweep file
    tone_freqs=np.load(tone_freqs_path) #find tone freqs
    #print(tone_freqs)
    if channels == 'some':
        tone_freqs = tone_freqs[start_channel + 23:stop_channel + 23 + 1]
        print(tone_freqs)
    ts_fr,Is_fr,Qs_fr=read_data(ts_path,channels=channels,start_channel=start_channel,stop_channel=stop_channel)    #note to self: limit tone_freqs to actively called channels; need to figure out channel numbering first
    
    #testing fixing the time breaks before the demod -- probably don't want to keep this but we'll see
    fs=512e6/(2**20)    #this line is incredibly important; need to make sure we match the data rate at all times; add an if statement for faster data rate data
    #fs=256e6/(2**19)
    ts_fr = np.arange(Is_fr.shape[1])/fs
    
    
    print(f'num of channels: {len(Is_fr)}')
    print(f'num of tones: {len(tone_freqs)}')
    
    """
    #depricated code for finding delay region
    #look at initial sweep
    plot_s21([initial_lo_sweep])
    
    
    #choose delay region - should automate finding an area without peaks later
    delay_region_start = float(input('Delay Region Start (GHz): '))*1e9
    delay_region_stop = float(input('Delay Region Stop (GHz): '))*1e9
    """
    #compute delay region
    print('looking for delay region')
    delay_region_start, delay_region_stop = find_freqs_cable_delay_subtraction(initial_lo_sweep,0.98,10000)
    print(f'start = {delay_region_start}')
    print(f'stop = {delay_region_stop}')
    
    #measure cable delay
    delays = measure_delay_test_given_freq(initial_lo_sweep,delay_region_start,delay_region_stop,plot=False)
    
    print(f'delay: {np.median(delays)}')

    #remove cable delay
    targeted_lo_sweep_rm=remove_delay(targeted_lo_sweep,
                                      np.median(delays),
                                      channels=channels,
                                      start_channel=start_channel,
                                      stop_channel=stop_channel)
    
    IQ_stream_rm=remove_delay_timestream(Is_fr+1j*Qs_fr,tone_freqs,np.median(delays))
    
    #measure circle parameters
    calibration=measure_circle_allch(targeted_lo_sweep_rm,
                                     tone_freqs,
                                     channels=channels,
                                     start_channel=start_channel,
                                     stop_channel=stop_channel) #finds circle center and initial phase for every channel
    
    print(calibration[0])
    #calibrate time stream
    data_cal=get_phase(IQ_stream_rm,calibration)

    #fig_testing, ax_testing = plt.subplots(1)
    #for i in [1]:
    #    ax_testing.plot(data_cal[i])
    
    #find nphi_0
    t_start=0
    t_stop=10

    n_phi0 = find_n_phi0(ts_fr[488*t_start:488*t_stop],data_cal[:,488*t_start:488*t_stop],f_sawtooth,plot=False)  #discard the first few seconds
    print(f'n_phi0: {n_phi0}')
    
    #find t0
    t0_array = np.array([])
    for current_channel in range(len(data_cal)):
        t0 = mea_reset_t0(ts_fr[488*t_start:488*t_stop],data_cal[current_channel,488*t_start:488*t_stop],f_sawtooth,plot=False)
        #ts_freq = 1/np.nanmedian(np.diff(ts_fr))
        #t0 = mea_reset_t0(ts_fr[ts_freq*t_start:ts_freq*t_stop],data_cal[current_channel,ts_freq*t_start:ts_freq*t_stop],f_sawtooth,plot=False)
        t0_array = np.append(t0_array,t0)

    t0_med = np.nanmedian(t0_array)
    
    #demod
    
    if method == 'simple' or method =='iv':
        t_demods=[]
        data_demods=[]
    elif method == 'fft':
        t_demods=np.array([])
        data_demods=np.array([])
        ch_count = 0
    start_idx = find_nearest_idx(ts_fr-ts_fr[0], t0_med)
    print(f'start index: {start_idx}')
    if display_mode == 'notebook':
        for chan in tqdm_notebook(range(data_cal.shape[0])):#np.arange(225,230,1):#range(data_cal.shape[0]):
            if method == 'iv':
                t_demod, data_demod = demodulate_for_iv(ts_fr[start_idx:]-ts_fr[start_idx], data_cal[chan, start_idx:], n_phi0, 3,f_sawtooth)
            
                t_demods.append(t_demod)
                data_demod_unwrap=np.unwrap(data_demod,period=1)
                data_demods.append(data_demod_unwrap)
            if method == 'simple':
                t_demod, data_demod, reset_indices = demodulate(ts_fr[start_idx:]-ts_fr[start_idx],
                                                                data_cal[chan, start_idx:],
                                                                n_phi0,
                                                                n,
                                                                f_sawtooth)
                t_demods.append(t_demod)
                data_demod_unwrap=np.unwrap(data_demod,period=1)
                data_demods.append(data_demod_unwrap)
            if method == 'fft':
                t_demod, data_demod, reset_indices = demodulate_with_fft(t=ts_fr,
                                                                        sig=data_cal[chan],
                                                                        start_index=start_idx,                                                                      
                                                                        f_fr=f_sawtooth,
                                                                        phase_units='nPhi0',
                                                                        correct_phase_jumps=correct_phase_jumps,
                                                                        phase_jump_threshold=phase_jump_threshold,
                                                                        plot_demod = False,
                                                                        plot_demod_title=None,
                                                                        intermediate_plotting_limits=[None,None],
                                                                        plot_chunking_process = False,
                                                                        plot_fft = False,
                                                                        plot_fft_no_dc = False,
                                                                        plot_limited_fft = False,
                                                                        plot_fit = False,
                                                                        plot_vectors = False)
                
                #print(t_demod)
                if ch_count == 0:
                    data_demods = data_demod
                else:
                    #t_demods = np.append(t_demods, np.array(t_demod))
                    data_demods = np.vstack([data_demods, np.array(data_demod)])
                t_demods = t_demod
                ch_count += 1
    elif display_mode == 'terminal':
        
        for chan in tqdm_terminal(range(data_cal.shape[0])):#np.arange(225,230,1):#range(data_cal.shape[0]):
            if method == 'simple':
                t_demod, data_demod, reset_indices = demodulate(ts_fr[start_idx:]-ts_fr[start_idx],
                                                                data_cal[chan, start_idx:],
                                                                n_phi0,
                                                                n,
                                                                f_sawtooth)
                t_demods.append(t_demod)
                data_demod_unwrap=np.unwrap(data_demod,period=1)
                data_demods.append(data_demod_unwrap)
            if method == 'fft':
                t_demod, data_demod, reset_indices = demodulate_with_fft(t=ts_fr,
                                                                        sig=data_cal[chan],
                                                                        start_index=start_idx,                                                                      
                                                                        f_fr=f_sawtooth,
                                                                        phase_units='nPhi0',
                                                                        correct_phase_jumps=False,
                                                                        phase_jump_threshold=0,
                                                                        plot_demod = False,
                                                                        plot_demod_title=None,
                                                                        intermediate_plotting_limits=[None,None],
                                                                        plot_chunking_process = False,
                                                                        plot_fft = False,
                                                                        plot_fft_no_dc = False,
                                                                        plot_limited_fft = False,
                                                                        plot_fit = False,
                                                                        plot_vectors = False)
                
                #print(t_demod)
                if ch_count == 0:
                    data_demods = data_demod
                else:
                    #t_demods = np.append(t_demods, np.array(t_demod))
                    data_demods = np.vstack([data_demods, np.array(data_demod)])
                t_demods = t_demod
                ch_count += 1
    

    data_demods=np.vstack(data_demods)
    if method == 'simple':
        t_demods=np.vstack(t_demods)
    
        data_dict = {'fr t': ts_fr, 
                    'fr data': data_cal, 
                    'nphi': n_phi0, 
                    't0': t0_med,
                    'start index': start_idx,
                    'demod t': t_demods[1], 
                    'demod data': data_demods, 
                    'channel freqs': tone_freqs, 
                    'fsawtooth': f_sawtooth,
                    'reset indices': reset_indices}
    elif method == 'fft':
        data_dict = {'fr t': ts_fr, 
                    'fr data': data_cal, 
                    'nphi': n_phi0, 
                    't0': t0_med,
                    'start index': start_idx,
                    'demod t': t_demods, 
                    'demod data': data_demods, 
                    'channel freqs': tone_freqs, 
                    'fsawtooth': f_sawtooth,
                    'reset indices': reset_indices,
                    'raw i':Is_fr,
                    'raw q':Qs_fr}

    return data_dict

def get_mean_current(bias_info,times,data):
    """
    get the mean of Ites for all channels based on bias values and times
    time is unix time
    data is the output dm.demodulate, in unit of phi0 numbers
    """
    data_bin=np.zeros((data.shape[0],bias_info.shape[0]))
    #print (data_bin.shape)
    for j in range(data.shape[0]):
        data_ch=data[j]
        for i in range(bias_info.shape[0]):
            bs_time=bias_info[i,0]
            chunck=data_ch[np.where((times>bs_time+0.4)&(times<bs_time+1.6))]
            data_bin[j,i]=np.mean(chunck)
    data_bin=np.apply_along_axis(unwrap_change_current_per_chan, 1, data_bin)       
    return data_bin    

def unwrap_change_current_per_chan(data_ch):
    data_ch_unwrap=np.unwrap(data_ch,discont=0.48,period=1)
    data_ch_uA=data_ch_unwrap*9
    return data_ch_uA

def IV_analysis_ch_new(bias_currents,resps,Rsh=0.4,filter_Rn_Al=False,plot='None'):
    """
    This method completely abandon the part that Ites might vary larger than 0.5 phi0 given the step limit of the Ibias
    i.e. it only cares about the Al normal state and part of Al transition state
    it assumes Rn_almn is 7.25mohm
    Outputs:
    dataframe containing timestream of Ites,Rtes,Vtes,Rn_almn,Rn_al
    """
    #only getting Al TES normal point

    Rn_almn=7.25
    peaks_nb,_=find_peaks(0-resps,width=20)
    if len(peaks_nb)==0 or peaks_nb[0] < 20:
        Rn_al=np.nan
        Rtes=np.ones(bias_currents.shape[0])*np.nan
        Vtes=np.ones(bias_currents.shape[0])*np.nan
        Ites=np.ones(bias_currents.shape[0])*np.nan
        bps=np.ones(bias_currents.shape[0])*np.nan
    else:
        peak_nb=peaks_nb[0]
        #print(peak_nb)
        resps_nb=resps[2:int(peak_nb-10)]
        bias_nb=bias_currents[2:peak_nb-10]
        r_ratio_al,shift = np.polyfit(bias_nb, resps_nb, 1)
        Rn_al=Rsh/r_ratio_al-Rsh*1e-3-Rn_almn*1e-3  #Ohm
        #print ('normal resistance for al TES',Rn_al)
        Ites=resps-shift #uA
        Ishunt=bias_currents*1e-3-Ites*1e-6 #A
        Vshunt=Ishunt*Rsh*1e-3 #V
        Rtes=Vshunt/(Ites*1e-6)-Rn_almn*1e-3 #Ohm
        bps=Rtes/Rn_al
        Vtes=Rtes*Ites#uV
        
        if filter_Rn_Al == True:
            if Rn_al < 5e-3 or Rn_al > 20e-3:##: #filter out non-responding channels
                Rn_al=np.nan
                Rtes=np.ones(bias_currents.shape[0])*np.nan
                Vtes=np.ones(bias_currents.shape[0])*np.nan
                Ites=np.ones(bias_currents.shape[0])*np.nan
                bps=np.ones(bias_currents.shape[0])*np.nan
        
        
        if plot=='IV':
            plt.gcf().subplots_adjust(bottom=0.2)
            plt.gcf().subplots_adjust(left=0.2)
            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.plot(Vtes,Ites,alpha=0.8)
            plt.xlabel('$V_{tes}$ (μV)',fontsize=18)
            plt.ylabel('$I_{tes}$ (μA)',fontsize=18)
        if plot=='bp':
            plt.gcf().subplots_adjust(bottom=0.2)
            plt.gcf().subplots_adjust(left=0.2)
            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.plot(bias_currents,bps*100,alpha=0.3)
            plt.ylim(0,1.2*100)
            plt.ylabel('$\%R_n$',fontsize=18)
            plt.xlabel(r'$I_{bias}$',fontsize=18)
    
    return Rn_al,Rtes,Vtes,Ites,bps

def detect_zero_and_fill(array):
    xnew = np.arange(array.shape[0])
    zero_idx = np.where(np.abs(array)<1e-2)
    xold = np.delete(xnew,zero_idx)
    array_sel = np.delete(array, zero_idx)
    f = interp1d(xold,array_sel,fill_value="extrapolate")
    array_new = f(xnew)
    return array_new


def IV_correction(resps):
    """
    This function smooth the IV curve and then find the normal and superconducting point
    it will then try to correct the jump caused by unwrapping large Ites changes with current Ibias resolution
    """
    #only getting Al TES normal point
    peaks_nb,_=find_peaks(0-resps,width=20)
    smooth=savgol_filter(resps, resps.shape[0], 10)
    smooth=detect_zero_and_fill(smooth)
    dd=np.diff(np.diff(smooth))
    ind_sc=np.argmin(dd)+2
    """
    if len(peaks_nb)==0 or peaks_nb[0]<30:
        return np.zeros(resps.shape[0])
    """
    peak_nb=peaks_nb[0]
    #here we know that before peaks_nb[0] and after peak_sc the Ites is monotonic 
    resps_nb=resps[2:int(peak_nb-10)]
    resps_sc=resps[int(ind_sc[0]+10):-10]
    resps_cor=np.zeros(resps.shape[0])
    resps_cor_acc=np.zeros(resps.shape[0])
    for i in range(resps.shape[0]):
        if i>=2 and i<peak_nb-10 and resps[i]-resps[i-1]>0.5:
            resps_cor[i]=-9
        if i>= ind_sc+10 and i<resps.shape[0]-5 and resps[i]-resps[i-1]>0.5:
            resps_cor[i]=-9
    for i in range(resps.shape[0]):
        if i>0:
            resps_cor_acc[i]=np.sum(resps_cor[:i+1])
    resps_corr=resps+resps_cor_acc
    return resps_corr, ind_sc


    
def IV_correction_median(resps, peak_nb, peak_sc):
    """
    This function smooth the IV curve and then find the normal and superconducting point
    it will then try to correct the jump caused by unwrapping large Ites changes with current Ibias resolution
    """ 
   
    #here we know that before peaks_nb[0] and after peak_sc the Ites is monotonic 
    #resps_nb=resps[2:int(peak_nb-10)]
    #resps_sc=resps[int(peak_sc+10):-10]
    resps_cor=np.zeros(resps.shape[0])
    resps_cor_acc=np.zeros(resps.shape[0])
    for i in range(resps.shape[0]):
        if i>=2 and i<peak_nb-10 and resps[i]-resps[i-1]>0.5:
            resps_cor[i]=-9
        if i>= peak_sc+10 and i<resps.shape[0]-5 and resps[i]-resps[i-1]>0.5:
            resps_cor[i]=-9
    for i in range(resps.shape[0]):
        if i>0:
            resps_cor_acc[i]=np.sum(resps_cor[:i+1])
    resps_corr=resps+resps_cor_acc
    return resps_corr

def find_transition_points(resps):
    peaks_nb,_=find_peaks(0-resps,width=20)
    max_ites=np.max(resps)
    min_ites=np.min(resps)
    smooth=savgol_filter(resps, resps.shape[0], 10)
    smooth=detect_zero_and_fill(smooth)
    print(f'peaks_nb: {peaks_nb}')
    print(f'max - min: {max_ites-min_ites}')
    print(f'peaks_sc: {peaks_sc}')
    peaks_sc,_=find_peaks(smooth,width=20,prominence=20)
    if len(peaks_nb)==0 or peaks_nb[0] < 30 or max_ites-min_ites<20 or len(peaks_sc)==0:
        return np.nan, np.nan
    else:      
        peak_nb=peaks_nb[0]
        peak_sc=peaks_sc[0]
        return peak_nb, peak_sc


def IV_analysis_ch_duo(bias_currents,resps,Rsh=0.4,filter_Rn_Al=False,plot='None'):#(bias_currents,resps,peak_nb,peak_sc,Rsh=0.4,filter_Rn_Al=False,plot='None'):
    """
    This method correct the SC and normal region for jump phi0 issue, and then fit for TES parameters
    Outputs:
    dataframe containing timestream of Ites,Rtes,Vtes,Rn_almn,Rn_al
    """ 
    peaks_nb,_=find_peaks(0-resps,width=20)
    max_ites=np.max(resps)
    min_ites=np.min(resps)
    if len(peaks_nb)==0 or peaks_nb[0] < 30 or max_ites-min_ites<20:
    #if max_ites-min_ites<20:
        Rn_almn=np.nan
        Rn_al=np.nan
        Rtes=np.ones(bias_currents.shape[0])*np.nan
        Vtes=np.ones(bias_currents.shape[0])*np.nan
        Ites=np.ones(bias_currents.shape[0])*np.nan
        bps=np.ones(bias_currents.shape[0])*np.nan
        Pbias=np.ones(bias_currents.shape[0])*np.nan
        resps_correct=np.ones(bias_currents.shape[0])*np.nan
    else:   
        peak_nb=peaks_nb[0]
        resps_correct,peak_sc=IV_correction(resps)
        #resps_correct=IV_correction_median(resps,peak_nb,peak_sc)
        resps_nb=resps_correct[2:int(peak_nb-10)]
        resps_sc=resps_correct[int(peak_sc+20):-10]
        bias_nb=bias_currents[2:int(peak_nb-10)]
        bias_sc=bias_currents[int(peak_sc+20):-10]
        #print(bias_nb.shape)
        #print(resps_nb.shape)
        #print(bias_sc.shape)
        #print(resps_sc.shape)
        r_ratio_al,shift = np.polyfit(bias_nb, resps_nb, 1)
        if resps_sc.shape[0]>5:
            r_ratio_almn,shift_sc = np.polyfit(bias_sc, resps_sc, 1)
            Rn_almn=Rsh/r_ratio_almn-Rsh*1e-3 #Ohm
            Rn_al=Rsh/r_ratio_al-Rsh*1e-3-Rn_almn  #Ohm
            #print ('normal resistance for al TES',Rn_al)
            Ites=resps_correct-shift #uA
            Ishunt=bias_currents*1e-3-Ites*1e-6 #A
            Vshunt=Ishunt*Rsh*1e-3 #V
            Rtes=Vshunt/(Ites*1e-6)-Rn_almn #Ohm
            bps=Rtes/Rn_al
            Vtes=Rtes*Ites#uV
            Pbias=Ites**2*(Rtes+Rn_almn) #pW the overall heating caused by both Al and Almn
        else:
            Rn_almn=7.25e-3
            Rn_al=Rsh/r_ratio_al-Rsh*1e-3-Rn_almn  #Ohm
            #print ('normal resistance for al TES',Rn_al)
            Ites=resps-shift #uA
            Ishunt=bias_currents*1e-3-Ites*1e-6 #A
            Vshunt=Ishunt*Rsh*1e-3 #V
            Rtes=Vshunt/(Ites*1e-6)-Rn_almn #Ohm
            bps=Rtes/Rn_al
            Vtes=Rtes*Ites#uV
            Pbias=Ites**2*(Rtes+Rn_almn) #pW the overall heating caused by both Al and Almn                   
        if filter_Rn_Al == True:
            if Rn_al < 5e-3 or Rn_al > 20e-3:##: #filter out non-responding channels
                Rn_al=np.nan
                Rtes=np.ones(bias_currents.shape[0])*np.nan
                Vtes=np.ones(bias_currents.shape[0])*np.nan
                Ites=np.ones(bias_currents.shape[0])*np.nan
                bps=np.ones(bias_currents.shape[0])*np.nan
        if plot == 'orgorg':
            plt.gcf().subplots_adjust(bottom=0.2)
            plt.gcf().subplots_adjust(left=0.2)
            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.plot(bias_currents,resps,alpha=0.8)
            plt.xlabel('$I_{bias}$ (mA)',fontsize=18)
            plt.ylabel('$I_{tes}$ (μA)',fontsize=18)
            
        
        if plot=='org':
            plt.gcf().subplots_adjust(bottom=0.2)
            plt.gcf().subplots_adjust(left=0.2)
            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.plot(bias_currents,Ites,alpha=0.8)
            plt.xlabel('$I_{bias}$ (mA)',fontsize=18)
            plt.ylabel('$I_{tes}$ (μA)',fontsize=18)



        if plot=='IV':
            plt.gcf().subplots_adjust(bottom=0.2)
            plt.gcf().subplots_adjust(left=0.2)
            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.plot(Vtes,Ites,alpha=0.8)
            plt.xlabel('$V_{tes}$ (μV)',fontsize=18)
            plt.ylabel('$I_{tes}$ (μA)',fontsize=18)
        if plot=='bp':
            plt.gcf().subplots_adjust(bottom=0.2)
            plt.gcf().subplots_adjust(left=0.2)
            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.plot(bias_currents,bps*100,alpha=0.3)
            plt.ylim(0,1.2*100)
            plt.ylabel('$\%R_n$',fontsize=18)
            plt.xlabel(r'$I_{bias}$',fontsize=18)
        if plot=='pbias':
            plt.gcf().subplots_adjust(bottom=0.2)
            plt.gcf().subplots_adjust(left=0.2)
            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.plot(Pbias,Rtes*1000,alpha=0.3)
            plt.ylim(0,15)
            plt.xlim(200,900)
            plt.ylabel(r'$R_{tes}$',fontsize=18)
            plt.xlabel(r'$P_{bias}$',fontsize=18)
        #print (Rn_almn)
        #print (Rn_al)
    return Rn_almn,Rn_al,Rtes,Vtes,Ites,bps,Pbias,resps_correct    

    
def full_iv_process(iv_file,f_sawtooth,demod_method='iv',Rsh=0.4,iv_path = '/home/matt/alicpt_data/IV_data', filter_Rn_Al=False, plot='None'):
    """
    wrapper examining all IV curves from a dataset
    """
    #only getting Al TES normal point
    
    bias_currents_path = find_file(f'{iv_path}/{iv_file}/', 'bias_data')
    bias_currents = np.loadtxt(bias_currents_path)    
    
    ts_path = find_file(f'{iv_path}/{iv_file}/', 'ts')
    ts_directory = ts_path.split('/')[-2]
    ts_filename = ts_path.split('/')[-1]

    print(f'iv_path: {iv_path}')
    print(f'ts_directory: {ts_directory}')
    
    print(f'tone init path: {iv_path.split("/")[-2]}/tone_initializations')

    iv_path_split = iv_path.split('/')
    
    demod_data = full_demod_process(ts_filename,
                                    f_sawtooth,
                                    method=demod_method,
                                    n=5,
                                    ts_path = f'{iv_path}/{iv_file}',
                                    tone_init_path = f'/{iv_path_split[1]}/{iv_path_split[2]}/{iv_path_split[3]}/tone_initializations')
    start_idx = find_nearest_idx(demod_data['fr t']-demod_data['fr t'][0], demod_data['t0'])
    
    data_demods_bin = get_mean_current(bias_currents,demod_data['demod t']+demod_data['fr t'][start_idx],demod_data['demod data'])
    
    Rn_almn_list = []
    Rn_al_list = []
    Rtes_list = []
    Vtes_list = []
    Ites_list = []
    bps_list = []
    Pbias_list = []
    resps_correct_list = []
    
    #peak_nb_array = np.array([])
    #peak_sc_array = np.array([])
    #for ch in tqdm_notebook(range(data_demods_bin.shape[0])):
    #    peak_nb, peak_sc = find_transition_points(data_demods_bin[ch])
    #    peak_nb_array = np.append(peak_nb_array, peak_nb)
    #    peak_sc_array = np.append(peak_sc_array, peak_sc)
    
    #print(f'peak_nb_array: {peak_nb_array}')
    #peak_nb_median = np.nanmedian(peak_nb_array)
    #print(f'peak_nb_median: {peak_nb_median}')
    #peak_sc_median = np.nanmedian(peak_sc_array)
    #print(f'peak_sc_median: {peak_sc_median}')

    for ch in tqdm_notebook(range(data_demods_bin.shape[0])):
        Rn_almn_ch,Rn_al_ch,Rtes_ch,Vtes_ch,Ites_ch,bps_ch,Pbias_ch,resps_correct_ch = IV_analysis_ch_duo(bias_currents[:,1],
                                                                                                          data_demods_bin[ch], 
                                                                                                          #peak_nb=peak_nb_median,
                                                                                                          #peak_sc=peak_sc_median,
                                                                                                          Rsh=0.4,
                                                                                                          filter_Rn_Al=filter_Rn_Al, 
                                                                                                          plot=plot)
        
        Rn_almn_list.append(Rn_almn_ch)
        Rn_al_list.append(Rn_al_ch)
        Rtes_list.append(Rtes_ch)
        #print (Rtes.shape)
        Vtes_list.append(Vtes_ch)
        Ites_list.append(Ites_ch)
        bps_list.append(bps_ch)
        Pbias_list.append(Pbias_ch)
        resps_correct_list.append(resps_correct_ch)
        
    #Rn_al = np.vstack(Rn_al)
    Rtes_list = np.vstack(Rtes_list)
    Vtes_list = np.vstack(Vtes_list)
    Ites_list = np.vstack(Ites_list)
    bps_list = np.vstack(bps_list)
    Pbias_list = np.vstack(Pbias_list)
    resps_correct_list = np.vstack(resps_correct_list)
    
    data_dict = {'Ibias': bias_currents[:,1], 
                 'Rn Al': Rn_al_list,
                 'Rn AlMn': Rn_almn_list,
                 'Rtes': Rtes_list,
                 'Vtes': Vtes_list,
                 'Ites': Ites_list,
                 'bps': bps_list,
                 'Pbias': Pbias_list,
                 'resps correct': resps_correct_list,
                 'binned data': data_demods_bin,
                 'time series data': demod_data}
       
    return data_dict

def get_channel_response_summary(IV_analysis_result,filepath=None):
    active_channel_chart = pd.DataFrame({'Channel Freq (Hz)': IV_analysis_result['demod data']['channel freqs'].real, 'Rn Al (mOhm)': IV_analysis_result['Rn Al']})
    if filepath != None:
        active_channel_chart.to_csv(filepath,sep=',')
    return active_channel_chart

def get_pbias(iv_data, bp_percent, split_pt=0,plot = False):
    
    bias_array = np.array([])
    for ch in range(len(iv_data['Vtes'])):
        if np.isnan(iv_data['bps'][ch]).all():
            bias_array = np.append(bias_array,np.nan)
            continue
        else:
            split_pt=int(len(iv_data['bps'][ch])/1.5)
            bp_index = find_nearest_idx(iv_data['bps'][ch][:split_pt]*100, 50)       
            bias_pt = iv_data['Pbias'][ch][bp_index]
            #print(bias_pt)
            bias_array = np.append(bias_array,bias_pt)

            if plot == True:
                plt.plot(data_i['Ibias'],data_i['bps'][ch]*100)
                plt.plot(data_i['Ibias'][Psat_index],data_i['bps'][ch][Psat_index]*100,'*')
                plt.vlines(data_i['Ibias'][Psat_index],0,100)
    
    return bias_array

def match_freqs(f1,f2,dist=1e5):
    f1_ind=np.vstack((f1,np.arange(f1.shape[0])))
    f1_ind=np.vstack((f1_ind,np.ones(f1.shape[0])*0))
    f2_ind=np.vstack((f2,np.arange(f2.shape[0])))
    f2_ind=np.vstack((f2_ind,np.ones(f2.shape[0])*1))
    fall=np.vstack((f1_ind.T,f2_ind.T))
    tree = spatial.cKDTree(fall[:,0:1])
    idups_raw=tree.query_ball_tree(tree,dist)
    idups_redu=[]
    for match in idups_raw:
        if len(match)!=2: continue
        else: 
            if int(fall[match[0],2])==0 and int(fall[match[1],2])==1:
                idups_redu.append(match)
    idups_ids=[fall[x,1].astype(int) for x in idups_redu]
    return idups_ids

def remove_chop(t,sig,demod_period,time_method='original',phase_units='rad',correct_phase_jumps=False,phase_jump_threshold=0,plot_demod = False,plot_demod_title=None,intermediate_plotting_limits=[None,None],plot_chunking_process = False,plot_fft = False,plot_fft_no_dc = False,plot_limited_fft = False,plot_fit = False,plot_vectors = False):
    
    if intermediate_plotting_limits[0] == None:
        intermediate_plotting_limits[0] = t[0]
    if intermediate_plotting_limits[1] == None:
        intermediate_plotting_limits[1] = t[-1]
    
    
    if plot_demod == True:
        fig_demod, ax_demod = plt.subplots(1)
        ax_demod.set_ylim([-0.05,1])
            
    chunk_count = 0
    
    #establish array for storing phase
    phase_array = np.array([])

    if plot_chunking_process == True:
        #print(sig[ch])
        fig1, ax1 = plt.subplots(1)
        ax1.plot(t,sig,'.-')
        ax1.set_xlabel('$t$ (s)')
        ax1.set_ylabel('Resonator Position (arb.)')
        #ax1.set_title(f'Ch. {ch}')

    #interpolate the data
    interpolation = interp1d(t, sig,fill_value='extrapolate')


    t_final = round((t[-1])/(demod_period))*(demod_period) #make final time the latest time that fits an integer number of demood periods; will interpolate to here
    t_start = t[0]
    t_elapsed = t_final - t_start
    n_reset_periods = t_elapsed * 1/demod_period #will be replaced when we have the fr reference

    t_interp = np.linspace(t_start, t_final, round(n_reset_periods)*1024) #interpolate so that every chunk has 1024 points; will make fft faster        
    sig_interp = interpolation(t_interp)

    if plot_chunking_process == True:
        fig2, ax2 = plt.subplots(1)
        ax2.plot(t_interp, sig_interp,'.')
        ax2.set_xlabel('$t$ (s)')
        ax2.set_ylabel('Resonator Position (arb.)')
        #ax2.set_title(f'Ch. {ch}')

    t_chunked = np.reshape(t_interp,(int(len(t_interp)/1024), 1024))
    sig_chunked = np.reshape(sig_interp,(int(len(sig_interp)/1024), 1024))
    sig_average = sig_chunked.mean(axis=1, keepdims=True)
    sig_chunked = sig_chunked - sig_average

    if plot_chunking_process == True:
        fig3, ax3 = plt.subplots(1)
        for chunk in range(len(t_chunked)):
            if np.median(t_chunked[chunk]) >= intermediate_plotting_limits[0] and np.median(t_chunked[chunk]) <= intermediate_plotting_limits[1]:
                ax3.plot(t_chunked[chunk],sig_chunked[chunk],'.-')
                ax3.set_xlabel('$t$ (s)')
                ax3.set_ylabel('Resonator Position (arb.)')
                #ax3.set_title(f'Ch. {ch}; Chunk {ch}; t_demod = {np.median(t_chunked[chunk])}')
                
                
    reset_mask = np.append(np.zeros(50),np.ones(1024-50*2))
    reset_mask = np.append(reset_mask,np.zeros(50))

    sig_chunked = np.array([row*reset_mask for row in sig_chunked])


    t_increase = (t_chunked[0,-1] + np.median(np.diff(t_chunked[0])))
    t_new = t_chunked + t_increase

    t_zero_padded = np.hstack((t_chunked, t_new))        
    sig_zero_padded = np.hstack((sig_chunked, np.zeros(sig_chunked.shape)))

    if plot_chunking_process == True and t[0] >= intermediate_plotting_limits[0] and t[0] <= intermediate_plotting_limits[1]:
        fig4, ax4 = plt.subplots(1)
        for chunk in range(len(t_zero_padded)):
            #fft_fit = ifft(sig_fft) #only waste computation time on computing the fit if we actually want to plot it; otherwise we'll just use the fft above
            ax4.plot(t_zero_padded[chunk],sig_zero_padded[chunk],'.')
            ax4.set_xlabel('$t$ (s)')
            ax4.set_ylabel('Resonator Position (arb.)')
            #ax4.set_title(f'Ch. {ch}; Chunk {ch}; t_demod = {np.median(t_chunked[chunk])}')

    sig_fft = fft(sig_zero_padded)   
    freq_fft = fftfreq(len(t_zero_padded[0]),np.median(np.diff(t_zero_padded[0])))

    if plot_fft == True:

        for chunk in range(len(t_zero_padded)):
            if np.median(t_chunked[chunk]) >= intermediate_plotting_limits[0] and np.median(t_chunked[chunk]) <= intermediate_plotting_limits[1]:
                fig5, ax5 = plt.subplots(1)
                ax5.stem(freq_fft,np.abs(sig_fft[chunk]))
                #ax5.set_title(f'Full FFT; Ch. {ch}; Chunk {chunk}; t_demod = {np.median(t_chunked[chunk])}')  
                ax5.set_xlabel('$f$ (Hz.)')
                ax5.set_ylabel('FFT Power (arb.)')


    #remove dc from fft:
    sig_fft_no_dc = deepcopy(sig_fft)
    sig_fft_no_dc[:,0] = 0

    if plot_fft_no_dc == True:

        for chunk in range(len(t_zero_padded)):
            if np.median(t_chunked[chunk]) >= intermediate_plotting_limits[0] and np.median(t_chunked[chunk]) <= intermediate_plotting_limits[1]:
                fig6, ax6 = plt.subplots(1)
                ax6.stem(freq_fft,np.abs(sig_fft_no_dc[chunk]))
                #ax6.set_title(f'No DC FFT; Ch. {ch}; Chunk {chunk}; t_demod = {np.median(t_chunked[chunk])}')  
                ax6.set_xlabel('$f$ (Hz.)')
                ax6.set_ylabel('FFT Power (arb.)')


    #keep largest fourier component
    sig_fft_reduced = deepcopy(sig_fft_no_dc)

    primary_bins = [np.argpartition(np.abs(row), -4)[-4:] for row in sig_fft_reduced] #identify the four greatest bins (two on each side of the fft)
    primary_bins = [np.sort(row) for row in primary_bins]
    #primary_bins = [np.array([17,2031]) for row in sig_fft_reduced]
    print(primary_bins)
    mask = np.zeros_like(sig_fft_reduced)
    [np.put(mask[row],primary_bins[row],1) for row in range(len(mask))]


    sig_fft_reduced = sig_fft_reduced * mask

    #
    if plot_limited_fft == True:

        for chunk in range(len(t_zero_padded)):
            if np.median(t_chunked[chunk]) >= intermediate_plotting_limits[0] and np.median(t_chunked[chunk]) <= intermediate_plotting_limits[1]:
                fig7, ax7 = plt.subplots(1)
                ax7.stem(freq_fft,np.abs(sig_fft_reduced[chunk]))  
                #ax7.set_title(f'Reduced FFT; Ch. {ch}; Chunk {chunk}; t_demod = {np.median(t_chunked[chunk])}')  
                ax7.set_xlabel('$f$ (Hz.)')
                ax7.set_ylabel('FFT Power (arb.)')

    if plot_fit == True:

        for chunk in range(len(t_zero_padded)):
            if np.median(t_chunked[chunk]) >= intermediate_plotting_limits[0] and np.median(t_chunked[chunk]) <= intermediate_plotting_limits[1]:
                limited_fit = ifft(sig_fft_reduced)
                fig8, ax8 = plt.subplots(1)
                ax8.plot(t_zero_padded[chunk], sig_zero_padded[chunk], '.')
                ax8.plot(t_zero_padded[chunk], limited_fit[chunk], '-')
                ax8.set_xlabel('$t$ (s)')
                ax8.set_ylabel('Resonator Position (arb.)')
                #ax8.set_title(f'Ch. {ch}; Chunk {chunk}; t_demod = {np.median(t_chunked[chunk])}')
    
    #find phase for each chunk          
    R1 = [np.real(sig_fft_reduced[row][primary_bins[row][0]]) if primary_bins[row][0] >= primary_bins[row][1] else np.real(sig_fft_reduced[row][primary_bins[row][1]]) for row in range(len(sig_fft_reduced))]
    R2 = [np.real(sig_fft_reduced[row][primary_bins[row][1]]) if primary_bins[row][0] >= primary_bins[row][1] else np.real(sig_fft_reduced[row][primary_bins[row][0]]) for row in range(len(sig_fft_reduced))]

    I1 = [np.imag(sig_fft_reduced[row][primary_bins[row][0]]) if primary_bins[row][0] >= primary_bins[row][1] else np.imag(sig_fft_reduced[row][primary_bins[row][1]]) for row in range(len(sig_fft_reduced))]
    I2 = [np.imag(sig_fft_reduced[row][primary_bins[row][1]]) if primary_bins[row][0] >= primary_bins[row][1] else np.imag(sig_fft_reduced[row][primary_bins[row][0]]) for row in range(len(sig_fft_reduced))]




    #note change here from unvectorized code: sets ange1 to 0 if Re=0 and Im=0
    angle1 = [np.angle(R1[row]+1j*I1[row]) if I1[row] > 0 else 2*np.pi + np.angle(R1[row]+1j*I1[row]) if I1[row] < 0 else 0 if I1[row] == 0 and R1[row] > 0 else np.pi if I1[row] == 0 and R1[row] < 0 else 0 for row in range(len(R1))]

    #note change here from unvectorized code: sets angle2 to 0 if Re=0 and Im=0
    angle2 = [np.angle(R2[row]+1j*I2[row]) if I2[row] > 0 else 2*np.pi + np.angle(R2[row]+1j*I2[row]) if I2[row] < 0 else 0 if I2[row] == 0 and R2[row] > 0 else np.pi if I2[row] == 0 and R2[row] < 0 else 0 for row in range(len(R2))]


    if plot_vectors == True:

        summed_vectors = [(R1[row]+I2[row])+1j*(I1[row]-R2[row]) if R2[row] > 0 and I2[row] > 0 and I1[row] < 0 else (R1[row]-I2[row])+1j*(I1[row]+R2[row]) if angle1[row] > angle2[row] else (R1[row]+I2[row])+1j*(I1[row]-R2[row]) if angle2[row] > angle1[row] else (R1[row]+R2[row])+1j*(I1[row]+I2[row]) for row in range(len(angle1))]


        for chunk in range(len(R1)):
            if np.median(t_chunked[chunk]) >= intermediate_plotting_limits[0] and np.median(t_chunked[chunk]) <= intermediate_plotting_limits[1]:
                fig9, ax9 = plt.subplots(1)
                ax9.set_aspect('equal', adjustable='box')
                vec1 = [0+1j*0, R1[chunk]+1j*I1[chunk]]
                vec2 = [0+1j*0, R2[chunk]+1j*I2[chunk]]
                summed = [0+1j*0, summed_vectors[chunk]]

                ax9.plot(np.real(vec1), np.imag(vec1),'-',label='vec1')
                ax9.plot(np.real(vec2), np.imag(vec2),'-',label='vec2')
                ax9.plot(np.real(summed), np.imag(summed),'-',label='summed')
                ax9.set_xlabel('Real')
                ax9.set_ylabel('Imaginary')
                ax9.set_title(f'Ch. {ch}; Chunk {chunk}; t_demod = {np.median(t_chunked[chunk])}')
                ax9.legend()


    #note change here form unvectorized code: sets interpolated_phase to the phase of the sum of the two angles if they exactly align from the beginning
    demod_data = [np.abs((R1[row]+I2[row])+1j*(I1[row]-R2[row]),) if R2[row] > 0 and I2[row] > 0 and I1[row] < 0 else np.abs((R1[row]-I2[row])+1j*(I1[row]+R2[row])) if angle1[row] > angle2[row] else np.abs((R1[row]+I2[row])+1j*(I1[row]-R2[row])) if angle2[row] > angle1[row] else np.abs((R1[row]+R2[row])+1j*(I1[row]+I2[row])) for row in range(len(angle1))]
    #interpolated_phase = np.unwrap(interpolated_phase)
    
    demod_t = np.array([np.median(row) for row in t_chunked])
    
    if plot_demod == True:
        fig_test, ax_test = plt.subplots(1)
        ax_test.plot(demod_t, demod_data, '.-')
    
    
    return demod_t, demod_data

def read_channel_file(channel_file):
    channel_standard = pd.read_csv(channel_file,sep=',')
    channel_nums = channel_standard['Channel Number'].to_numpy(int)
    channel_freqs = channel_standard['Frequency'].to_numpy(float)
    bias_lines = channel_standard['Bias Line'].to_numpy(int)
    optically_active = channel_standard['Optically Active'].to_numpy(int)
    selected_channels = channel_standard['Selected'].to_numpy(int)
    
    return channel_standard, channel_nums, channel_freqs, bias_lines, optically_active, selected_channels


def channel_setup_all(init_directory, channel_standard_file):

    initial_lo_sweep_path = find_file(init_directory, 'lo_sweep_initial')
    targeted_lo_sweep_path = find_file(init_directory, 'lo_sweep_targeted_2')
    tone_freqs_path = find_file(init_directory, 'freq_list_lo_sweep_targeted_1')
    
    channel_standard, channel_nums, channel_freqs, bias_lines, optically_active, selected_channels = read_channel_file(channel_standard_file)
    
    #standard channels to use
    active_channels = np.argwhere(selected_channels == 1)
    print(active_channels)
    use_standard_channels = np.reshape(active_channels,(1,active_channels.shape[0]))[0]  
    
    #loading up tone init
    initial_lo_sweep=np.load(initial_lo_sweep_path) #find initial lo sweep file
    targeted_lo_sweep=np.load(targeted_lo_sweep_path) #find targeted sweep file
    tone_freqs=np.load(tone_freqs_path) #find tone freqs

    

    match = match_freqs(tone_freqs,channel_freqs[use_standard_channels],dist=1e5)
    use_current_init_channels = np.array([item[0] for item in match])
    use_current_init_channels = use_current_init_channels[0:int(len(use_current_init_channels)/2)]    
    
    #frequencies of channels    
    current_freqs = channel_freqs[use_current_init_channels]


    delay_region_start, delay_region_stop = find_freqs_cable_delay_subtraction(initial_lo_sweep,0.98,10000)
    print(f'start = {delay_region_start}')
    print(f'stop = {delay_region_stop}')
    
    #measure cable delay
    delays = measure_delay_test_given_freq(initial_lo_sweep,delay_region_start,delay_region_stop,plot=False)
    
    
    
    #remove cable delay
    targeted_lo_sweep_rm=remove_delay_live(targeted_lo_sweep,
                                           np.median(delays),
                                           channels=range(len(tone_freqs)))
    
    #measure circle parameters
    calibration=measure_circle_live(targeted_lo_sweep_rm,
                                    tone_freqs,
                                    channels=range(len(tone_freqs))) #finds circle center and initial phase for every channel
    
    
    print(f'tone freqs: {tone_freqs}')

    return np.median(delays), calibration, tone_freqs, match, use_current_init_channels, current_freqs




def channel_setup(init_directory, channel_standard_file):
        
    initial_lo_sweep_path = find_file(init_directory, 'lo_sweep_initial')
    targeted_lo_sweep_path = find_file(init_directory, 'lo_sweep_targeted_2')
    tone_freqs_path = find_file(init_directory, 'freq_list_lo_sweep_targeted_1')
    
    channel_standard, channel_nums, channel_freqs, bias_lines, optically_active, selected_channels = read_channel_file(channel_standard_file)
    
    #standard channels to use
    active_channels = np.argwhere(selected_channels == 1)
    print(active_channels)
    use_standard_channels = np.reshape(active_channels,(1,active_channels.shape[0]))[0]    
    
    #loading up tone init
    initial_lo_sweep=np.load(initial_lo_sweep_path) #find initial lo sweep file
    targeted_lo_sweep=np.load(targeted_lo_sweep_path) #find targeted sweep file
    tone_freqs=np.load(tone_freqs_path) #find tone freqs
    
    #converting standard channels to tone init channels
    match = match_freqs(tone_freqs,channel_freqs[use_standard_channels],dist=1e5)
    print(match)
    use_current_init_channels = np.array([item[0] for item in match])
    use_current_init_channels = use_current_init_channels[0:int(len(use_current_init_channels)/2)]
    print(use_current_init_channels)
    #print(use_current_init_channels)
    
    
    #frequencies of channels    
    current_freqs = channel_freqs[use_current_init_channels]
    
    #compute delay region
    print('looking for delay region')
    delay_region_start, delay_region_stop = find_freqs_cable_delay_subtraction(initial_lo_sweep,0.98,10000)
    print(f'start = {delay_region_start}')
    print(f'stop = {delay_region_stop}')
    
    #measure cable delay
    delays = measure_delay_test_given_freq(initial_lo_sweep,delay_region_start,delay_region_stop,plot=False)
    
    
    
    #remove cable delay
    targeted_lo_sweep_rm=remove_delay_live(targeted_lo_sweep,
                                           np.median(delays),
                                           channels=use_current_init_channels)
    
    #measure circle parameters
    calibration=measure_circle_live(targeted_lo_sweep_rm,
                                    tone_freqs,
                                    channels=use_current_init_channels) #finds circle center and initial phase for every channel
    
    
    
    return np.median(delays), calibration, use_current_init_channels, current_freqs


def demod_routine_live_stats_on_all(ts_file, channels, all_channel_freqs, delay, calibration, cadence = 3, f_sawtooth= 15, data_rate = 2**19, correct_phase_jumps = False, threshold = 0.3):

    print(f'all chan freqs: {all_channel_freqs}')
    
    fs=512e6/data_rate      
    truncate = int(cadence * fs)


    f = h5py.File(ts_file, "r", libver='latest', swmr=True)    
    
    ts_fr = np.asarray(f['/time_ordered_data/timestamp'][-truncate:])
    Is_fr = f['/time_ordered_data/adc_i'][:,-truncate:]
    Qs_fr = f['/time_ordered_data/adc_q'][:,-truncate:]

    ts_len = len(ts_fr)
    Is_len = Is_fr.shape[1]
    Qs_len = Qs_fr.shape[1]

    mlen = int(min([ts_len, Is_len, Qs_len]))

    ts_trunc = deepcopy(ts_fr[:mlen])
    Is_trunc = deepcopy(Is_fr[:][:mlen])
    Qs_trunc = deepcopy(Qs_fr[:][:mlen])

    print(len(Is_trunc))

    IQ_stream_rm=remove_delay_timestream(Is_trunc+1j*Qs_trunc,np.array(all_channel_freqs),delay)

    print(f'len IQ_stream_rm: {len(IQ_stream_rm)}')
    print(f'len calbiration: {len(calibration)}')

    data_cal=get_phase(IQ_stream_rm,calibration)

    t_start=0
    t_stop= int(cadence/2)*2
        
    t0 = np.array([mea_reset_t0(ts_fr[int(fs)*t_start:int(fs)*t_stop],
                                data_cal[current_channel,int(fs)*t_start:int(fs)*t_stop],
                                f_sawtooth,plot=False)
                   for current_channel in range(len(data_cal))])

    t0_med = np.nanmedian(t0)
    
    start_idx = find_nearest_idx(ts_fr-ts_fr[0], t0_med)

    ch_count = 0
    data_demods = np.array([])
    #now loop over just the channels we want to view
    for chan in channels:

        

        print(f'shape t plug: {ts_fr.shape}')
        print(f'shape data plug: {data_cal.shape}')

        t_demod, data_demod, reset_indices = demodulate_with_fft(t=ts_fr,
                                                                 sig=data_cal[chan],
                                                                 start_index=start_idx,                                                                      
                                                                 f_fr=f_sawtooth,
                                                                 phase_units='nPhi0',
                                                                 correct_phase_jumps=correct_phase_jumps,
                                                                 phase_jump_threshold=threshold,
                                                                 plot_demod = False,
                                                                 plot_demod_title=None,
                                                                 intermediate_plotting_limits=[None,None],
                                                                 plot_chunking_process = False,
                                                                 plot_fft = False,
                                                                 plot_fft_no_dc = False,
                                                                 plot_limited_fft = False,
                                                                 plot_fit = False,
                                                                 plot_vectors = False)
        
        
        print(f'len data_demod: {len(data_demod)}')
        print(f'shape data_demods: {data_demods.shape}')
        
        if ch_count == 0:
            data_demods = data_demod        
        else:
            data_demods = np.vstack([data_demods, np.array(data_demod)])
        
        t_demods = t_demod
        ch_count += 1

        print(f'data demods in loop: {data_demods}')
        
    return t_demods, data_demods




def demod_routine_live(ts_file, channels, channel_freqs, delay, calibration, cadence = 3, f_sawtooth= 15, data_rate = 2**19, correct_phase_jumps = False, threshold = 0.3):
    
    #ts_fr,Is_fr,Qs_fr=read_data_live(ts_file,
    #                                 channels=channels)
    fs=512e6/data_rate      
    truncate = int(cadence * fs)


    f = h5py.File(ts_file, "r", libver='latest', swmr=True)    
    
    
    ts_fr = np.asarray(f['/time_ordered_data/timestamp'][-truncate:])
    Is_fr = f['/time_ordered_data/adc_i'][:,-truncate:]
    Qs_fr = f['/time_ordered_data/adc_q'][:,-truncate:]
    
    #Is_fr = Is_fr_all[channels,:]
    #Qs_fr = Qs_fr_all[channels,:]
    
    #Is_fr = f['/time_ordered_data/adc_i'][channels,-truncate:]
    #Qs_fr = f['/time_ordered_data/adc_q'][channels,-truncate:]

    

    ts_len = len(ts_fr)
    Is_len = Is_fr.shape[1]
    Qs_len = Qs_fr.shape[1]

    print(f'ts len orig: {ts_fr.shape}')
    print(f'Is len orig: {Is_fr.shape}')
    print(f'Qs len orig: {Qs_fr.shape}')

    mlen = int(min([ts_len, Is_len, Qs_len]))
    print(f'mlen: {mlen}')

    #mlen = min(map(len, [ts_fr, Is_fr[0][:], Qs_fr[0][:]]))
    print(mlen)

    ts_trunc = deepcopy(ts_fr[:mlen])
    Is_trunc = deepcopy(Is_fr[:][:mlen])
    Qs_trunc = deepcopy(Qs_fr[:][:mlen])

    print(f't shape: {ts_trunc.shape}')
    print(f'I shape: {Is_trunc.shape}')
    print(f'Q shape: {Qs_trunc.shape}')

   
    #fs=512e6/data_rate  
    
    #truncate = int(cadence * fs)
    #ts_fr = np.arange(Is_fr.shape[1])/fs
    
    #stop_idx = np.min([len(ts_fr), Is_fr.shape[1], Qs_fr.shape[1]])
    #input(f'{len(ts_fr)}, {len(Is_fr)}, {len(Qs_fr)}')

    #ts_fr = ts_fr[-truncate:stop_idx]
    #input(ts_fr)

    #Is_fr = Is_fr[:][-truncate:stop_idx]
    #input(Is_fr)

    #Qs_fr = Qs_fr[:][-truncate:stop_idx]   
    #input(Qs_fr)

    IQ_stream_rm=remove_delay_timestream(Is_trunc+1j*Qs_trunc,channel_freqs,delay)

    print(f'IQ stream rm: {IQ_stream_rm.shape}')
    
    data_cal=get_phase(IQ_stream_rm,calibration) 

    print(f'data cal: {data_cal.shape}')  
        
    t_start=0
    t_stop= int(cadence/2)*2
        
    t0 = np.array([mea_reset_t0(ts_fr[int(fs)*t_start:int(fs)*t_stop],
                                data_cal[current_channel,int(fs)*t_start:int(fs)*t_stop],
                                f_sawtooth,plot=False)
                   for current_channel in range(len(data_cal))])

    t0_med = np.nanmedian(t0)
    
    start_idx = find_nearest_idx(ts_fr-ts_fr[0], t0_med)
    
    ch_count = 0
    data_demods = np.array([])
    #for chan in range(data_cal.shape[0]):
    for chan in channels:

        #print(f'len t to demod: {len(ts_fr)}')
        #print(f'len data to demod: {len(data_cal[chan])}')

        print(f'shape t plug: {ts_fr.shape}')
        print(f'shape data plug: {data_cal.shape}')

        #t_demod, data_demod = demodulate_with_template(t=ts_fr, sig = data_cal[chan], n_packets = 32, correct_phase_jumps = True, phase_jump_threshold=0.4)
        #t_demod, data_demod = demodulate_with_fft_internal_fr(t=ts_fr, sig = data_cal[chan], n_packets = 32, correct_phase_jumps = True, phase_jump_threshold=0.4)#,start_idx=start_idx)
        t_demod, data_demod, reset_indices = demodulate_with_fft(t=ts_fr,
                                                                 sig=data_cal[chan],
                                                                 start_index=start_idx,                                                                      
                                                                 f_fr=f_sawtooth,
                                                                 phase_units='nPhi0',
                                                                 correct_phase_jumps=correct_phase_jumps,
                                                                 phase_jump_threshold=threshold,
                                                                 plot_demod = False,
                                                                 plot_demod_title=None,
                                                                 intermediate_plotting_limits=[None,None],
                                                                 plot_chunking_process = False,
                                                                 plot_fft = False,
                                                                 plot_fft_no_dc = False,
                                                                 plot_limited_fft = False,
                                                                 plot_fit = False,
                                                                 plot_vectors = False)
        
        
        print(f'len data_demod: {len(data_demod)}')
        print(f'shape data_demods: {data_demods.shape}')




        #data_demod = data_demod[:150]
        #t_demod = t_demod[:150]
        
        if ch_count == 0:
            data_demods = data_demod        
        else:
            data_demods = np.vstack([data_demods, np.array(data_demod)])
        
        t_demods = t_demod
        ch_count += 1

        print(f'data demods in loop: {data_demods}')
        
    return t_demods, data_demods
    

def main():
    t, adc_i, adc_q = read_data('/home/user/Documents/AliCPT/ali_offline_demod/ALICPT_RDF_20231017100614.hd5')
    eta_array = 2*np.ones((1002,1))
    corrected_complex_data = apply_correction(eta_array, adc_i, adc_q)
    print(adc_i[22:25])
    print(adc_q[22:25])
    
    """
    t, sig = generate_science_signal(sig_type = 'sin',t_length=10,phase=np.pi/4)
    t, sig = scale_science_signal(t,sig,0.1)
    t, sig = apply_gaussian_noise(t,sig,0.00)
    t, ramp = generate_flux_ramp(t, plot_len=len(t)//200)
    t, full, no_sig = realistic_squid_modulation(t, sig, ramp, squid_lambda = 0.3, plot_len=len(t)//200)
    sampled_t, sampled_sig = sample_squid(t, full, ramp, plot_len = 400)
    slow_T, slow_TOD = demodulate(sampled_t, sampled_sig, 4, 40)
    plt.plot(t,sig)
    plt.plot(slow_T,slow_TOD,'.')
    """
    

if __name__ == "__main__":
    main()



"""
        elif ch_count == 1:
            len_data_demod = len(data_demod)
            if len_data_demod > len(data_demods):
                print('triggered 1')
                data_demod = data_demod[:len(data_demods)]
            elif len_data_demod < len(data_demods):
                diff = np.abs(len_data_demod - len(data_demods))
                nan_append = np.empty((1,diff))
                nan_append[:] = 0
                data_demod = np.append([data_demod,nan_append])
            data_demods = np.vstack([data_demods, np.array(data_demod)])
        else:
            len_data_demod = len(data_demod)
            if len_data_demod > data_demods.shape[1]:
                print('triggered 2')
                data_demod = data_demod[:data_demods.shape[1]]
            elif len_data_demod < data_demods.shape[1]:
                diff = np.abs(len_data_demod = data_demods.shape[1])
                nan_append = np.empty((1,diff))
                nan_append[:] = 0
                data_demod = np.append([data_demod, nan_append])
            data_demods = np.vstack([data_demods, np.array(data_demod)])
        """