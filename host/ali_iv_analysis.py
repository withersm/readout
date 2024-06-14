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
import lambdafit as lf
from scipy.interpolate import CubicSpline,interp1d
import h5py
from tqdm.notebook import trange, tqdm
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
def read_data(filename):
    file = h5py.File(filename, 'r')
    adc_i = np.array(file['time_ordered_data']['adc_i'])
    adc_i = np.delete(adc_i, slice(0,22), 0)
    adc_q = file['time_ordered_data']['adc_q']
    adc_q = np.delete(adc_q, slice(0,22), 0)
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
def full_demod_routine():
    pass
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
def remove_delay(target_sweeps,delay):
    target_sweeps_rm=target_sweeps.copy()
    for i in range(target_sweeps.shape[1]):
        target_sweep=target_sweeps[:,i,:]
        freqs=target_sweep[0,:]
        delay_fac=np.exp(1j*delay*2*np.pi*freqs)
        target_sweeps_rm[1,i,:]=target_sweeps[1,i,:]*delay_fac
    return target_sweeps_rm
def remove_delay_timestream(stream,f0s,delay):
    stream_rm=np.zeros((f0s.shape[0],stream.shape[1]),dtype = 'complex_')
    for i in range(f0s.shape[0]):
        delay_fac=np.exp(1j*2*np.pi*f0s[i]*delay)
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
def measure_circle_allch(target_sweeps,f0s):
    cals=[]
    for i in range(target_sweeps.shape[1]):
        sweep=target_sweeps[:,i,:]
        cal=measure_circle(sweep,f0s[i])
        cals.append(cal)
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
   
    return f_start, f_end
def full_demod_process(ts_file, f_sawtooth, tone_init_path = '/home/matt/alicpt_data/tone_initializations', ts_path = '/home/matt/alicpt_data/time_streams'):
    #unpack data -> eventually change so that you give the ts data path and the function finds the associated tone initialization files
    
    init_freq = ts_file.split('_')[3]
    init_time = ts_file.split('_')[4]    
    init_directory = f'{tone_init_path}/fcenter_{init_freq}_{init_time}/'
    
    initial_lo_sweep_path = find_file(init_directory, 'lo_sweep_initial')
    targeted_lo_sweep_path = find_file(init_directory, 'lo_sweep_targeted_2')
    tone_freqs_path = find_file(init_directory, 'freq_list_lo_sweep_targeted_1')
    ts_path = f'{ts_path}/{ts_file}'    
    
    initial_lo_sweep=np.load(initial_lo_sweep_path) #find initial lo sweep file
    targeted_lo_sweep=np.load(targeted_lo_sweep_path) #find targeted sweep file
    tone_freqs=np.load(tone_freqs_path) #find tone freqs
    ts_fr,Is_fr,Qs_fr=read_data(ts_path)    

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
    
    #remove cable delay
    targeted_lo_sweep_rm=remove_delay(targeted_lo_sweep,np.median(delays))
    IQ_stream_rm=remove_delay_timestream(Is_fr+1j*Qs_fr,tone_freqs,np.median(delays))
    
    #measure circle parameters
    calibration=measure_circle_allch(targeted_lo_sweep_rm,tone_freqs) #finds circle center and initial phase for every channel
    
    #calibrate time stream
    data_cal=get_phase(IQ_stream_rm,calibration)
    
    #find nphi_0
    t_start=0
    t_stop=10
    n_phi0 = find_n_phi0(ts_fr[488*t_start:488*t_stop],data_cal[:,488*t_start:488*t_stop],f_sawtooth,plot=False)  #discard the first few seconds
    
    #find t0
    t0_array = np.array([])
    for current_channel in range(len(data_cal)):
        t0 = mea_reset_t0(ts_fr[488*t_start:488*t_stop],data_cal[current_channel,488*t_start:488*t_stop],f_sawtooth,plot=False)
        t0_array = np.append(t0_array,t0)
    t0_med = np.nanmedian(t0_array)
    
    #demod
    
    t_demods=[]
    data_demods=[]
    start_idx = find_nearest_idx(ts_fr-ts_fr[0], t0_med)
    for chan in tqdm(range(data_cal.shape[0])):#np.arange(225,230,1):#range(data_cal.shape[0]):
        t_demod, data_demod = demodulate(ts_fr[start_idx:]-ts_fr[start_idx], data_cal[chan, start_idx:], n_phi0, 3,f_sawtooth)
        t_demods.append(t_demod)
        data_demod_unwrap=np.unwrap(data_demod,period=1)
        data_demods.append(data_demod_unwrap)
    data_demods=np.vstack(data_demods)
    t_demods=np.vstack(t_demods)
    
    data_dict = {'fr t': ts_fr, 'nphi': n_phi0, 't0': t0_med, 'demod t': t_demods[1], 'demod data': data_demods, 'channel freqs': tone_freqs, 'fsawtooth': f_sawtooth}
    
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
    f = interp1d(xold,array_sel)
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
    if len(peaks_nb)==0 or peaks_nb[0]<30:
        return np.zeros(resps.shape[0])
    peak_nb=peaks_nb[0]
    #here we know that before peaks_nb[0] and after ind_sc the Ites is monotonic 
    resps_nb=resps[2:int(peak_nb-10)]
    resps_sc=resps[int(ind_sc+10):-10]
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
    print(resps_corr)
    print(ind_sc)
    return resps_corr, ind_sc

def IV_analysis_ch_duo(bias_currents,resps,Rsh=0.4,filter_Rn_Al=False,plot='None'):
    """
    This method correct the SC and normal region for jump phi0 issue, and then fit for TES parameters
    Outputs:
    dataframe containing timestream of Ites,Rtes,Vtes,Rn_almn,Rn_al
    """ 
    peaks_nb,_=find_peaks(0-resps,width=20)
    max_ites=np.max(resps)
    min_ites=np.min(resps)
    if len(peaks_nb)==0 or peaks_nb[0] < 20 or max_ites-min_ites<20:
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
        resps_nb=resps_correct[2:int(peak_nb-10)]
        resps_sc=resps_correct[int(peak_sc+10):-10]
        bias_nb=bias_currents[2:peak_nb-10]
        bias_sc=bias_currents[int(peak_sc+10):-10]
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


def full_iv_process(iv_file,f_sawtooth,Rsh=0.4,iv_path = '/home/matt/alicpt_data/IV_data', filter_Rn_Al=False, plot='None'):
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
    for ch in tqdm(range(data_demods_bin.shape[0])):
        Rn_almn_ch,Rn_al_ch,Rtes_ch,Vtes_ch,Ites_ch,bps_ch,Pbias_ch,resps_correct_ch = IV_analysis_ch_duo(bias_currents[:,1],data_demods_bin[ch],Rsh=0.4,filter_Rn_Al=filter_Rn_Al, plot=plot)
        
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
                 'resps correct':resps_correct_list,
                 'demod data': demod_data}
       
    return data_dict
def get_channel_response_summary(IV_analysis_result,filepath=None):
    active_channel_chart = pd.DataFrame({'Channel Freq (Hz)': IV_analysis_result['demod data']['channel freqs'].real, 'Rn Al (mOhm)': IV_analysis_result['Rn Al']})
    if filepath != None:
        active_channel_chart.to_csv(filepath,sep=',')
    return active_channel_chart
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
