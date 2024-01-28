import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.formatter.useoffset'] = False
import matplotlib.pyplot as plt
import scipy
from scipy.signal import sawtooth, square
import pandas as pd
import glob as gl
import os
import cmath

from scipy.signal import sawtooth, square,find_peaks
from scipy import spatial
import lambdafit as lf
from scipy.interpolate import CubicSpline
import h5py


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
    print(division_factor)
    
    sampled_t = t[0:len(t):division_factor]
    print(len(sampled_t))
    sampled_signal = sig[0:len(sig):division_factor]
    print(len(sampled_signal))
    
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
    print(corrected_complex_data[22:25])
    
    return corrected_complex_data

def extract_science_signal(corrected_complex_data):
    #demodulate each channel, return an array with a row of demod data for each channel
    pass


def demodulate(t, sig, n_Phi0, f_sawtooth, plot = True, plot_len = None):
    print(len(sig))
    print(t[len(t)-1])
    chunksize = len(sig) / t[len(t)-1] / f_sawtooth
    n_chunks = int(len(t)//chunksize)
    
    print(n_chunks)

    #sig -= (max(sig)+min(sig))/2
    
    #print(2*np.pi*n_Phi0*f_sawtooth)
    
    slow_t = np.full(shape=n_chunks, dtype=float, fill_value=np.nan)
    slow_TOD = np.full(shape=n_chunks, dtype=float, fill_value=np.nan)
    for ichunk in range(n_chunks):
        #print(ichunk)
        """
        if ichunk == 10:#ichunk == 4 or ichunk ==5 or ichunk == 9 or ichunk == 10:
            print('at 4')
            continue
        """
        start = int(ichunk*chunksize)
        stop = int((ichunk+1)*chunksize)
        #print(len(sampled_signal[start:stop]))
        #print(stop)
        num = np.sum(sig[start:stop]*np.sin(2*np.pi*n_Phi0*f_sawtooth*(t[start:stop]-t[start])))
        den = np.sum(sig[start:stop]*np.cos(2*np.pi*n_Phi0*f_sawtooth*(t[start:stop]-t[start])))
        slow_TOD[ichunk] = np.arctan2(num, den)
        #print(slow_TOD[ichunk])
        slow_t[ichunk] = t[(start+stop)//2]
        print(ichunk)
        print(slow_t[ichunk])
        #print(slow_TOD)
    
    
    #slow_t = slow_t[~np.isnan(slow_TOD)]
    #slow_TOD = np.unwrap(slow_TOD[~np.isnan(slow_TOD)])
    slow_TOD /= 2*np.pi # convert to Phi0
    slow_TOD -= np.average(slow_TOD) # DC subtract
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
    peaks_neg,_=find_peaks(0-delta_fs_ch, distance=10)
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
        n_phi0 = mea_nphi0(time,data_cal[i,:],f_sawtooth,plot=plot)
        n_phi0_array = np.append(n_phi0_array, n_phi0)

    n_phi0 = np.median(n_phi0_array)
    
    return n_phi0

def mea_reset_t0(times,delta_fs_ch,reset_freq,plot=False):
    peaks_pos_flag,_=find_peaks(delta_fs_ch, distance=10,height=(np.min(delta_fs_ch)*5/6.,np.max(delta_fs_ch)*5/6.))
    peaks_neg_flag,_=find_peaks(delta_fs_ch, distance=10,height=(-np.max(delta_fs_ch)*5/6.,np.min(delta_fs_ch)*5/6.))
    peaks_flag=np.concatenate((peaks_pos_flag,peaks_neg_flag))
    peaks_flag.sort()
    t_flag_raw=times[peaks_flag]
    t_flag_raw_reset=(t_flag_raw-times[0])*reset_freq
    t0s=t_flag_raw-times[0]-t_flag_raw_reset.astype(int)/reset_freq
    if plot==True:
        plt.plot(times-times[0],delta_fs_ch)
        plt.scatter(times[peaks_flag]-times[0],delta_fs_ch[peaks_flag],color='r')
        plt.xlim(0,1)
        plt.show()
    return np.median(t0s)

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
def plot_s21(lo_data):
    plt.figure(figsize=(14,8))
    for current_data in lo_data:
        ftones = np.concatenate(current_data[0])
        sweep_Z = np.concatenate(current_data[1])
        
        mag = 20* np.log10(np.abs(sweep_Z))
        
        plt.plot(ftones, mag.real, '-',alpha=1)
        
    plt.grid()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('$|S_{21}|$')
    plt.show()
            
def plot_timestream(time, data, start_time = None, end_time = None, channel_nums = [0]):
    
    plt.figure(figsize=(11,8))
    for current_channel in channel_nums:
        plt.plot(time-time[0], data[current_channel], label = current_channel)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Phase (rad.)')
    plt.xlim([start_time,end_time])    
    plt.legend()
    plt.show()

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
