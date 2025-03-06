# packages

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

from scipy.signal import sawtooth, square,find_peaks, savgol_filter
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

# for pandas visual number display 

pd.set_option('display.precision', 6)
pd.set_option('display.float_format', '{:.10f}'.format)

# import ali_offline_demod.py 

import ali_offline_demod as aod


# modified/new functions


#Functions for reading, processing, and demodulating real data
def read_data(filename, chunk='all', chunk_start=None, chunk_stop=None, single_channel=None):
    # read in file
    file = h5py.File(filename, 'r') 
    # pre-setting range for the for-loop iterating to fix the 0-22 rows of resonator buffer 
    buffer_range_fixed = range(22, (file['time_ordered_data']['adc_i'].shape[0])) 
    # iterate depending on chunk argument 
    if chunk == 'all':
        ch = np.array([channel - 22 for channel in buffer_range_fixed])
        t = np.array(file['time_ordered_data']['timestamp'])
        adc_i = np.array([file['time_ordered_data']['adc_i'][channel] for channel in buffer_range_fixed])
        adc_q = np.array([file['time_ordered_data']['adc_q'][channel] for channel in buffer_range_fixed])
        
    elif chunk == 'some': 
        ch = np.array([channel - 22 for channel in buffer_range_fixed[chunk_start:chunk_stop + 1]])
        t = np.array(file['time_ordered_data']['timestamp'])
        adc_i = np.array([file['time_ordered_data']['adc_i'][channel] for channel in buffer_range_fixed[chunk_start:chunk_stop + 1]])
        adc_q = np.array([file['time_ordered_data']['adc_q'][channel] for channel in buffer_range_fixed[chunk_start:chunk_stop + 1]])
        
    elif chunk == 'single':
        ch = np.array(buffer_range_fixed[single_channel] - 22)
        t = np.array(file['time_ordered_data']['timestamp'])
        adc_i = np.array(file['time_ordered_data']['adc_i'][buffer_range_fixed[single_channel]])
        adc_q = np.array(file['time_ordered_data']['adc_q'][buffer_range_fixed[single_channel]])
                              
    return t, adc_i, adc_q, ch, file # this function will now have 4 outputs instead of 3





# new functions

def load_fix_xy_txt(t_xy_file):
    t_xy = pd.read_csv(t_xy_file, sep=',') # beam map x,y time data raw (starts and ends need correction)

    # there's a space before "end, x, and y" in the column names, correct this for ease
    t_xy = t_xy.rename(columns={' end':'end', ' x': 'x', ' y': 'y'}) 

    # pull out arrays
    start_1 = np.array(t_xy['start'])
    stop_1 = np.array(t_xy['end'])

    # delete first value
    start_1_new = np.delete(start_1, 0)
    start_1_new = np.pad(start_1_new, (0, 1)) # keep lengths the same, add zero at end (FOR NOW)

    # confirm
    print(start_1[0:3])
    print(start_1_new[0:3])
    print(start_1[1]==start_1_new[0])

    # set new table with flipped columns (switch em)
    start_new = stop_1
    stop_new = start_1_new 
    t_xy_new = t_xy.copy()
    t_xy_new['start'] = start_new
    t_xy_new['end'] = stop_new

    return t_xy_new



def get_timestream_chunk_idx_ALL(t_ts, t_xy_table): # t_xy_starts, t_xy_ends):

    t_ts = np.asarray(t_ts) 

    ts_idxs = []

    for i in range(len(t_xy_table)): # iterate through each row (measurement window)
    
        # set start and end objects
        t_xy_start = t_xy_table['start'][i] 
        t_xy_end = t_xy_table['end'][i]

        # set sub-range so that chunk being indexed doesn't include time while mapper is moving
        start_range = t_ts[t_ts >= t_xy_start]
        end_range = t_ts[t_ts <= t_xy_end]

        # find index of min of aboslute difference between the sub-range of timestream time range and values from x-y mapper file
        ts_start_range_idx = (np.abs(start_range - t_xy_start)).argmin()
        ts_stop_range_idx = (np.abs(end_range - t_xy_end)).argmin()

        # get index of where actual timestream equals the value found previously
        ts_start_idx = np.where(t_ts == start_range[ts_start_range_idx])[0][0]
        ts_end_idx = np.where(t_ts == end_range[ts_stop_range_idx])[0][0]

        # append indices
        ts_idxs.append([ts_start_idx, ts_end_idx])

    ts_idxs = np.array(ts_idxs)

    return ts_idxs



def get_timestream_chunk_idx_w_time_indexing(t_ts, t_xy_table, t_chunk_index_start, t_chunk_index_stop): 

    t_ts = np.asarray(t_ts) 

    ts_idxs = []

    #for i in range(len(t_xy_table)): # iterate through each row (measurement window)
    
    for i in range(t_chunk_index_start, t_chunk_index_stop):
        # set start and end objects
        t_xy_start = t_xy_table['start'][i] 
        t_xy_end = t_xy_table['end'][i]

        # set sub-range so that chunk being indexed doesn't include time while mapper is moving
        start_range = t_ts[t_ts >= t_xy_start]
        end_range = t_ts[t_ts <= t_xy_end]

        # find index of min of aboslute difference between the sub-range of timestream time range and values from x-y mapper file
        ts_start_range_idx = (np.abs(start_range - t_xy_start)).argmin()
        ts_stop_range_idx = (np.abs(end_range - t_xy_end)).argmin()

        # get index of where actual timestream equals the value found previously
        ts_start_idx = np.where(t_ts == start_range[ts_start_range_idx])[0][0]
        ts_end_idx = np.where(t_ts == end_range[ts_stop_range_idx])[0][0]

        # append indices
        ts_idxs.append([ts_start_idx, ts_end_idx])

    ts_idxs = np.array(ts_idxs)

    return ts_idxs, t_xy_table.loc[t_chunk_index_start:t_chunk_index_stop]



def get_ts_chunks_WORKING(ts_filename, t_xy_file, chunk='some', chunk_start=None, chunk_stop=None, 
                          single_channel=None, time_chunk_index_stop=10):
    
    t, i_data, q_data, ch, file = read_data(ts_filename, chunk=chunk, chunk_start=chunk_start, chunk_stop=chunk_stop)
    

    print(len(i_data))
    print(i_data[0])

    t_xy_table = load_fix_xy_txt(t_xy_file)

    ts_indexes = get_timestream_chunk_idx_ALL(t, t_xy_table.loc[0:time_chunk_index_stop])

    time_chunks = []
    I_chunks_final = []
    Q_chunks_final = []
    
    for i in range(len(ts_indexes)): # index time chunks as i 

        idx_range = ts_indexes[i]
        t_idxed = t[idx_range[0]:idx_range[1]]

        time_chunks.append(t_idxed)

        I_chunk=[]
        Q_chunk=[]

        for j in range(len(i_data)): # individual channels as j 

            i_ts = i_data[j]
            q_ts = q_data[j]

            i_idxed = i_ts[idx_range[0]:idx_range[1]]
            q_idxed = q_ts[idx_range[0]:idx_range[1]]

            I_chunk.append(i_idxed)
            Q_chunk.append(q_idxed)


        I_chunks_final.append(I_chunk)
        Q_chunks_final.append(Q_chunk) 
    
    return time_chunks, I_chunks_final, Q_chunks_final


def get_ts_chunks_time_indexing(ts_filename, t_xy_file, chunk='some', chunk_start=None, chunk_stop=None, 
                                single_channel=None, time_chunk_index_start=0, time_chunk_index_stop=10):
    
    t, i_data, q_data, ch, file = read_data(ts_filename, chunk=chunk, chunk_start=chunk_start, chunk_stop=chunk_stop)
    

    print(len(i_data))
    print(i_data[0])

    t_xy_table = load_fix_xy_txt(t_xy_file)

    ts_indexes, txy_idxed_table = get_timestream_chunk_idx_w_time_indexing(t, t_xy_table, t_chunk_index_start = time_chunk_index_start, 
                                                                           t_chunk_index_stop = time_chunk_index_stop)

    time_chunks = []
    I_chunks_final = []
    Q_chunks_final = []
    
    for i in range(len(ts_indexes)): # index time chunks as i 

        idx_range = ts_indexes[i]
        t_idxed = t[idx_range[0]:idx_range[1]]

        time_chunks.append(t_idxed)

        I_chunk=[]
        Q_chunk=[]

        for j in range(len(i_data)): # individual channels as j 

            i_ts = i_data[j]
            q_ts = q_data[j]

            i_idxed = i_ts[idx_range[0]:idx_range[1]]
            q_idxed = q_ts[idx_range[0]:idx_range[1]]

            I_chunk.append(i_idxed)
            Q_chunk.append(q_idxed)


        I_chunks_final.append(I_chunk)
        Q_chunks_final.append(Q_chunk) 
    
    return time_chunks, I_chunks_final, Q_chunks_final, txy_idxed_table
    

def modified_demod_process_for_tchunks(t_chunk, i_chunk, q_chunk, ts_file, f_sawtooth, method = 'fft', correct_phase_jumps = False, 
                       phase_jump_threshold = 0.4, n=0, channels='all',start_channel=0,stop_channel=1000,
                       tone_init_path = '/home/matt/alicpt_data/tone_initializations', 
                       ts_path = '/home/matt/alicpt_data/time_streams', display_mode = 'terminal'):
    
    #n is number of points blanked before and after fr reset; only used when method='simple'
    #unpack data -> eventually change so that you give the ts data path and the function finds the associated tone initialization files

    print('using full_demod_process')

    init_freq = ts_file.split('_')[3]
    print(init_freq)
    init_time = ts_file.split('_')[4]
    print(init_time)
    init_directory = f'{tone_init_path}/fcenter_{init_freq}_{init_time}/'
    print(init_directory)
    
    initial_lo_sweep_path = aod.find_file(init_directory, 'lo_sweep_initial')
    targeted_lo_sweep_path = aod.find_file(init_directory, 'lo_sweep_targeted_2')
    tone_freqs_path = aod.find_file(init_directory, 'freq_list_lo_sweep_targeted_1')
    ts_path = f'{ts_path}/{ts_file}'    
    
    initial_lo_sweep=np.load(initial_lo_sweep_path) #find initial lo sweep file
    targeted_lo_sweep=np.load(targeted_lo_sweep_path) #find targeted sweep file
    tone_freqs=np.load(tone_freqs_path) #find tone freqs
    #print(tone_freqs)

    if channels == 'some':
        tone_freqs = tone_freqs[start_channel:stop_channel + 1]
        print(tone_freqs)

    #ts_fr,Is_fr,Qs_fr=read_data(ts_path,channels=channels,start_channel=start_channel,stop_channel=stop_channel)    #note to self: limit tone_freqs to actively called channels; need to figure out channel numbering first
    
    # input for ts_chunk is just full 
    ts_fr = np.asarray(t_chunk)
    Is_fr = np.asarray(i_chunk)
    Qs_fr = np.asarray(q_chunk)

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
    delay_region_start, delay_region_stop = aod.find_freqs_cable_delay_subtraction(initial_lo_sweep,0.98,10000)
    print(f'start = {delay_region_start}')
    print(f'stop = {delay_region_stop}')
    
    #measure cable delay
    delays = aod.measure_delay_test_given_freq(initial_lo_sweep,delay_region_start,delay_region_stop,plot=False)
    
    print(f'delay: {np.median(delays)}')

    #remove cable delay
    targeted_lo_sweep_rm=aod.remove_delay(targeted_lo_sweep,
                                      np.median(delays),
                                      channels=channels,
                                      start_channel=start_channel,
                                      stop_channel=stop_channel + 1)
    
    IQ_stream_rm=aod.remove_delay_timestream(Is_fr+1j*Qs_fr,tone_freqs,np.median(delays))
    
    #measure circle parameters
    calibration=aod.measure_circle_allch(targeted_lo_sweep_rm,
                                     tone_freqs,
                                     channels=channels,
                                     start_channel=start_channel,
                                     stop_channel=stop_channel + 1) #finds circle center and initial phase for every channel
    
    print(calibration[0])
    #calibrate time stream
    data_cal=aod.get_phase(IQ_stream_rm,calibration)

    # fig_testing, ax_testing = plt.subplots(1)
    # for i in [1]:
    #     ax_testing.plot(data_cal[i])
    #     ax_testing.set_title('Calibration Test')
    
    #find nphi_0
    t_start=0
    t_stop=10

    n_phi0 = aod.find_n_phi0(ts_fr[488*t_start:488*t_stop],data_cal[:,488*t_start:488*t_stop],f_sawtooth,plot=False)  #discard the first few seconds
    print(f'n_phi0: {n_phi0}')
    
    #find t0
    t0_array = np.array([])
    for current_channel in range(len(data_cal)):
        t0 = aod.mea_reset_t0(ts_fr[488*t_start:488*t_stop],data_cal[current_channel,488*t_start:488*t_stop],f_sawtooth,plot=False)
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
    start_idx = aod.find_nearest_idx(ts_fr-ts_fr[0], t0_med)
    print(f'start index: {start_idx}')
    if display_mode == 'notebook':
        for chan in tqdm_notebook(range(data_cal.shape[0])):#np.arange(225,230,1):#range(data_cal.shape[0]):
            if method == 'iv':
                t_demod, data_demod = aod.demodulate_for_iv(ts_fr[start_idx:]-ts_fr[start_idx], data_cal[chan, start_idx:], n_phi0, 3,f_sawtooth)
            
                t_demods.append(t_demod)
                data_demod_unwrap=np.unwrap(data_demod,period=1)
                data_demods.append(data_demod_unwrap)
            if method == 'simple':
                t_demod, data_demod, reset_indices = aod.demodulate(ts_fr[start_idx:]-ts_fr[start_idx],
                                                                data_cal[chan, start_idx:],
                                                                n_phi0,
                                                                n,
                                                                f_sawtooth)
                t_demods.append(t_demod)
                data_demod_unwrap=np.unwrap(data_demod,period=1)
                data_demods.append(data_demod_unwrap)
            if method == 'fft':
                t_demod, data_demod, reset_indices = aod.demodulate_with_fft(t=ts_fr,
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
                t_demod, data_demod, reset_indices = aod.demodulate(ts_fr[start_idx:]-ts_fr[start_idx],
                                                                data_cal[chan, start_idx:],
                                                                n_phi0,
                                                                n,
                                                                f_sawtooth)
                t_demods.append(t_demod)
                data_demod_unwrap=np.unwrap(data_demod,period=1)
                data_demods.append(data_demod_unwrap)
            if method == 'fft':
                t_demod, data_demod, reset_indices = aod.demodulate_with_fft(t=ts_fr,
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