# flow for beam map time chunking demod 

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


# ----- define files (integrate into actual function later)

ts_f = '/home/matt/ali_drive_mnt/beam_map_data/toneinit_fcenter_4250.0_20240308061355_t_20240308062133/ts_toneinit_fcenter_4250.0_20240308061355_t_20240308062147.hd5'
times_xy = '/home/matt/ali_drive_mnt/beam_map_data/toneinit_fcenter_4250.0_20240308061355_t_20240308062133/beam_map_data_20240308062147.txt'
t_xy = pd.read_csv(times_xy, sep=',') # beam map x,y time data raw (starts and ends need correction)
t_xy

# there's a space before "end, x, and y" in the column names, correct this for ease
t_xy = t_xy.rename(columns={' end':'end', ' x': 'x', ' y': 'y'}) 




# ----- rearranging x-y time table ---------

# pull out arrays
start_1 = np.array(t_xy['start'])
stop_1 = np.array(t_xy['end'])
# delete first 
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


# ----- Reading Timestream File --------

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
        ch = np.array([channel - 22 for channel in buffer_range_fixed[chunk_start:chunk_stop]])
        t = np.array(file['time_ordered_data']['timestamp'])
        adc_i = np.array([file['time_ordered_data']['adc_i'][channel] for channel in buffer_range_fixed[chunk_start:chunk_stop]])
        adc_q = np.array([file['time_ordered_data']['adc_q'][channel] for channel in buffer_range_fixed[chunk_start:chunk_stop]])
        
    elif chunk == 'single':
        ch = np.array(buffer_range_fixed[single_channel] - 22)
        t = np.array(file['time_ordered_data']['timestamp'])
        adc_i = np.array(file['time_ordered_data']['adc_i'][buffer_range_fixed[single_channel]])
        adc_q = np.array(file['time_ordered_data']['adc_q'][buffer_range_fixed[single_channel]])
                              
    return t, adc_i, adc_q, ch, file # this function will now have 4 outputs instead of 3


# for now, just going to separate them out, later on might be better to have it return a dictionary
t, i, q, ch, file = read_data(ts_f,chunk='some', chunk_start=10, chunk_stop=12)

# array of (t, i, q) for both channels
ch_10 = np.asarray((t, (i[0] + 1j*q[0])))
ch_11 = np.asarray((t, (i[1] + 1j*q[1]))) 


# ------- get timestream index by comparing xy-times -----

# baby steps: just extract the times first

# operating on the pandas frame, can change later? 
# note: trying this out, need to input a much shorter table .... ?

def get_timestream_chunk_idx_ALL(t_ts, t_xy_table): # t_xy_starts, t_xy_ends):

    t_ts = np.asarray(t_ts) 

    ts_idxs = []

    for i in range(len(t_xy_table)):

        t_xy_start = t_xy_table['start'][i]
        t_xy_end = t_xy_table['end'][i]

        start_range = t_ts[t_ts >= t_xy_start]
        end_range = t_ts[t_ts <= t_xy_end]

        ts_start_range_idx = (np.abs(start_range - t_xy_start)).argmin()
        ts_stop_range_idx = (np.abs(end_range - t_xy_end)).argmin()

        ts_start_idx = np.where(t_ts == start_range[ts_start_range_idx])[0][0]
        ts_end_idx = np.where(t_ts == end_range[ts_stop_range_idx])[0][0]

        ts_idxs.append([ts_start_idx, ts_end_idx])

    ts_idxs = np.asarray(ts_idxs)

    return ts_idxs

# apply index finding function
idxs_test = get_timestream_chunk_idx_ALL(t, t_xy_new.iloc[0:11])


# ----- actually extract signal chunk with indices -----

# the ch_ts here is a combined (t, (I + jQ)) array

def get_time_chunked_signal(ch_ts, t_xy_table, ts_idxs): 

    t_ts = ch_ts[0]
    iq_ts = ch_ts[1]

    chunk_idx_start = ts_idxs[:,0]
    chunk_idx_end = ts_idxs[:,1]
    
    ts_chunks = []

    for i in range(len(ts_idxs)):

        t_idxed = t_ts[chunk_idx_start[i]:chunk_idx_end[i]]
        sig_idxed = iq_ts[chunk_idx_start[i]:chunk_idx_end[i]]

        #t_sig_chunked = ch_ts[:, chunk_idx_start[i]:chunk_idx_end[i]]

        ts_chunks.append([t_idxed, sig_idxed])
        

    x_map = np.asarray(t_xy_table['x'])
    y_map = np.asarray(t_xy_table['y']) 
    ts_chunks = np.asarray(ts_chunks)

    return x_map, y_map, ts_chunks

# apply function: 

x, y, ts_chunked = get_time_chunked_signal(ch_10, t_xy_new.loc[0:11], idxs_test)