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
# import lambdafit as lf
from scipy.interpolate import CubicSpline,interp1d
import h5py

from tqdm import tqdm as tqdm_terminal
from tqdm.notebook import trange, tqdm_notebook
from scipy.signal.windows import hann

from scipy.fft import fft, ifft, fftfreq
from copy import deepcopy
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import curve_fit


# ----------- FUNCTIONS ---------------

# FUNCTION 1: 
    # Read in .hd5 file 
    # store each channel's [t, i, q] data in separate dataframes 
    # store all channels' dataframes in one dictionary
    # returns the dictionary

def read_chunk_channels(filename):# single_channel, time='all', t_start=0, t_stop=28000):
    file = h5py.File(filename, 'r')
    # make arrays 
    i = np.array(file['time_ordered_data']['adc_i'])
    i = np.delete(i, slice(0,22), 0)
    q = np.array(file['time_ordered_data']['adc_q'])
    q = np.delete(q, slice(0,22), 0)
    t = np.array(file['time_ordered_data']['timestamp'])
    
    #initialize channel data dictionary
    channel_dict = {}

    # create structured array [t, i, q] for each channel
    for channel in range(i.shape[0]):
        # then store each channel as dataframe
        freqs_frame = pd.DataFrame({'t':t, 
                                    'i': i[channel], 
                                    'q': q[channel]
                                    })
        # store channel n in the greater dictionary
        channel_dict[channel] = freqs_frame
    
    return channel_dict




# FUNCTION 2: 
    # read file 
    # chunk channels into their own dataframes 
    # isolate desired channel
    # returns channel chunk as dataframe (just no time/event index component)

def read_chunk_isolate_channel(filename, single_channel):
    file = h5py.File(filename, 'r')
    # make arrays 
    i = np.array(file['time_ordered_data']['adc_i'])
    i = np.delete(i, slice(0,22), 0)
    q = np.array(file['time_ordered_data']['adc_q'])
    q = np.delete(q, slice(0,22), 0)
    t = np.array(file['time_ordered_data']['timestamp'])
    
    #initialize channel data dictionary
    channel_dict = {}

    # create structured array [t, i, q] for each channel
    for channel in range(i.shape[0]):
        # then store each channel as dataframe
        freqs_frame = pd.DataFrame({'t':t, 
                                    'i': i[channel], 
                                    'q': q[channel]
                                    })
        # store channel n in the greater dictionary
        channel_dict[channel] = freqs_frame
    
    selected_channel = channel_dict.get(single_channel)
    if selected_channel is None: 
        raise ValueError(f"channel {single_channel} not found")

    selected_channel = selected_channel.rename_axis('Event')

    return selected_channel



# FUNCTION 3: 
    # read file 
    # chunk channels into their own dataframes 
    # isolate desired channel
    # isolate time chunk within isolated channel 
    # returns channel and time chunk as Dataframe

def read_chunk_isolate_channel_time(filename, single_channel, time='all', t_index_start=None, t_index_stop=None):
    file = h5py.File(filename, 'r')
    # make arrays 
    i = np.array(file['time_ordered_data']['adc_i'])
    i = np.delete(i, slice(0,22), 0)
    q = np.array(file['time_ordered_data']['adc_q'])
    q = np.delete(q, slice(0,22), 0)
    t = np.array(file['time_ordered_data']['timestamp'])
    
    #initialize channel data dictionary
    channel_dict = {}

    # create structured array [t, i, q] for each channel
    for channel in range(i.shape[0]):
        # then store each channel as dataframe
        freqs_frame = pd.DataFrame({'t':t, 
                                    'i': i[channel], 
                                    'q': q[channel]
                                    })
        # store channel n in the greater dictionary
        channel_dict[channel] = freqs_frame
    
    # get desired channel
    selected_channel = channel_dict.get(single_channel)
    if selected_channel is None: 
        raise ValueError(f"channel {single_channel} not found")
    selected_channel = selected_channel.rename_axis('Event')

    # getting desired time chunk range (all or specific)
    if time == 'all':
        chunked_channel_time = selected_channel
    elif time == 'some':
        if t_index_start is None or t_index_stop is None: 
            raise ValueError('provide slice boundaries for time = some')
        chunked_channel_time = selected_channel.iloc[t_index_start:t_index_stop]

    return chunked_channel_time

## REDUCED version of that ^^ 

def read_chunk_isolate_channel_time_reduced_pd(filename, single_channel, time='all', t_index_start=None, t_index_stop=None):
    file = h5py.File(filename, 'r')
    # make arrays 
    i = np.array(file['time_ordered_data']['adc_i'])
    i = np.delete(i, slice(0,22), 0)
    q = np.array(file['time_ordered_data']['adc_q'])
    q = np.delete(q, slice(0,22), 0)
    t = np.array(file['time_ordered_data']['timestamp'])
    
    if single_channel >= i.shape[0]:
        raise ValueError(f"channel {single_channel} not found")
    
    i_data = i[single_channel]
    q_data = q[single_channel]

    selected_channel_frame = pd.DataFrame({'t': t, 
                                           'i': i_data,
                                           'q': q_data
                                           })

    selected_channel_frame = selected_channel_frame.rename_axis('Event')

    # getting desired time chunk range (all or specific)
    if time == 'all':
        chunked_channel_time = selected_channel_frame
    elif time == 'some':
        if t_index_start is None or t_index_stop is None: 
            raise ValueError('provide slice boundaries for time = some')
        chunked_channel_time = selected_channel_frame.iloc[t_index_start:t_index_stop]

    return chunked_channel_time



# ---------- with Numpy -----------

def chunk_channel_time_np_reduced(filename, single_channel, time='all', t_index_start=None, t_index_stop=None):
    file = h5py.File(filename, 'r')

    # make arrays 
    i = np.array(file['time_ordered_data']['adc_i'])
    i = np.delete(i, slice(0,22), 0)
    q = np.array(file['time_ordered_data']['adc_q'])
    q = np.delete(q, slice(0,22), 0)
    t = np.array(file['time_ordered_data']['timestamp'])

    single_channel_data = np.column_stack((t, i[single_channel], q[single_channel]))
    
     # getting desired time chunk range (all or specific)
    if time == 'all':
        chunked_channel_time = single_channel_data
    elif time == 'some':
        if t_index_start is None or t_index_stop is None: 
            raise ValueError('provide slice boundaries for time = some')
        chunked_channel_time = single_channel_data[t_index_start:t_index_stop,:]

    return chunked_channel_time
