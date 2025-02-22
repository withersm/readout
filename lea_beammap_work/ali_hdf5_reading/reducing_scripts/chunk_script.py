# packages

import numpy as np
import pandas as pd 
import h5py
import argparse

# arguments for command line 
parser = argparse.ArgumentParser()
parser.add_argument('--filename', '-f', required=True, help='give filename')
parser.add_argument('--chunk', '-r', required=True, help='all, some, or single')
parser.add_argument('--chunk_start', '-c0', type=int, required=False, help='start of chunk if selecting range')
parser.add_argument('--chunk_stop', '-cf', type=int, required=False, help='end of chunk if selecting range')
parser.add_argument('--single_channel', '-c', type=int, required=False, help='single channel if selecting only one')

args = parser.parse_args()

# ------ Function ------

def read_hdf5_chunk_channels(filename, chunk='all', chunk_start=None, chunk_stop=None, single_channel=None):
    # read in file
    file = h5py.File(filename, 'r') 
    # pre-setting range for the for-loop iterating to fix the 0-22 rows of resonator buffer 
    buffer_range_fixed = range(22, (file['time_ordered_data']['adc_i'].shape[0])) 
    # create empty list
    chunk_list = []
    # iterate depending on chunk argument 
    if chunk == 'all': # for all channels
        for channel in buffer_range_fixed: 
            chunk_list.append(pd.DataFrame({'channel': channel - 22,
                                        't': np.array(file['time_ordered_data']['timestamp']), 
                                        'i': np.array(file['time_ordered_data']['adc_i'][channel]), 
                                        'q': np.array(file['time_ordered_data']['adc_q'][channel])}).rename_axis('Event #'))
    elif chunk == 'some': # for range of channels
        for channel in buffer_range_fixed[chunk_start:chunk_stop]:
            chunk_list.append(pd.DataFrame({'channel': channel - 22,
                                        't': np.array(file['time_ordered_data']['timestamp']), 
                                        'i': np.array(file['time_ordered_data']['adc_i'][channel]), 
                                        'q': np.array(file['time_ordered_data']['adc_q'][channel])}).rename_axis('Event #'))
    elif chunk == 'single': # for single channel
        channel = buffer_range_fixed[single_channel]
        chunk_list = pd.DataFrame({'channel': channel - 22,
                                    't': np.array(file['time_ordered_data']['timestamp']), 
                                    'i': np.array(file['time_ordered_data']['adc_i'][channel]), 
                                    'q': np.array(file['time_ordered_data']['adc_q'][channel])}).rename_axis('Event #')
    return chunk_list

# -------- Executing -------- 

if __name__ == "__main__":
    filename = args.filename
    single_channel = args.single_channel
    chunk = args.chunk
    chunk_start = args.chunk_start
    chunk_stop = args.chunk_stop
    single_channel = args.single_channel
    
    print(read_hdf5_chunk_channels(filename, chunk, chunk_start, chunk_stop, single_channel))
