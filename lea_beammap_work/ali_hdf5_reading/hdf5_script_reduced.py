import numpy as np
import matplotlib as mpl
import pandas as pd
import h5py
import argparse 
import os 
import subprocess


# adding arguments for command line 
parser = argparse.ArgumentParser()
parser.add_argument('--filename', '-f', required=True, help='give filename')
parser.add_argument('--single_channel', '-chan', type=int, required=True, help='single desired channel to chunk')
parser.add_argument('--time', '-t', choices=['all', 'some'], default='all', help='time chunked or not')
parser.add_argument('--t_index_start', '-t0', type=int, help='event index start')
parser.add_argument('--t_index_stop', '-t1', type=int, help='event index stop')

args = parser.parse_args()

# REDUCED VERSION OF THAT ^^
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

# Second function is creating Excel file and automatically opening it: 

def chunk_to_excel(dataframe, filename):
    excel_filename = filename.replace(".hd5", f"_CHANNEL_{args.single_channel}.xlsx")
    dataframe.to_excel(excel_filename, index=True)

    # Opening excel file 
    if os.name == 'nt':  # Windows
        os.startfile(excel_filename)
    elif os.name == 'posix':  # macOS or Linux
        subprocess.call(['open', excel_filename])  # macOS
        # subprocess.call(['xdg-open', excel_filename])  # Linux

# -------- Executing -------- 

if __name__ == "__main__":
    filename = args.filename
    single_channel = args.single_channel
    time = args.time
    t_index_start = args.t_index_start
    t_index_stop = args.t_index_stop

    chunked_data = read_chunk_isolate_channel_time_reduced(filename, single_channel, time, t_index_start, t_index_stop)
    print('data has been chunked')
    chunk_to_excel(chunked_data, filename)