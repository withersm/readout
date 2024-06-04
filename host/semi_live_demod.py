import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import time
import h5py
import glob
import os
import sys
sys.path.append('/home/matt/readout/host/')
import ali_offline_demod as dm

show_time = 30
cadence = 3
data_rate = 2**19
f_sawtooth = 65



list_of_files = glob.glob('/home/matt/ali_drive_mnt/time_streams/*.hd5') # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)

print(f'Live streaming: {latest_file}')

tone_init_file_split = latest_file.split('_')
tone_init_file = f'/home/matt/ali_drive_mnt/tone_initializations/{tone_init_file_split[-5]}_{tone_init_file_split[-4]}_{tone_init_file_split[-3]}'

if float(tone_init_file_split[-4]) == 4250: 
    channel_standard = '/home/matt/readout/host/channel_standards/LO_4250MHz.csv'
elif float(tone_init_file_split[-4]) == 4750: 
    channel_standard = '/home/matt/readout/host/channel_standards/LO_4750MHz.csv'
elif float(tone_init_file_split[-4]) == 5250: 
    channel_standard = '/home/matt/readout/host/channel_standards/LO_5250MHz.csv'
elif float(tone_init_file_split[-4]) == 5750: 
    channel_standard = '/home/matt/readout/host/channel_standards/LO_5750MHz.csv'
elif float(tone_init_file_split[-4]) == 6250: 
    channel_standard = '/home/matt/readout/host/channel_standards/LO_6250MHz.csv'
elif float(tone_init_file_split[-4]) == 6750: 
    channel_standard = '/home/matt/readout/host/channel_standards/LO_6750MHz.csv'

print(f'Setting up channels using:\nChannel standard: {channel_standard}\nTone init: {tone_init_file}')
delay, calibration, channels, channel_freqs = dm.channel_setup(tone_init_file+'/', 
                                                               channel_standard)

fig, ax = plt.subplots(1)
t_buffer = []
data_buffer = []
def animate(i):    

    #f = h5py.File(latest_file, "r", libver='latest', swmr=True)
    #tset = np.asarray(f['/time_ordered_data/timestamp'])
    #dset = f['/time_ordered_data/adc_i']

    t_demod, data_demod = dm.demod_routine_live(latest_file, 
                                                channels, 
                                                channel_freqs, 
                                                delay, 
                                                calibration, 
                                                cadence = cadence, 
                                                f_sawtooth= f_sawtooth, 
                                                data_rate = data_rate)
    
    t_buffer = np.append(t_buffer,t_demod)
    data_buffer = np.append(data_buffer, data_demod)
    
    ax.plot(t_buffer, data_buffer)

    # dump data from buffer if too long
    if len(t_buffer) >= show_time*512e6/data_rate:
        t_buffer = t_buffer[len(t_demod):]
        data_buffer = data_buffer[len(t_demod):]
    
    
        

try:
    ani = animation.FuncAnimation(fig, animate, interval=1000)
    plt.show()
except KeyboardInterrupt:
     print("Finishing...")
     exit()







"""

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    global data_cont_t
    global data_cont_v

    f = h5py.File(latest_file, "r", libver='latest', swmr=True)
    tset = np.asarray(f['/time_ordered_data/timestamp'])
    dset = f['/time_ordered_data/adc_i']

    
    ax1.clear()
    ax1.plot(tset, dset[30])

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()

"""