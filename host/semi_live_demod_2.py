import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sys
sys.path.append('/home/matt/readout/host')
import ali_offline_demod as dm
import glob
import h5py
import os
import pandas as pd

#Input variables
fs = 256e6 / 2**19 #Hz
f_sawtooth = 15 #Hz
length = 10 #s
correct_phase_jumps = True
phase_jump_threshold = 0.3


#Find the most recent file
list_of_files = glob.glob('/home/matt/ali_drive_mnt/time_streams/*.hd5') # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)

print(f'Live streaming: {latest_file}')

tone_init_file_split = latest_file.split('_')
tone_init_file = f'/home/matt/ali_drive_mnt/tone_initializations/{tone_init_file_split[-5]}_{tone_init_file_split[-4]}_{tone_init_file_split[-3]}'
print(tone_init_file)

#Pull associated tone init
initial_lo_sweep_path = dm.find_file(tone_init_file+'/', 'lo_sweep_initial')
targeted_lo_sweep_path = dm.find_file(tone_init_file+'/', 'lo_sweep_targeted_2')
tone_freqs_path = dm.find_file(tone_init_file+'/', 'freq_list_lo_sweep_targeted_1')

initial_lo_sweep=np.load(initial_lo_sweep_path) #find initial lo sweep file
targeted_lo_sweep=np.load(targeted_lo_sweep_path) #find targeted sweep file
tone_freqs=np.load(tone_freqs_path) #find tone freqs

#Find delay region
print('looking for delay region')
delay_region_start, delay_region_stop = dm.find_freqs_cable_delay_subtraction(initial_lo_sweep,0.98,10000)
print(f'start = {delay_region_start}')
print(f'stop = {delay_region_stop}')

#measure cable delay
delays = dm.measure_delay_test_given_freq(initial_lo_sweep,delay_region_start,delay_region_stop,plot=False)

#remove cable delay
targeted_lo_sweep_rm = dm.remove_delay(targeted_lo_sweep,
                                       np.median(delays),
                                       channels='all')

#compute calibration
calibration=dm.measure_circle_allch(targeted_lo_sweep_rm,
                                    tone_freqs,
                                    channels='all') #finds circle center and initial phase for every channel

#read channel standard
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

print(f'Matching up channels using:\nChannel standard: {channel_standard}\nTone init: {tone_init_file}')

#Read channel standard
channel_std_df = pd.read_csv(channel_standard, sep=',')
selected_standard_chs = channel_std_df['Selected'].to_numpy(int)
standard_freqs = channel_std_df['Frequency'].to_numpy(float)

#Find indices of requested standard channels
use_standard_idxs = np.argwhere(selected_standard_chs == 1)
use_standard_idxs = np.reshape(use_standard_idxs, (1,len(use_standard_idxs)))[0]

#Find freqs of requested standard channels
use_standard_freqs = standard_freqs[use_standard_idxs]

#Match these indices to indices from tone init
ch_match = dm.match_freqs(tone_freqs,use_standard_freqs,dist=1e5)
ch_match = ch_match[:int(len(ch_match)/2)]

print(ch_match)

use_current_channels = np.array([i[0] for i in ch_match])

#Declare plot
fig, ax = plt.subplots(1)

#find start time for time referencing
f_start = h5py.File(latest_file, "r", libver='latest', swmr=True)
start_time = np.asarray(f_start['/time_ordered_data/timestamp'][0])
f_start.close()

#demoding loop
def animate(i):
    #global latest_file

    start_time_count = 0
    
    f = h5py.File(latest_file, "r", libver='latest', swmr=True)
    
    ts_fr = np.asarray(f['/time_ordered_data/timestamp'][-488*10:])
    Is_fr = f['/time_ordered_data/adc_i'][:,-488*10:]
    Qs_fr = f['/time_ordered_data/adc_q'][:,-488*10:]

    

    print(f'Is_fr: {Is_fr[30]}')

    t_demod, data_demod_stacked = dm.full_demod_process_live(ts_fr,
                                                             Is_fr,
                                                             Qs_fr,
                                                             fs,
                                                             f_sawtooth,
                                                             use_current_channels,
                                                             length,
                                                             correct_phase_jumps,
                                                             phase_jump_threshold,
                                                             delays,
                                                             calibration,
                                                             tone_freqs,
                                                             start_time)
    ax.cla()
    for channel in range(len(data_demod_stacked)):
        ax.plot(t_demod, data_demod_stacked[channel] - np.average(data_demod_stacked[channel]),alpha=0.5)

    ax.set_ylim([-0.05,0.05])
    ax.set_xlabel('$t$ (s)')
    ax.set_ylabel('Phase ($N_{\\Phi_0}$)')


ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()