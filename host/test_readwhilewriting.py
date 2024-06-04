import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import time
import h5py
import glob
import os

list_of_files = glob.glob('/home/matt/ali_drive_mnt/time_streams/*.hd5') # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)

#fname = 'ALICPT_RDF_20231103180251.hd5'

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    global data_cont_t
    global data_cont_v

    f = h5py.File(latest_file, "r", libver='latest', swmr=True)
    tset = np.asarray(f['/time_ordered_data/timestamp'])
    dset = f['/time_ordered_data/adc_i']

    
    ax1.clear()
    ax1.plot(dset[30,:])

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()


