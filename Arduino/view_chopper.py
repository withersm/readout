import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import time

fname = 'test_output.txt'


fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

data_cont_t = np.array([])
data_cont_v = np.array([])
def animate(i):
    global data_cont_t
    global data_cont_v

    n_rows_plot = 2000
    n_rows = sum(1 for row in open(fname, 'r'))
    data = pd.read_csv(fname, skiprows=n_rows-n_rows_plot, sep = ' ', names=['time','voltage'], dtype=float)
    
    #data_cont_t = np.append(data_cont_t,data['time'])
    #data_cont_v = np.append(data_cont_v,data['voltage'])
    #print(data_cont_t)
    
    ax1.clear()
    ax1.step(data['time'], data['voltage'])

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()


