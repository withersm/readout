import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import pandas as pd
import time

def find_minima(sweep_file, peak_prominence = 2, plot=False):
    
    data = np.load(sweep_file)

    ftones = np.concatenate(data[0])
    sweep_Z = np.concatenate(data[1])

    mag = 20* np.log10(np.abs(sweep_Z))

    maximum=max(mag)

    troughs, _= find_peaks(-mag,height=[-maximum,0],prominence=peak_prominence)

    if plot == True:
        plt.plot(ftones, mag.real,'-')
        plt.plot(ftones[troughs], mag[troughs].real,'.')
        plt.xlabel("Frequency (GHz)")
        filename_split = sweep_file.split("_")
        plt.savefig(f'./frequency_lists/png/res_loc_fcenter_{filename_split[-2]}_{int(time.time())}.png',dpi=150)
            
    return(ftones[troughs], mag[troughs].real)

def find_targeted_minima(sweep_file):

    data = np.load(sweep_file)
    freq = np.zeros(data.shape[1])

    for i in range(data.shape[1]):
        data_i = data[:,i,:]

        index = np.argmin(np.abs(data_i[1,:]))
        freq[i] = data_i[0,index].real  
    
    return freq  

def save_array(array, name):
    np.save(name, array)
    
def load_array(name):
    array = np.load(name)
    
    return array

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()

    return idx, array[idx]

def find_calibration(sweep_file, f0, delta_n, filename=None):
    data = np.load(sweep_file)

    ftones = np.concatenate(data[0])
    sweep_Z = np.concatenate(data[1])

    mag = 20 * np.log10(np.abs(sweep_Z))

    eta = np.array([])
    for freq in f0:
        #index =  np.where(ftones==freq)[0]
        index, _ = find_nearest(ftones, freq)               
        
        f_min = ftones[index-delta_n]
        f_max = ftones[index+delta_n]
        s21_min = sweep_Z[index-delta_n]
        s21_max = sweep_Z[index+delta_n]
        
        eta = np.append(eta, (f_max - f_min) / (s21_max - s21_min))
        print(eta)

    print(len(f0))
    print(len(eta))
        
    calibration = pd.DataFrame({'f0': f0.real, 'eta': eta})
    
    if filename != None:
        calibration.to_csv(filename,sep=',')
        
    return calibration