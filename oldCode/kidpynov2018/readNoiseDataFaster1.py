# Read dirfiles that are produced by kidPy_nov2018 
# Run this script in the noise dirfile directory
from readDirfile import * 
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from scipy import signal
import Tkinter
from Tkinter import *
#Variables 
global dirfiles2
global values
dirfiles2 = []
values = []
#GUI Frame Setup
root = Tkinter.Tk()
root.geometry('800x450')
#dirfolders = glob.glob("dirfiles/test4*") # grabs all folders starting with noise
#dirfolders = glob.glob("//media/muchacho/onion/dirfiles/toltec_tran_3*") # grabs all folders starting with noise

# Folder directory
dirfolders = glob.glob("/home/muchacho/container-data/noise022420/*/*.dir") # grabs all folders
#creating Listbox, Label and GUI
dirfiles=Listbox(root,width=110,height=20,selectmode=MULTIPLE)
label=Label(root,text='Select Avaliable Folder(s)')
label.pack()

#dirfiles=[]

#For loop to load all folders into Listbox
for i in dirfolders:
    dirfiles.insert(END,i)
#    print('dirfiles = '),dirfiles

#Loading Folders to GUI
dirfiles.pack()
#print(dirfiles)

#Method for loading selected files into values and passing to the loop Def
def do():
  values = [[dirfiles.get(x)] for x in dirfiles.curselection()]
  values.sort()
#  print('values = '),values
  root.destroy()
  loop(values)
    
Button(root,text='Enter',command=do).pack()
lim = 2**15
###########################################
# Phase Power spectrum using the Welch Periodiogram method
############################################
def powerSpectrum(I,Q,fs):
  #f, Pxx = signal.welch( (I**2+Q**2)**(1./2.) , fs, nperseg=256) # mag
  f, Pxx = signal.welch( np.arctan2(Q,I) , fs, nperseg=lim) # phase 
  return f, Pxx #Returns frequency in Hz and Power in timestream_unit^2/Hz

fs1 = 488.28125 # Hz
channel = raw_input("Choose channel: ")
chan = int(channel)
plt.ion()
plt.figure()

#print('values = '),values
#print('dirfiles = '),dirfiles
#print('dirfiles2 = '),dirfiles2
def loop(dirfiles2):
  print('values3 = '),dirfiles2
  for i in range(len(dirfiles2)):
    I, Q = getIQ(dirfiles2[i][0],chan)
    I, Q = I[0:lim], Q[0:lim]
    f,Pxx = powerSpectrum(I,Q,fs1)
    plt.semilogx(f,10.0*np.log10(Pxx),label=dirfiles2[i][0])
    plt.legend()
    #plt.loglog(f,Pxx,label=dirfolders[i])
plt.ylabel("$S_{\delta \phi}$ [$\\frac{dBc}{Hz}$]",size=20)
#plt.ylabel("$S_{\delta \phi}$ [$\\frac{rad^2}{Hz}$]",size=20)
plt.xlabel("Frequency [Hz]",size=20)
#plt.legend()
plt.tight_layout()
plt.grid()
#plt.savefig("phasePowerSpectrum.png")
plt.show()
root.mainloop()
