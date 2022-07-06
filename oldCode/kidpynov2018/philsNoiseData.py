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
#dirfolders = glob.glob("dirfiles/test4*") # grabs all folders starting with noise
#dirfolders = glob.glob("//media/muchacho/onion/dirfiles/toltec_tran_3*") # grabs all folders starting with noise
#GUI Frame Setup


# Get Channel Number
chan = 0
try:
  chan = int(raw_input("Channel Number: "))
except:
  print "NaN"
  exit()


root = Tkinter.Tk()
root.geometry('800x450')

# Folder directory
dirfolders = glob.glob("/home/muchacho/container-data/noise022420/*/*.dir") # grabs all folders
dirfolders.sort()
#creating Listbox, Label and GUI
dirfiles=Listbox(root,width=110,height=20,selectmode=MULTIPLE)
label=Label(root,text='Select Avaliable Folder(s)')
label.pack()

#dirfiles=[]

#For loop to load all folders into Listbox
for i in dirfolders:
    #j = i.split('/') # Should extract and list dir files by their name only
    #k = j[-2] + "/" + j[-1]
    dirfiles.insert(END,i)

#Loading Folders to GUI
dirfiles.pack()
#print(dirfiles)

#Method for loading selected files into values and passing to the loop Def
def do():
  values = [[dirfiles.get(x)] for x in dirfiles.curselection()]
#  print('values = '),values
  root.destroy()

  loop(values)
    
Button(root,text='Enter',command=do).pack()
lim = 2**18
###########################################
# Phase Power spectrum using the Welch Periodiogram method
############################################
def powerSpectrum(I,Q,fs):
  #f, Pxx = signal.welch( (I**2+Q**2)**(1./2.) , fs, nperseg=256) # mag
  f, Pi = signal.welch(I/((np.mean(I)**2+np.mean(Q)**2)**(1./2.)), fs, nperseg=8192) # I
  f, Pq = signal.welch(Q/((np.mean(I)**2+np.mean(Q)**2)**(1./2.)),  fs, nperseg=8192) # I
  #f, Pxx = signal.welch( np.arctan2(Q,I) , fs, nperseg=256)#lim) # phase 
  return f, Pq, Pi #Returns frequency in Hz and Power in timestream_unit^2/Hz


  #print('values = '),values
  #print('dirfiles = '),dirfiles
  #print('dirfiles2 = '),dirfiles2
def loop(dirfiles2):
  fs1 = 488.28125 # Hz
  plt.ion()
  plt.figure()
  print('values3 = '),dirfiles2
  for i in range(len(dirfiles2)):
    I, Q = getIQ(dirfiles2[i][0],chan)
    print(len(I))
#    I, Q = I[0:lim], Q[0:lim]
    f,Pq,Pi = powerSpectrum(I,Q,fs1)
    plt.semilogx(f,10.0*np.log10(Pi),label=dirfiles2[i][0])
    plt.semilogx(f,10.0*np.log10(Pq),label=dirfiles2[i][0])
    plt.legend(loc="bottom-right")
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
