# Read dirfiles that are produced by kidPy_nov2018 
# Run this script in the noise dirfile directory
from readDirfile import * 
import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy import signal

dirfolders = glob.glob("/media/muchacho/onion/dirfiles/test1*") # grabs all folders starting with noise
dirfiles=[]
for i in range(len(dirfolders)):
    dirfiles.append(glob.glob(dirfolders[i]+"/*.dir"))

###########################################
# Phase Power spectrum using the Welch Periodiogram method
############################################
def powerSpectrum(I,Q,fs):
  #f, Pxx = signal.welch( (I**2+Q**2)**(1./2.) , fs, nperseg=256) # mag
  f, Pxx = signal.welch( np.arctan2(Q,I) , fs, nperseg=1024) # phase 
  return f, Pxx #Returns frequency in Hz and Power in timestream_unit^2/Hz

fs1 = 488.28125 # Hz
lim = 50000
start = 10000
end = 50000
chan = 0
plt.ion()
plt.figure()
for i in range(len(dirfiles)):
    I, Q = getIQ(dirfiles[i][0],chan)
    #I, Q = I[0:lim], Q[0:lim]
    #I, Q = I[start:end], Q[start:end]
    #I, Q = I[-lim:-1], Q[-lim:-1]
    f,Pxx = powerSpectrum(I,Q,fs1)
    plt.semilogx(f,10.0*np.log10(Pxx),label=dirfolders[i])
    #plt.loglog(f,Pxx,label=dirfolders[i])
plt.ylabel("$S_{\delta \phi}$ [$\\frac{dBc}{Hz}$]",size=20)
#plt.ylabel("$S_{\delta \phi}$ [$\\frac{rad^2}{Hz}$]",size=20)
plt.xlabel("Frequency [Hz]",size=20)
plt.legend()
plt.tight_layout()
#plt.savefig("phasePowerSpectrum.png")
plt.show()
