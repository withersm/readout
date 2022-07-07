# Read dirfiles that are produced by kidPy_nov2018 
# Run this script in the noise dirfile directory
from readDirfile import * 
import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy import signal

#dirfolders = glob.glob("dirfiles/test4*") # grabs all folders starting with noise
dirfolders = glob.glob("//media/muchacho/onion/dirfiles/toltec_tran_3*") # grabs all folders starting with noise
dirfiles=[]
for i in range(len(dirfolders)):
    dirfiles.append(glob.glob(dirfolders[i]+"/*.dir"))
#    print('values = '),dirfiles
###########################################
# Phase Power spectrum using the Welch Periodiogram method
############################################
def powerSpectrum(I,Q,fs):
  #f, Pxx = signal.welch( (I**2+Q**2)**(1./2.) , fs, nperseg=256) # mag
  f, Pxx = signal.welch( np.arctan2(Q,I) , fs, nperseg=256) # phase 
  return f, Pxx #Returns frequency in Hz and Power in timestream_unit^2/Hz

fs1 = 488.28125 # Hz
lim = 25000
chan = 1
plt.ion()
plt.figure()
print('values = '),dirfiles
for i in range(len(dirfiles)):
    I, Q = getIQ(dirfiles[i][0],chan)
    I, Q = I[0:lim], Q[0:lim]
    f,Pxx = powerSpectrum(I,Q,fs1)
    plt.plot(f,10.0*np.log10(Pxx),label=dirfolders[i])
    #plt.loglog(f,Pxx,label=dirfolders[i])
plt.ylabel("$S_{\delta \phi}$ [$\\frac{dBc}{Hz}$]",size=20)
#plt.ylabel("$S_{\delta \phi}$ [$\\frac{rad^2}{Hz}$]",size=20)
plt.xlabel("Frequency [Hz]",size=20)
plt.legend()
plt.tight_layout()
#plt.savefig("phasePowerSpectrum.png")
plt.show()
