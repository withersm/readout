# generate a .np file which is a list of frequencies to be loaded to the QDR
import numpy as np
import matplotlib.pyplot as plt
# create numpy array of freqs
c = 2.998e8 # m/s
freqs = np.linspace(10.0e6,249.0e6,240)
# no correction
phases = np.zeros(len(freqs))
amps = np.ones(len(freqs))

np.save("rel_amp",amps)
np.save("rel_phase",phases)
np.save("freq_gen",freqs)
