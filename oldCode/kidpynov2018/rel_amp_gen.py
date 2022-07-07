# generate a .np file which is a list of relative amplitudes  
import numpy as np
amps = np.array((1.0181, 1.0181, 1.0181, 1.0181))
#amps = np.array((1.018,1.02))
np.save("rel_amp",amps)
