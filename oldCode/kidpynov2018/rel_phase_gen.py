# generate a .np file which is a list of relative phases  
import numpy as np
phases = np.array((0.00746, -0.0067, -0.0181, -0.03))# radians
np.save("rel_phase",phases)
