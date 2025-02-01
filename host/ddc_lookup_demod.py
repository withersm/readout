import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq



#infrastructure for generating ddc lookup table for flux ramp demod

#planned procedure:
"""
1. Tone initalization
2. Flux ramp activation
3. Flux ramp characterization - collect time series data for each detector
4. Pass ts data to python
5. Compute first harmonic for each detector
6. Generate a new ddc lookup table that takes into account beat frequency from returned datastream not
matching up with the bin and the flux ramp
7. Upload ddc lookup table to the RFSoC and collect ts data -> data should now be naturally removing the 
fr

***NOTE that this procedure will need to be completed every time the fr parameters are changed



"""
