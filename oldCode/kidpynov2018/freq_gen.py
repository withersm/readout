# generate a .np file which is a list of frequencies to be loaded to the QDR
import numpy as np
# create numpy array of freqs
#freqs = np.array([21.0e6,35.0e6])
"""
freqs = [
134.77387,
123.51759,
106.6331660,
91.03517600,
87.01507500,
84.60301500,
81.70854300,
78.97487400,
76.88442200,
74.47236200,
70.77386900,
66.43216100
]
"""
#freqs=np.array(freqs)
#freqs = np.array((-179.2e6,-10.24e6,128.0e6))
#freqs = np.array((10.0e6,55.0e6))
#freqs = np.linspace(-255.0e6,255.0e6,500) # 500 tones
freqs = np.linspace(-255.0e6,255.0e6,655) # 500 tones
#freqs = np.linspace(10.0e6,255.0e6,100) # 100 positive tones
#freqs = np.linspace(10.0e6,255.0e6,64) # 64 positive tones
#freqs = np.linspace(10.0e6,255.0e6,10) # 10 positive tones
np.save("freq_gen",freqs)

