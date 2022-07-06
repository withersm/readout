# correspondance table generator
import numpy as np

x = np.array((0,1,2))
y = np.array((0,0,0))
chan = np.linspace(0,2,3)
correspondance_table = np.zeros((len(chan),8))
for i in range(len(chan)):
  correspondance_table[i][0] = chan[i] 
  correspondance_table[i][5] = x[i]
  correspondance_table[i][6] = y[i]
  
corr_filename = "generated_correspondance_table.csv"
np.savetxt(corr_filename,correspondance_table,delimiter=",")
