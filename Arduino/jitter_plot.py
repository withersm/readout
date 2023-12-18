import matplotlib.pyplot as plt
import numpy as np

f = np.loadtxt("serial_output.txt", delimiter = " ", usecols=(0))


k = [a-f[0] for a in f]
idx = np.linspace(0, len(f)-1, len(f))*0.005

offset = k-idx

plt.figure()
plt.plot(idx, offset*1000, ".")
plt.xlabel("Time [s]")
plt.ylabel("nth point timing - n*5ms [ms]")
plt.title("serial_monitor.py timing accuracy")
plt.show()
