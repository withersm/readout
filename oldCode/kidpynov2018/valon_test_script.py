# example script for valon dual chan synth

import valon_synth9 as valon

v = valon.Synthesizer("/dev/ttyUSB0") # usb device name for valon, check with dmesg command at terminal

print(v.get_frequency(1))
print(v.get_frequency(2))

v.set_frequency(2,512) # channel, frequency in MHz
v.set_rf_level(2,5) # channel, frequency in MHz
