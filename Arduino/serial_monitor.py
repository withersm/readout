import serial
import signal
import time

def run_task(signum, frame):
    ard.write(b'S')
    time.sleep(0.0015)
    if ard.in_waiting:
        read_bytes = ard.readline()
    else:
        read_bytes = "0"
    #print(read_bytes)
    if len(read_bytes) == 11:
        digital = read_bytes[0]
        # 10-bit 5V range: resolution 4.88mV
        ADC1 = (read_bytes[1] + read_bytes[2]*2**8)*0.00488
        ADC2 = (read_bytes[3] + read_bytes[4]*2**8)*0.00488
        ADC3 = (read_bytes[5] + read_bytes[6]*2**8)*0.00488
        ADC4 = (read_bytes[7] + read_bytes[8]*2**8)*0.00488
        #print("DIGITAL: %s, ADC: %.5f V, %.5f V, %.5f V, %.5f V\n"%(bin(digital), ADC1, ADC2, ADC3, ADC4))
        f.write(str(time.time())+' '+bin(digital)+' '+str(ADC1)+' '+str(ADC2)+' '+str(ADC3)+' '+str(ADC4)+'\n')
    else:
        print("0\n")

name = '/dev/ttyACM2'
baud = 115200
timeout = 1
ard = serial.Serial(name, baud, timeout = timeout)

f = open("serial_output.txt", "a")


time.sleep(1)
print(ard.read(100))
time.sleep(1)
signal.signal(signal.SIGALRM, run_task)
signal.setitimer(signal.ITIMER_REAL, 0.005, 0.005)
#try:
while True:
    signal.pause()
#except Exception as e:
#    print(e)
f.close()



