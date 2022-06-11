import socket
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from scipy import signal

DEFAULT_UDP_IP = "192.168.3.40"
DEFAULT_UDP_PORT = 4096

class udpcap():
    def __init__(self, UDP_IP = DEFAULT_UDP_IP, UDP_PORT = DEFAULT_UDP_PORT):
        self.UDP_IP = UDP_IP
        self.UDP_PORT = UDP_PORT
        print(self.UDP_IP)
        print(self.UDP_PORT)

    def bindSocket(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.UDP_IP,self.UDP_PORT))

    def parse_packet(self):
        data = self.sock.recv(9000)
        if len(data) <  8000:
            print("invalid packet recieved")
        print(len(data))
        spec_data = np.fromstring(data, dtype = '<i')
        return spec_data # int32 data type
    
    def capture_packets(self, N_packets):
        packets = np.zeros(shape=(2052,N_packets))
        #packets = np.zeros(shape=(2051,N_packets))
        for i in range(N_packets):
            data_2 = self.parse_packet()
            packets[:,i] = data_2 
        return packets
    def release(self):
        self.sock.close()

def parse_packet_wtime(c=2**32):
  data = sock.recv(9000)
  if len(data) <  8000:
    print("invalid packet recieved")
  spec_data = np.fromstring(data, dtype = '<i')
  # time stamp code
  timestamp_data = np.fromstring(data, dtype = '>i')[-4:]
  tds=[]
  for td in timestamp_data:
    tds.append(hex(c+td)[2:].replace("-",""))
  print(tds)
  td_str=''.join(tds)[6:]
  return td_str, spec_data # int32 data type  


def get_timestamp(timestamp_data):
    """
      Casting packet timing information 
    """
    td=int(timestamp_data,16)
    print(timestamp_data)
    print(td)
    free_count   = (td & int("FFFFFFFF0000000000000000000000",16))>>88
    packet_count = (td & int("00000000FFFFFFFF00000000000000",16))>>56
    pps_count    = (td & int("0000000000000000FFFFFFFF000000",16))>>24
    GPIO         = (td & int("000000000000000000000000FFFF00",16))>>8 
    PMOD         = (td & int("0000000000000000000000000000FF",16))    

    return (free_count, packet_count, pps_count, GPIO, PMOD)

def check_pps_count():
  pcount=[]
  for i in range(25):
    sleep(0.5)
    t_info, _ = parse_packet()
    (_, _, pps_count, _, _) = get_timestamp(t_info)
    #(_, packet_count, _, _, _) = get_timestamp(t_info)
    pcount.append(pps_count)
  return np.array(pcount)

def check_packet_count():
  pcount=[]
  for i in range(25):
    sleep(0.5)
    t_info, _ = parse_packet()
    #(_, _, pps_count, _, _) = get_timestamp(t_info)
    (_, packet_count, _, _, _) = get_timestamp(t_info)
    pcount.append(packet_count)
  return np.array(pcount)


def print_time():
  t_info, _ = parse_packet()
  (free_count, packet_count, pps_count, GPIO, PMOD) = get_timestamp(t_info)
  #print("""
  #free_count    |  0x{free_count  :x}
  #packet_count  |  0x{packet_count:x}
  #pps_count     |  0x{pps_count   :x}
  #GPIO          |  0x{GPIO        :x}
  #PMOD          |  {PMOD        :b}b
  #""")

  return

def print_max_value():
  for i in range(100000):
    data = parse_packet()
    I, Q = data[0::2], data[1::2]
    maxI_idx = np.where(abs(I)==max(abs(I)))[0][0]
    maxQ_idx = np.where(abs(Q)==max(abs(Q)))[0][0]
    print(str(maxI_idx)+" "+str(I[maxI_idx])+" "+str(maxQ_idx)+" "+str(Q[maxQ_idx]))
  return 0

def print_packet_data(Ichan, Qchan):
  for i in range(100000):
    data_2 = parse_packet()
    print(str(data_2[Ichan])+" " +str(data_2[Qchan]))
  return 0

def otra_print_packet_data(pkt_index):
  chan=2*pkt_index
  for i in range(100000):
    data_2 = parse_packet()
    print("I: {} Q: {}".format(data_2[chan+1],data_2[chan]))
  return 0



def phase_noise(I_idx, Q_idx, N_packets = 1024, nperseg=1024):
  data = capture_packets(N_packets)
  #I, Q = data[packet_index], np.roll(data[packet_index+1],1)
  f, S = signal.welch(np.arctan2(data[Q_idx],data[I_idx]),fs=512e6/2**20,nperseg=nperseg)
  plt.semilogx(f,10*np.log10(S))
  plt.ylabel("dBc/Hz", fontsize=16); plt.xlabel("freq.", fontsize=16)
  #if save:
  #  np.savez("~/Pictures/phasenosie",f,S)
  plt.show()
  return 0

def phase_noise2(packet_index, N_packets = 1024, nperseg=1024, alpha=1.0, save=False):
  data = capture_packets(N_packets)
  I, Q = data[packet_index], np.roll(data[packet_index+1],1)
  f, S = signal.welch(np.arctan2(Q,I),fs=512e6/2**20,nperseg=nperseg)
  plt.semilogx(f,10*np.log10(S),color="blue",alpha=alpha)
  plt.ylabel("dBc/Hz", fontsize=16); plt.xlabel("freq.", fontsize=16)
  plt.show()
  if save:
    np.savez("1k_phasenosie",f,S)
  return 0

def phase_noise_all(nstart, nstop, nstep=1, N_packets = 1024, nperseg=1024,color="blue", alpha=1.0):
  data = capture_packets(N_packets)
  for a in range(nstart,nstop,nstep):
    I, Q = data[2*a], np.roll(data[2*a+1],1)
    f, S = signal.welch(np.arctan2(Q,I),fs=512e6/2**20,nperseg=nperseg)
    plt.semilogx(f,10*np.log10(S),color=color,alpha=alpha)
  plt.ylabel("$S_{\phi}$ [dBc/Hz]", fontsize=14); plt.xlabel("Frequency [Hz]", fontsize=14)
  plt.tight_layout()
  plt.show()
  return 

def phase_noise_iq(packet_index, N_packets = 1024, nperseg=1024):
  data = capture_packets(N_packets)
  I, Q = data[packet_index], np.roll(data[packet_index+1],1)
  f, Si = signal.welch( I, fs=512e6/2**20, nperseg=nperseg)
  f, Sq = signal.welch( Q, fs=512e6/2**20, nperseg=nperseg)
  phi = ( Si + Sq)/(np.mean(I)**2 + np.mean(Q)**2)/2.
  plt.semilogx(f,10*np.log10(phi))#,label='Phil Noise')
  #plt.semilogx(f,10*np.log10(Si),label='I')
  #plt.semilogx(f,10*np.log10(Sq),label='Q')
  plt.ylabel("$S_{\phi}$ [dBc/Hz]", fontsize=14); plt.xlabel("Frequency [Hz]", fontsize=14)
  plt.legend()
  plt.show()
  return 0

import os
def live_plot(beep=False,ymax=3e9):
  plt.figure()
  while 1:
    data = parse_packet()
    I, Q = data[0::2][0:1024], data[1::2][0:1024]
    Ia, Qa = np.array(I).astype(float), np.array(Q).astype(float)
    plt.clf()
    IQmag=np.sqrt(Ia**2 + Qa**2)
    if (beep and np.max(IQmag)>0.5e9):
      os.system("beep -r 2")
      print("Max Found!")
    plt.plot(10.*np.log10(IQmag))
    plt.ylim(1, 99)#ymax)
    plt.xlim(0,1024)
    plt.pause(0.01)
  plt.show()
  return 0

def magick():
  pkts=capture_packets(8192)
  I=pkts[129,:]
  Q=pkts[130,:]
  
  plt.subplot(211)
  plt.hist(I,bins=1024)
  plt.subplot(212)
  plt.hist(Q,bins=1024)
  plt.show()
