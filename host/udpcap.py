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
        data = self.sock.recv(8208 * 1)
        if len(data) <  8000:
            print("invalid packet recieved")
            return
        datarray = bytearray(data)
        
        # the above step unrolls the c0ffee data stream
        for i in range(499):
            datarray[214+i*16:216+i*16]=datarray[230+i*16:232+i*16]


        # now allow a shift of the bytes
        byte_off = 6
        for i in range(byte_off):
            datarray.append(0)
        spec_data = np.frombuffer(datarray, dtype = '<i', offset = byte_off)
        # offset allows a shift in the bytes
        return spec_data # int32 data type
       
    def capture_packets(self, N_packets):
        packets = np.zeros(shape=(2052,N_packets))
        #packets = np.zeros(shape=(2051,N_packets))
        counter = 0
        for i in range(N_packets):
            data_2 = self.parse_packet()
            packets[:,i] = data_2 
            if i%488 == 0:
                print("{}/{} captured".format(i, N_packets))
        return packets
    
    def release(self):
        self.sock.close()


