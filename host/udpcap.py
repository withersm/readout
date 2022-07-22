import socket
from turtle import pd
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from scipy import signal
import h5py

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
        
        # now allow a shift of the bytes
        spec_data = np.frombuffer(datarray, dtype = '<i')
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
                print("{}/{} captured ({:.3f}% Complete)".format(i, N_packets, 
                    (N_packets/488)*100.0))
        return packets

    def __ldcHelper(self):

        """
        We want to use this subprocess to copy memory out of our numpy array
        and into the h5py data set. The hdf5 backend controls when to flush it's 
        buffers automatically however we're still working on performance
        issues and this may help mitigate them.
        """
        pass

    def LongDataCapture(self, fname, nPackets):
        """
        Captures packets and saved them to an hdf5 type file
        N : int
            Number of packets to save
        fname : string
            file name / path
        """
        try:
            print("capture {} packets".format(nPackets))
            dFile = h5py.File(fname, 'w')
            pkts = dFile.create_dataset("PACKETS",(2052, nPackets), dtype=h5py.h5t.NATIVE_INT32, chunks=True, maxshape=(None, None))
            print("Begin Capture")
            for i in range(nPackets):
                pkts[:, i] = self.parse_packet()
                if i > 0 and i % 488 == 0:
                    print("{}/{} captured ({:.2f}% Complete)".format(i, nPackets, ((i/nPackets)*100.0)))
            dFile.close()
        except Exception as errorE:
            raise(errorE)
        return True


    def shortDataCapture(self, fname, nPackets):
        """
        Performes sub 60 seconds data captures using only memory. 
        Data is then transferred to a file after the collection is complete.
        N : int
            Number of packets to save
        fname : string
            file name / path
        """

        assert nPackets < 488*60, "METHOD NOT INTENDED FOR LONG DATA CAPTURES > 488*60"
        pData = np.zeros((2052, nPackets))
        try:
            print("capture {} packets".format(nPackets))
            dFile = h5py.File(fname, 'w')
            pkts = dFile.create_dataset("PACKETS",(2052, nPackets), dtype=h5py.h5t.NATIVE_INT32, chunks=True, maxshape=(None, None))
            print("Begin Capture")
            for i in range(nPackets):
                pData[:, i] = self.parse_packet()
                if i > 0 and i % 488 == 0:
                    print("{}/{} captured ({:.2f}% Complete)".format(i, nPackets, ((i/nPackets)*100.0)))
            pkts[...] = pData
            dFile.close()
        except Exception as errorE:
            raise(errorE)
        return True

    def release(self):
        self.sock.close()


