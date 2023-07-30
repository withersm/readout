"""
Overview
--------

This file is used to simulate UDP data being sent over ethernet 
in order to test multi-threading in udp2.py.

By default, the data is sent to address 127.0.0.1 (localhost)
on port 1234. However, the port may be passed in via the command line.

This code runs until interupted via ctrl-c

.. code:: bash

    python udp_sender.py 4096

"""
import socket
import random
import time
import numpy as np
import sys

if __name__ == "__main__":
    # Specify the destination host and port
    host = "127.0.0.1"  # Change this to the receiver's IP address
    if len(sys.argv) > 1:
        port = int(sys.argv[1])  # Change this to the receiver's port
    else:
        port = 1234
    # Number of packets and size of each packet
    num_packets = 488
    packet_size = 2048
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    i = 0
    print(f"sending testdata to {host}:{port}")
    while True:
        pktpersecond = 1 / 489

        try:
            print(f"sending {i}-th set of packets to to {host}:{port}")
            for _ in range(num_packets):
                data = np.random.randint(low=0, high=2**14, size=packet_size)
                # Send the packet to the specified host and port
                udp_socket.sendto(data, (host, port))
                time.sleep(pktpersecond)
            i += 1

        except KeyboardInterrupt:
            print("\nKeyboard Interrupt")
            udp_socket.close()
            exit()
