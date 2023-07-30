
import socket
import random
import time
import numpy as np

if __name__ == "__main__":
    # Specify the destination host and port
    host = "127.0.0.1"  # Change this to the receiver's IP address
    port = 12345       # Change this to the receiver's port

    # Number of packets and size of each packet
    num_packets = 488
    packet_size = 2048
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    i = 0
    while True:
        # time.sleep(1)
        try:
            # if i >= 50:
            #     break
            print(f"sending {i}-th set of packets")
            for _ in range(num_packets):
                data  = np.random.randint(low = 0, high=2**14, size=packet_size)
                # Send the packet to the specified host and port
                udp_socket.sendto(data, (host, port))
            i += 1

        except KeyboardInterrupt:
            print("\nKeyboard Interrupt")
            udp_socket.close()
            exit()