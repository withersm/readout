"""
@author: Cody Roberson
@date: June 14, 2022
@file: redisControl.py
@description:

"""

#user check since we can't run without root priviliges
import getpass
if getpass.getuser() != "root":
    print("rfsocInterface.py: root priviliges are required, please run as root.") 
    exit()


import redis
import rfsocInterface
import numpy as np
import json
from time import sleep

class cli:
    def __init__(self, host="192.168.2.10"):
        self.r = redis.Redis(host=host)
        self.p = self.r.pubsub() 
        
        self.p.subscribe("picard") # command and control
        
        sleep(1)
        if self.p.get_message()['data'] == 1:
            print("Successfully subscribed to Captain Picard, Awaiting Commands...") 
            self.rfsoc = rfsocInterface.rfsocInterface() # create interface object
        else:
            print("Something went wrong when subscibing to Captain Picard's Commands")
        
        self.p.subscribe("ping")
        sleep(1)
        if self.p.get_message()['data'] == 2:
            print("Successfully subscribed to PING, Awaiting Commands...") 
        else:
            print("Something went wrong when subscibing to the PING channel")


    def listen(self):
        print("Starting Listener")
        try:
            for message in self.p.listen():
                if message is not None and isinstance(message, dict):
                    # Here was expect to have a valid message
                    chan = message['channel']
                    if chan == b"ping":
                        print(message)
                        if message['data'] == b"hello?":
                            print("Received Ping")
                            self.r.publish("ping", "Hello World")
                    elif chan == b"picard":
                        # command parsing goes here
                        cmd = "nil"
                        try:
                            data = message['data'].decode("ASCII") 
                            cmd = json.loads(data)
                        except:
                            print("JSON FORMAT PARSE ERROR")
                            return
                        
                        if cmd['cmd'] == "ulBitstream":
                            if (len(cmd['args']) == 0):
                                print("Writing default bitstream")
                                self.rfsoc.uploadOverlay()
                                print("Done")
                            else:
                                print("Writing Specified bitstream")
                                print(cmd['args'][0])
                                self.rfsoc.uploadOverlay(bitsream = cmd['args'][0])
                                print("Done")
                            self.r.set("status", "free")

                        elif cmd['cmd'] == "initRegs":
                            print("Initializing Registers")
                            self.rfsoc.initRegs()
                            print("Done")
                            self.r.set("status", "free")
                        elif cmd['cmd'] == "ulWaveform":
                            if (len(cmd['args']) == 0):
                                print("Writing Full Comb")
                                self.rfsoc.writeWaveform(None, vna=True)
                                print("Done")
                            else:
                                print("Writing Specified Waveform")
                                print(cmd['args'][0])
                                self.rfsoc.writeWaveform(np.array(cmd['args'][0]), vna=False)
                                print("Done")
                            self.r.set("status", "free")
                        elif cmd['cmd'] == "exit":
                            print("Exiting as Commanded")
                            return
                        else:
                            print("Command Not Recognized")
                    else:
                        print("Invalid Channel Error, something has gone horribly wrong. Turbolifts are down.")

        except KeyboardInterrupt:
            print("Exiting")
            return


def main():
    client = cli()
    client.listen()

if __name__ == "__main__":
    main()
