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

class cli:
    def __init__(self, host="192.168.2.10"):
        self.r = redis.Redis(host=host)
        self.p = self.r.pubsub()
        self.p.subscribe("picard") # command and control
        if self.p.get_message()['data'] == 1:
            print("Successfully subscribed to Captain Picard, Awaiting Commands...") 
            self.rfsoc = rfsocInterface.rfsocInterface() # create interface object
        else:
            print("Something went wrong when subscibing to Captain Picard's Commands")

    def listen(self):
        print("Starting Listener")
        try:
            for message in self.p.listen():
                if message is not None and isinstance(message, dict):
                    # Here was expect to have a valid message
                    chan = message['channel']
                    if chan == b"picard":
                        # command parsing goes here
                        cmd = "nil"
                        try:
                            data = message['data'].decode("ASCII") 
                            cmd = json.loads(data)
                        except:
                            print("JSON FORMAT PARSE ERROR")
                        
                        if cmd['cmd'] == "ulBitstream":
                            print("Uploading bitstream...")
                            self.rfsoc.uploadOverlay()
                            print("Done")
                        elif cmd['cmd'] == "initRegs":
                            print("Initializing Registers")
                            self.rfsoc.initRegs()
                            print("Done")
                        elif cmd['cmd'] == "ulWaveform":
                            if len(cmd['args'][0]) == 0:
                                print("Writing Full Comb")
                                self.rfsoc.writeWaveform(vna=True)
                                print("Done")
                            else:
                                print("Writing Specified Waveform")
                                print(cmd['args'][0])
                                self.rfsoc.writeWaveform(np.array(cmd['args'][0]), vna=False)
                                print("Done")
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
