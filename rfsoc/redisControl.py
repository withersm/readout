import redis
import rfsocInterface
import numpy as np

class cli:
    def __init__(self, host="192.168.2.10"):
        self.r = redis.Redis(host=host)
        self.p = self.r.pubsub()
        self.p.subscribe("picard") # command and control
        self.rfsoc = rfsocInterface.rfsocInterface() # create interface object

    def listen(self):
        try:
            for message in self.p.listen():
                if message is not None and isinstance(message, dict):
                    ch = message.get('channel')
                    d = message.get('data')
                    if d != 1:
                        print("message on {} had data {}".format(ch.decode("ASCII"), 
                            d.decode("ASCII")))
                    # now for the meat and gravy of the project
                    # decode message as commands 
                    if ch == b"picard":
                        if d == b"ulBitstream":
                            self.rfsoc.uploadOverlay()
                        elif d == b"initReg":
                            self.rfsoc.initRegs()
                        elif d == b"ulWaveform":
                            self.rfsoc.writeWaveform(None, vna=True)
                    if d == b'exit':
                        return
        except KeyboardInterrupt:
            print("Exiting")
            return

