"""
@author: Cody Roberson
@date: March 9 2023
@file: redisControl.py
@description:
This is a prototype redis message listener and command dispatcher. 
Further work needed to abstract away the messaging system and implement error
logging/handling. 

Format for accepting commands from redis
message = {
    'cmd' : 'cmd here',
    'args' : [
        arg0,
        arg1,
        arg2,
        etc
    ]
}

Format for replying to commands from redis 
message = {
    'cmd' : 'relay command',
    'status' : 'OK'|'FAIL',
    'data' : 'nil' | <arbitrary data>
}

"""

# user check since we can't run without root priviliges
import getpass

if getpass.getuser() != "root":
    print("rfsocInterface.py: root priviliges are required, please run as root.")
    exit()


import redis
import rfsocInterface
import numpy as np
import json
from time import sleep
import config


class Cli:
    def __init__(self, configFile: str = "./config.cfg"):
        self.conf = config.GeneralConfig(configFile)
        self.cfg = self.conf.cfg
        host = self.cfg.redis_host
        port = self.cfg.redis_port
        self.r = redis.Redis(host=host, port=port)
        self.p = self.r.pubsub()

        self.p.subscribe("picard")  # subscribe to uuid.

        sleep(1)
        if self.p.get_message()["data"] == 1:
            print("Successfully subscribed to Command Channel, Awaiting Commands...")
            self.rfsoc = rfsocInterface.rfsocInterface()  # create interface object
        else:
            print("Failed to Subscribe to Command Channel")

    def listen(self):
        print("Starting Listener")
        try:
            for message in self.p.listen():
                if message["type"] != "message":
                    continue
                # Here was expect to have a valid message
                chan = message["channel"]
                if chan == bytes("picard", "ASCII"):
                    # command parsing goes here
                    cmd = "nil"
                    try:
                        data = message["data"].decode("ASCII")
                        cmd = json.loads(data)
                    except json.JSONDecodeError:
                        print("JSON FORMAT PARSE ERROR")
                        return

                    # Upload Bitstream
                    if cmd["cmd"] == "ping":
                        print("received ping")
                        self.r.publish("ping", "Hello World")
                    elif cmd["cmd"] == "ulBitstream":
                        if len(cmd["args"]) == 0:
                            print("Writing default bitstream")
                            self.rfsoc.uploadOverlay()
                            print("Done")
                        else:
                            print("Writing Specified bitstream")
                            print(cmd["args"][0])
                            self.rfsoc.uploadOverlay(bitstream=cmd["args"][0])
                            print("Done")
                        self.r.set("status", "free")

                    elif cmd["cmd"] == "initRegs":
                        print("Initializing Registers")
                        self.rfsoc.initRegs(
                            self.cfg.dstmac_msb,
                            self.cfg.dstmac_lsb,
                            self.cfg.src_ipaddr,
                            self.cfg.dst_ipaddr,
                        )
                        print("Done")
                        self.r.set("status", "free")

                    elif cmd["cmd"] == "ulWaveform":
                        if len(cmd["args"]) == 0:
                            print("Writing Full Comb")
                            freqs = self.rfsoc.writeWaveform([], [], vna=True)
                            reply = {
                                "cmd": "ulWaveform",
                                "status": "OK",
                                "data": freqs.tolist(),
                            }
                            self.r.publish("picard_reply", json.dumps(reply))
                            print("Done")
                        else:
                            print("Writing Specified Waveform")
                            print(cmd["args"][0])
                            freqs = self.rfsoc.writeWaveform(
                                cmd["args"][0], cmd["args"][1], vna=False
                            )
                            reply = {
                                "cmd": "ulWaveform",
                                "status": "OK",
                                "data": freqs.tolist(),
                            }
                            self.r.publish("picard_reply", json.dumps(reply))
                            print("Done")
                        self.r.set("status", "free")
                    elif cmd["cmd"] == "exit":
                        print("Exiting as Commanded")
                        return
                    else:
                        print("Command Not Recognized")

        except KeyboardInterrupt:
            print("Exiting")
            return


def main():
    client = Cli()
    client.listen()


if __name__ == "__main__":
    main()
