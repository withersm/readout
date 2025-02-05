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

    def reply(self, cmd: str, status: str, args):
        msg = {"cmd": cmd, "status": status, "data": args}
        self.r.publish("picard_reply", json.dumps(msg))

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
                        print("dstmac_mbs {}       dstmac_lsb {}".format(self.cfg.dstmac_msb, self.cfg.dstmac_lsb))
                        self.rfsoc.initRegs(
                            self.cfg.dstmac_msb,
                            self.cfg.dstmac_lsb,
                            self.cfg.src_ipaddr,
                            self.cfg.dst_ipaddr,
                        )
                        print("Done")
                        self.r.set("status", "free")
                    
                    elif cmd["cmd"] == "changeDDC":
                        newLUT_I = np.array(cmd["args"][0]).astype(float)
                        newLUT_Q = np.array(cmd["args"][1]).astype(float)
                        #freqs = np.array(cmd["args"][2]).astype(float)
                        #print(f'freqs = {freqs}')
                        print(f'newLUT_I = {newLUT_I}')
                        print(f'newLUT_Q = {newLUT_Q}')
                        ddc_I, ddc_Q = self.rfsoc.changeDDC(newLUT_I + 1j*newLUT_Q)
                        reply = {
                            "cmd": 'changeDDC',
                            "status": "OK",
                            "data": {'ddc_I': ddc_I.tolist(), "ddc_Q": ddc_Q.tolist()}
                        }
                        self.r.publish("picard_reply", json.dumps(reply))
                        self.r.set("status", "free")
                        print('Done')

                    elif cmd["cmd"] == "resetDDC":
                        ddc_I, ddc_Q = self.rfsoc.resetDDC()
                        reply = {
                            "cmd": 'resetDDC',
                            "status": "OK",
                            "data": {'ddc_I': ddc_I.tolist(), 'ddc_Q': ddc_Q.tolist()}
                        }
                        self.r.publish("picard_reply", json.dumps(reply))
                        self.r.set("status", "free")
                        print('Done')

                    elif cmd["cmd"] == "ulWaveform":
                        if len(cmd["args"]) == 3: 
                            print("Writing Full Comb")
                            freqs = self.rfsoc.writeWaveform([], [], vna=True, accum_length=cmd["args"][0], demod=cmd["args"][1], demod_filepath=cmd["args"][2]) 
                            reply = {
                                "cmd": "ulWaveform",
                                "status": "OK",
                                "data": freqs.tolist()
                            }
                            self.r.publish("picard_reply", json.dumps(reply))
                            print("Done")
                        else:
                            print("Writing Specified Waveform")
                            print(cmd["args"][0])
                            freqs = self.rfsoc.writeWaveform(
                                    cmd["args"][0], cmd["args"][1], vna=False, accum_length=cmd["args"][2], demod=cmd["args"][3], demod_filepath=cmd["args"][4]
                            )
                            reply = {
                                "cmd": "ulWaveform",
                                "status": "OK",
                                "data": freqs.tolist()                               
                            }
                            self.r.publish("picard_reply", json.dumps(reply))
                            print("Done")
                        self.r.set("status", "free")
                    elif cmd["cmd"] == "get_last_flist":
                        print("get_last_flist")
                        self.reply(
                            "get_last_flist", "OK", self.rfsoc.last_flist.tolist()
                        )
                    elif cmd["cmd"] == "get_last_alist":
                        print("get last alist")
                        self.reply(
                            "get_last_alist", "OK", self.rfsoc.last_alist.tolist()
                        )
                    elif cmd["cmd"] == "get_last_plist":
                        print("get_last_plist")
                        self.reply(
                            "get_last_plist", "OK", self.rfsoc.last_plist.tolist()
                        )
                    elif cmd["cmd"] == "exit":
                        print("Exiting as Commanded")
                        return
                    elif cmd["cmd"] == "set_pmod":
                        print("setting pmod Registers")
                        self.rfsoc.set_pmod(
                            int(cmd["args"][0]),
                            int(cmd["args"][1])
                        )

                        reply = {
                            "cmd": "set_pmod",
                            "status": "OK",
                            "data": "",
                        }

                        self.r.publish("picard_reply", json.dumps(reply))
                    elif cmd["cmd"] == "read_pmod":
                        print("reading pmod registers")
                        self.reply(
                            "read_pmod", "OK", self.rfsoc.read_pmod()
                        )
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
