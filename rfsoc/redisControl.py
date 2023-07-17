"""
@author: Cody Roberson
@date: July 16, 2023
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

Some portions of this code is 'borrowed' from 
    https://github.com/TheJabur/CCATpHive/blob/main/alcove.py
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
import logging

# setup logging config, stolen from https://github.com/TheJabur/CCATpHive/blob/main/queen.py
logging.basicConfig(
    filename="ops.log",
    level=logging.DEBUG,
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
    format="{asctime} {levelname} {filename}:{lineno}: {message}",
)


class Cli:
    def __init__(self):
        # Get data from argparser
        self.conf = config.GeneralConfig("rfsocconfig.cfg")
        host = self.conf.cfg.redis_host
        port = int(self.conf.cfg.redis_port)

        self.rcli = redis.Redis(host=host, port=port)
        # self.r = redis.Redis(host=host)
        self.pubsub = self.rcli.pubsub()
        self.name = self.conf.cfg.rfsocName
        self.pubsub.subscribe(self.name)
        self.rfsoc = rfsocInterface.rfsocInterface()

    def reply(self, cmd, ret):
        message = {"cmd": cmd, "data": ret}
        jd = json.dumps(message)
        self.rcli.publish(self.name + "_reply", jd)

    def run(self):
        logging.info("Starting Listener")
        for msg in self.pubsub.listen():
            if msg["type"] != "message":
                continue
            #            chan = msg['channel'].decode('utf-8')
            data = msg["data"].decode("utf-8")
            data = json.loads(data)
            cmd = data["cmd"]
            args = data["args"]

            func = getattr(self.rfsoc, cmd)
            try:
                ret = func(*args)
                self.reply(cmd, ret)
            except Exception as e:
                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                message = template.format(type(e).__name__, e.args)
                logging.error(message)

    # def listen(self):
    #     print("Starting Listener")
    #     try:
    #         for message in self.p.listen():
    #             if message["type"] != "message":
    #                 continue
    #             # Here was expect to have a valid message
    #             chan = message["channel"]
    #             if chan == bytes(self.uuid, "ascii"):
    #                 # command parsing goes here
    #                 cmd = "nil"
    #                 try:
    #                     data = message["data"].decode("ASCII")
    #                     cmd = json.loads(data)
    #                 except json.JSONDecodeError:
    #                     print("JSON FORMAT PARSE ERROR")
    #                     return

    #                 # Upload Bitstream
    #                 if cmd["cmd"] == "ulBitstream":
    #                     if len(cmd["args"]) == 0:
    #                         print("Writing default bitstream")
    #                         self.rfsoc.uploadOverlay()
    #                         print("Done")
    #                     else:
    #                         print("Writing Specified bitstream")
    #                         print(cmd["args"][0])
    #                         self.rfsoc.uploadOverlay(bitsream=cmd["args"][0])
    #                         print("Done")
    #                     self.r.set("status", "free")

    #                 elif cmd["cmd"] == "initRegs":
    #                     print("Initializing Registers")
    #                     self.rfsoc.initRegs()
    #                     print("Done")
    #                     self.r.set("status", "free")

    #                 elif cmd["cmd"] == "ulWaveform":
    #                     if len(cmd["args"]) == 0:
    #                         print("Writing Full Comb")
    #                         freqs = self.rfsoc.writeWaveform(None, vna=True)
    #                         reply = {
    #                             "cmd": "ulWaveform",
    #                             "status": "OK",
    #                             "data": freqs.tolist(),
    #                         }
    #                         self.r.publish("picard_reply", json.dumps(reply))
    #                         print("Done")
    #                     else:
    #                         print("Writing Specified Waveform")
    #                         print(cmd["args"][0])
    #                         freqs = self.rfsoc.writeWaveform(
    #                             np.array(cmd["args"][0]), vna=False
    #                         )
    #                         reply = {
    #                             "cmd": "ulWaveform",
    #                             "status": "OK",
    #                             "data": freqs.tolist(),
    #                         }
    #                         self.r.publish("picard_reply", json.dumps(reply))
    #                         print("Done")
    #                     self.r.set("status", "free")
    #                 elif cmd["cmd"] == "exit":
    #                     print("Exiting as Commanded")
    #                     return
    #                 else:
    #                     print("Command Not Recognized")

    #     except KeyboardInterrupt:
    #         print("Exiting")
    #         return


def main():
    client = Cli()
    Cli.run()


if __name__ == "__main__":
    main()
