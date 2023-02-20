"""
This software is a work in progress. It is a console interface designed 
to operate the BLAST RFSOC firmware. 

Copyright (C) 2022 <Authors listed below>
Author: Cody Roberson, Ryan Stephenson Adrian Sinclair, Philip Mauskopf
      (carobers@asu.edu)

subscribe message Out[6]: {'type': 'subscribe', 'pattern': None, 'channel': b'picard', 'data': 1}

 ******  REVISIONS *****
2022-07-25
    - Began overhaul of command <-> repsonse. Previously we only had a busy please wait approach
        however now we will be utilizing pub-sub to facilitate replys with data payloads.
        Write waveform for instance shall now reply back with the written baseband frequencies.
"""

import numpy as np
import os
import redis
import json
import configparser
import time
import udpcap
import datetime
import valon5009


def testConnection(r):
    try:
        tr = r.set("testkey", "123")
        return True
    except redis.exceptions.ConnectionError as e:
        print(e)
        return False


def wait_for_free(r, delay=0.25, timeout=10):
    count = 0
    r.set("status", "busy")
    while r.get("status") != b"free":
        time.sleep(delay)
        count = count + 1
        if count == timeout:
            print("TIMED OUT WHILE WAITING FOR RFSOC")
            return False
    return True


def wait_for_reply(redis_pubsub, cmd, max_timeout=15):
    """

    This is the eventual replacement for the waitForFree() method.
    We want to have smarter replies that can ferry data back from the RFSOC.

    Once a command is sent out, listen for a reply on the <cmd_reply> channel
        Format for replying to commands from redis
            message = {
                'cmd' : 'relay command',
                'status' : 'OK'|'FAIL',
                'data' : 'nil' | <arbitrary data>
            }

    :param max_timeout: int :
        time in seconds to wait for the RFSOC to reply. If it fails in this time,
        this should indicate a failure of some kind has occured
    :param cmd: str :
        Command sent out
    :param redis_pubsub: redis.Redis.pubsub :
        reference to a Redis pubsub object that has already subscribed to relevant channels
    """
    current_time = 0
    while current_time < max_timeout:
        m = redis_pubsub.get_message()
        if m is not None and m["channel"] == b"picard_reply":
            msg = m["data"].decode("ASCII")
            data = json.loads(msg)
            if data["cmd"] == cmd and data["status"] == "OK":
                return True, data["data"]
            else:
                return False, data["data"]
        time.sleep(1)
        current_time = current_time + 1
    print(
        "WARINNG: TIMED OUT WAITING FOR REPLY -->  def waitForReply(redisIF, cmd, maxTimeout = 15):"
    )


def checkBlastCli(r, p):
    """
    Rudamentary "is the rfsoc control software running" check.
    """
    r.publish("ping", "hello?")
    count = 1
    delay = 0.5
    timeout = 6
    while 1:
        m = p.get_message()
        if m is not None and m["data"] == b"Hello World":
            print("redisControl is running")
            return True
        if count >= timeout:
            print("RFSOC didn't reply, is it running redisControl?")
            return False

        time.sleep(delay)
        count = count + 1


def sweep(loSource, udp, f_center, freqs, N_steps=500):
    """
    Perform an LO Sweep using valon 5009's and save the data

    loSource : valon5009.Synthesizer
        Valon 5009 Device Object instance
    f_center : float
        Frequency center
    freqs : float[]
        List of Baseband Frequencies returned from rfsocInterface.py's writeWaveform()
    udp : udpcap.udpcap object instance
        udp data capture utility. This is our bread and butter for taking data from ethernet
    N_steps : int
        Number of steps to sweep


    Credit: Dr. Adrian Sinclair (adriankaisinclair@gmail.com)
    """
    tone_diff = np.diff(freqs)[0] / 1e6  # MHz
    flo_step = tone_diff / N_steps
    flo_start = f_center - tone_diff / 2.0  # 256
    flo_stop = f_center + tone_diff / 2.0  # 256

    flos = np.arange(flo_start, flo_stop, flo_step)
    udp.bindSocket()

    def temp(lofreq):
        # self.set_ValonLO function here
        loSource.set_frequency(valon5009.SYNTH_B, lofreq)
        # Read values and trash initial read, suspecting linear delay is cause..
        Naccums = 5
        I, Q = [], []
        for i in range(Naccums):
            d = udp.parse_packet()
            It = d[::2]
            Qt = d[1::2]
            I.append(It)
            Q.append(Qt)
        I = np.array(I)
        Q = np.array(Q)
        Imed = np.median(I, axis=0)
        Qmed = np.median(Q, axis=0)

        Z = Imed + 1j * Qmed
        Z = Z[0 : len(freqs)]

        print(".", end="")

        return Z

    sweep_Z = np.array([temp(lofreq) for lofreq in flos])

    f = np.array([flos * 1e6 + ftone for ftone in freqs]).flatten()
    sweep_Z_f = sweep_Z.T.flatten()
    udp.release()

    return f, sweep_Z_f


def loSweep(loSource, udp, freqs=[], f_center=400):
    """
    vnaSweep: perform a stepped frequency sweep centered at f_center and save result as s21.npy file
    f_center: center frequency for sweep in [MHz]
    """
    print(freqs)
    f, sweep_Z_f = sweep(loSource, udp, f_center, freqs)
    np.save("s21.npy", np.array((f, sweep_Z_f)))
    print("s21.npy saved.")


def menu(captions, options):
    """Creates menu for terminal interface
    inputs:
        list captions: List of menu captions
        list options: List of menu options
    outputs:
        int opt: Integer corresponding to menu option chosen by user"""
    print("\t" + captions[0] + "\n")
    for i in range(len(options)):
        print("\t" + "\033[32m" + str(i) + " ..... " "\033[0m" + options[i] + "\n")
    opt = None
    try:
        opt = eval(input())
    except KeyboardInterrupt:
        exit()
    return opt


class kidpy:
    def __init__(self):
        # Pull config
        config = configparser.ConfigParser()
        config.read("generalConfig.conf")
        self.__redis_host = config["REDIS"]["host"]
        self.__customWaveform = config["DSP"]["customWaveform"]
        self.__customSingleTone = config["DSP"]["singleToneFrequency"]
        self.__saveData = config["DATA"]["saveFolder"]
        self.__ValonPorts = config["VALON"]["valonSerialPorts"].split(",")
        self.__valon_RF1_SYS2 = config["VALON"]["rfsoc1System2"]
        self.__valon_RF1_SYS1 = config["VALON"]["rfsoc1System1"]

        # setup redis
        self.r = redis.Redis(self.__redis_host)
        self.p = self.r.pubsub()
        self.p.subscribe("ping")
        time.sleep(1)
        if self.p.get_message()["data"] != 1:
            print("Failed to Subscribe to redis Ping Channel")
            exit()
        self.p.subscribe("picard_reply")
        time.sleep(1)
        if self.p.get_message()["data"] != 2:
            print("Failed to Subscribe redis picard_reply channel")
            exit()

        # check that the rfsoc is running redisControl.py
        os.system("clear")
        if not checkBlastCli(self.r, self.p):
            exit()

        # Differentiate 5009's connected to the system
        self.valon = None
        for v in self.__ValonPorts:
            self.valon = valon5009.Synthesizer(v.replace(" ", ""))
            print(self.valon.getSN())

        self.__udp = udpcap.udpcap()
        self.current_waveform = []
        caption1 = "\n\t\033[95mKID-PY2 RFSoC Readout\033[95m"
        self.captions = [caption1]

        self.__main_opts = [
            "Upload firmware",
            "Initialize system & UDP conn",
            "Write test comb (single or multitone)",
            "Write stored comb from config file",
            "I <-> Q Phase offset",
            "Take Raw Data",
            "LO Sweep",
            "Exit",
        ]

    def begin_ui(self):
        pass

    def main_opt(self):
        """
        Main user interface routing

        r : redis Server instance
        p : redis.pubsub instance
        udp : udpcap object instance
        """
        while 1:

            conStatus = testConnection(self.r)
            if conStatus:
                print("\033[0;36m" + "\r\nConnected" + "\033[0m")
            else:
                print(
                    "\033[0;31m"
                    + "\r\nCouldn't connect to redis-server double check it's running and the generalConfig is correct"
                    + "\033[0m"
                )
            opt = menu(self.captions, self.__main_opts)
            if conStatus == False:
                resp = input(
                    "Can't connect to redis server, do you want to continue anyway? [y/n]: "
                )
                if resp != "y":
                    exit()
            if opt == 0:  # upload firmware
                os.system("clear")
                cmd = {"cmd": "ulBitstream", "args": []}
                cmdstr = json.dumps(cmd)
                self.r.publish("picard", cmdstr)
                self.r.set("status", "busy")
                print("Waiting for the RFSOC to upload its bitstream...")
                if wait_for_free(self.r, 0.75, 25):
                    print("Done")

            if opt == 1:  # Init System & UDP conn.
                os.system("clear")
                print("Initializing System and UDP Connection")
                cmd = {"cmd": "initRegs", "args": []}
                cmdstr = json.dumps(cmd)
                self.r.publish("picard", cmdstr)
                if wait_for_free(self.r, 0.5, 5):
                    print("Done")

            if opt == 2:  # Write test comb
                prompt = input("Full test comb? y/n ")
                os.system("clear")
                if prompt == "y":
                    print("Waiting for the RFSOC to finish writing the full comb")
                    cmd = {"cmd": "ulWaveform", "args": []}
                    cmdstr = json.dumps(cmd)
                    self.r.publish("picard", cmdstr)

                else:
                    print(
                        "Waiting for the RFSOC to write single {} MHz Tone".format(
                            float(self.__customSingleTone) / 1e6
                        )
                    )
                    cmd = {
                        "cmd": "ulWaveform",
                        "args": [[float(self.__customSingleTone)]],
                    }
                    cmdstr = json.dumps(cmd)
                    self.r.publish("picard", cmdstr)

                success, self.current_waveform = wait_for_reply(
                    self.p, "ulWaveform", max_timeout=10
                )
                if success:
                    print("Wrote Waveform")
                    print(self.current_waveform)
                else:
                    print("Failed to write waveform")

            if opt == 3:  # write stored comb
                os.system("clear")
                print(
                    "Waiting for the RFSOC to finish writing the custom frequency list: \r\n{}".format(
                        self.customWaveform
                    )
                )
                fList = []

                # separate values from config and remove ',' before converting to number and
                # sending the list of values up to the DAC
                for value in self.__customWaveform.split():
                    s = value.replace(",", "")
                    fList.append(float(s))

                cmd = {"cmd": "ulWaveform", "args": [fList]}
                cmdstr = json.dumps(cmd)
                self.r.publish("picard", cmdstr)
                success, self.current_waveform = wait_for_reply(
                    self.p, "ulWaveform", max_timeout=10
                )
                if success:
                    print("Wrote Waveform")
                    print(self.current_waveform)
                else:
                    print("Failed to write waveform")

            if opt == 4:
                print("Not Implemented")

            if opt == 5:
                t = 0
                try:
                    t = int(input("How many seconds of data?: "))
                    print(t)
                except ValueError:
                    print("Error, not a valid Number")
                except KeyboardInterrupt:
                    return

                if t <= 0:
                    print("Can't sample 0 seconds")
                else:
                    os.system("clear")
                    print("Binding Socket")
                    self.__udp.bindSocket()
                    print("Capturing packets")
                    fname = (
                        self.__saveData
                        + "kidpyCaptureData{0:%Y%m%d%H%M%S}.h5".format(
                            datetime.datetime.now()
                        )
                    )
                    print(fname)
                    if t < 60:
                        self.__udp.shortDataCapture(fname, 488 * t)
                    else:
                        self.__udp.LongDataCapture(fname, 488 * t)
                    print("Releasing Socket")
                    self.__udp.release()

            if opt == 6:  # Lo Sweep
                # valon should be connected and differentiated as part of bringing kidpy up.
                os.system("clear")
                print("LO Sweep")
                loSweep(self.valon, self.__udp, self.current_waveform, 400)

                pass

            if opt == 7:  # get system state
                exit()

            return 0


def main():
    k = kidpy()
    try:
        while 1:
            k.main_opt()
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
