"""
:Date: 2023-07-20
:Version: 2.0.0
:Authors: - Cody Roberson
          - Adrian Sinclair
          - Ryan Stephenson
          - Philip Mauskopf
          - Jack Sayers

kidpy is where users can interact with the mkid readout system. Simply launch with
.. codeblock::
    python kidpy.py


When an operation is selected, a command is created and published on a redis
pubsub channel. Any listening RFSOC(s) would then parse and execute the specified command
    
"""

import numpy as np
import sys, os
import redis
import json
import configparser
import time
import serial.tools.list_ports_linux
import udpcap
import datetime
import valon5009
import sweeps
import udp2
import data_handler
import matplotlib.pyplot as plt
import transceiver
import bias_board

# from datetime import date
# from datetime import datetime
import pdb
import glob
import logging
from time import sleep

### Logging ###
# Configures the logger such that it prints to a screen and file including the format
__LOGFMT = "%(asctime)s|%(levelname)s|%(filename)s|%(lineno)d|%(funcName)s|%(message)s"

logging.basicConfig(format=__LOGFMT, level=logging.DEBUG)
logger = logging.getLogger(__name__)
__logh = logging.FileHandler("./kidpy.log")
logging.root.addHandler(__logh)
logger.log(100, __LOGFMT)
__logh.flush()
__logh.setFormatter(logging.Formatter(__LOGFMT))
################


default_f_center = 6000

# for the ONR features
onr_repo_dir = os.path.expanduser("~") + "/onrkidpy"
onr_check = glob.glob(onr_repo_dir)
if np.size(onr_check) == 0:
    onr_flag = False
else:
    onr_flag = True
if onr_flag:
    sys.path.append(onr_repo_dir)
    import onrkidpy
    import onr_fit_lo_sweeps as fit_lo
    import onr_motor_control
    import subprocess

    motor = onr_motor_control.SKPR_Motor_Control()


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
    return False, None


def checkBlastCli(r, p):
    """
    Rudamentary "is the rfsoc control software running" check.
    """
    cmd = {"cmd": "ping", "args": []}
    cmd = json.dumps(cmd)
    r.publish("picard", cmd)
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


def write_fList(self, fList, ampList):
    """
    Function for writing tones to the rfsoc. Accepts both numpy arrays and lists.
    :param fList: List of desired tones
    :type fList: list
    :param ampList: List of desired amplitudes
    :type ampList: list
    .. note::
        fList and ampList must be the same size
    """
    log = logger.getChild("write_fList")
    f = fList
    a = ampList

    # Convert to numpy arrays as needed
    if isinstance(fList, np.ndarray):
        f = fList.tolist()
    if isinstance(ampList, np.ndarray):
        a = ampList.tolist()

    # Format Command based on provided parameters
    cmd = {}
    if len(f) == 0:
        cmd = {"cmd": "ulWaveform", "args": []}
    elif len(f) > 0 and len(a) == 0:
        a = np.ones_like(f).tolist()
        cmd = {"cmd": "ulWaveform", "args": [f, a]}
    elif len(f) > 0 and len(a) > 0:
        assert len(a) == len(
            f
        ), "Frequency list and Amplitude list must be the same dimmension"
        cmd = {"cmd": "ulWaveform", "args": [f, a]}
    else:
        log.error("Weird edge case, something went very wrong.....")
        return

    cmdstr = json.dumps(cmd)
    self.r.publish("picard", cmdstr)
    success, _ = wait_for_reply(self.p, "ulWaveform", max_timeout=10)
    if success:
        log.info("Wrote waveform.")
    else:
        log.error("FAILED TO WRITE WAVEFORM")


def menu(captions, options):
    """Creates menu for terminal interface
    inputs:
        list captions: List of menu captions
        list options: List of menu options
    outputs:
        int opt: Integer corresponding to menu option chosen by user"""
    log = logger.getChild("menu")
    print("\t" + captions[0] + "\n")
    for i in range(len(options)):
        print("\t" + "\033[32m" + str(i) + " ..... " "\033[0m" + options[i] + "\n")
    opt = None
    try:
        x = input("Option? ")
        opt = int(x)
    except KeyboardInterrupt:
        exit()
    except ValueError:
        print("Not a valid option")
        return 999999
    except TypeError:
        print("Not a valid option")
        return 999999
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
        self.udx1 = transceiver.Transceiver()
        self.udx1.connect("/dev/IFSLICE2")
        self.udx1.set_synth_out(default_f_center)
        # for v in self.__ValonPorts:
        #    self.valon = valon5009.Synthesizer(v.replace(' ', ''))

        self.bias = bias_board.Bias("/dev/BIASBOARD")

        self.__udp = udpcap.udpcap()
        self.current_waveform = []
        self.current_amplitude = []
        caption1 = "\n\t\033[95mKID-PY2 RFSoC Readout\033[95m"
        self.captions = [caption1]

        self.__main_opts = [
            "Upload firmware",
            "Initialize system & UDP conn",
            "Write test comb (single or multitone)",
            "Write stored comb from config file",
            "I <-> Q Phase offset [not functional yet]",
            "Take Raw Data",
            "LO Sweep",            
            "Bias Board Control",
            "IF Slice Control",
            "Exit",
        ]

        self.bias_caption = ["Bias board control."]
        self.__bias_opts = [
            "Get all I and V Values",
            "Get TES Channel I",
            "Get TES Channel V",
            "Set TES Channel I",
            "Set TES Channel V",
            "Set LNA Channel Vd",
            "Set LNA Channel Vg",
            "Set LNA Channel Id",
            "Return"
        ]

        self.if_caption = ["IF slice control."]
        self.__if_opts = [
            "Check connection",
            "Get loopback",
            "Set loopback",
            "Get synthesizer frequency",
            "Set synthesizer frequency",
            "Get LO output frequency",
            "Set LO output frequency",
            "Return"
        ]

    def get_tone_list(self):
        lo_freq = valon5009.Synthesizer.get_frequency(self.udx1, valon5009.SYNTH_B)
        tones = lo_freq * 1.0e6 + np.asarray(self.get_last_flist())
        return tones

    def get_last_flist(self):
        log = logger.getChild("kidpy.get_last_flist")
        cmd = json.dumps({"cmd": "get_last_flist", "args": []})
        self.r.publish("picard", cmd)
        status, data = wait_for_reply(self.p, "get_last_flist", 3)
        if status:
            return np.array(data)
        else:
            log.error("The rfsoc didn't return back our data")
            return None

    def get_last_alist(self):
        log = logger.getChild("kidpy.get_last_alist")
        cmd = json.dumps({"cmd": "get_last_alist", "args": []})
        self.r.publish("picard", cmd)
        status, data = wait_for_reply(self.p, "get_last_alist", 2)
        if status:
            return np.array(data)
        else:
            log.error("The rfsoc didn't return back our data")
            return None

    def main_opt(self):
        log = logger.getChild(__name__)
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
            if opt == 999999:
                pass
            if opt == 0:  # upload firmware
                os.system("clear")
                cmd = {"cmd": "ulBitstream", "args": []}
                cmdstr = json.dumps(cmd)
                self.r.publish("picard", cmdstr)
                self.r.set("status", "busy")
                print("Waiting for the RFSOC to upload it's bitstream...")
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
                    write_fList(self, [], [])
                else:
                    print(
                        "Waiting for the RFSOC to write single {} MHz Tone".format(
                            float(self.__customSingleTone) / 1e6
                        )
                    )
                    write_fList(self, [float(self.__customSingleTone)], [])

            if opt == 3:  # write stored comb
                os.system("clear")
                # not used

            if opt == 4:
                print("Not Implemented")

            """

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
                        + "kidpyCaptureData{0:%Y%m%d%H%M%S}.hd5".format(
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
            """
            if opt == 5: # collect raw data
                t_length = 0
                try:
                    t_length = int(input("How many seconds of data?: "))
                    print(t_length)
                except ValueError:
                    print("Error, not a valid Number")
                except KeyboardInterrupt:
                    return

                if t_length <= 0:
                    print("Can't sample 0 seconds")
                else:

                    f = self.get_last_flist()
                    t = time.strftime("%Y%m%d%H%M%S")
                    rfsoc1 = data_handler.RFChannel(
                    f"./ALICPT_RDF_{t}.hd5",
                    "192.168.3.40",
                    self.get_last_flist(),
                    self.get_last_alist(),
                    port=4096,
                    name="rfsoc2",
                    n_tones=len(f),
                    attenuator_settings=np.array([20.0, 10.0]),
                    tile_number=2,
                    rfsoc_number=2,
                    lo_sweep_filename="",
                    lo_freq=default_f_center
                    )
                                
                    udp2.capture([rfsoc1], sleep, t_length)

            if opt == 6:  # Lo Sweep
                # valon should be connected and differentiated as part of bringing kidpy up.
                os.system("clear")
                print("LO Sweep")
                
                try:
                    freq_center = float(input('Set frequency center (MHz): '))
                except ValueError:
                    print("Error, not a valid Number")
                except KeyboardInterrupt:
                    return
                
                try:        
                    freq_step = float(input('Set frequency step (MHz): '))
                except ValueError:
                    print("Error, not a valid Number")
                except KeyboardInterrupt:
                    return

                try:
                    nsteps = int(input('Set number of frequency steps: '))   
                except ValueError:
                    print("Error, not a valid Number")
                except KeyboardInterrupt:
                    return 
                
                if freq_center < 4000 or freq_center > 8000:
                    print("Center frequency must be between 4000 and 8000 MHz.")
                elif freq_step <= 0:
                    print("Frequency step must be > 0.")
                elif nsteps <= 0:
                    print("Number of frequency steps must be > 0.")
                else:  
                    filename = sweeps.loSweep(
                            self.udx1,
                            self.__udp,
                            self.get_last_flist(),
                            f_center=default_f_center,
                            freq_step=0,
                            N_steps=nsteps
                    )

                    # plot result
                    sweeps.plot_sweep(f"./{filename}.npy")

            if opt == 7: #bias board control
                #desired options
                #"Get all I and V Values",
                #"Get TES Channel I",
                #"Get TES Channel V",
                #"Set TES Channel I",
                #"Set TES Channel V",
                #"Set LNA Channel Vd",
                #"Set LNA Channel Vg",
                #"Set LNA Channel Id",
                #"Return"
                
                while True:                
                    bias_opt = menu(self.bias_caption, self.__bias_opts)

                    if bias_opt == 0:                
                        self.bias.getAllIV()
                    elif bias_opt == 1:
                        try:
                            TES_channel = int(input('TES bias channel? (1-4): '))
                        except ValueError:
                            print("Error, not a valid Number")
                        except KeyboardInterrupt:
                            return
                        TES_current = self.bias.get_iTES(TES_channel)
                        print("TES Bias Channel %d current: %.3f mA"%(TES_channel, TES_current))
                        print("\n\n")

                    elif bias_opt == 2:
                        try:
                            TES_channel = int(input('TES bias channel? (1-4): '))
                        except ValueError:
                            print("Error, not a valid Number")
                        except KeyboardInterrupt:
                            return
                        TES_voltage = self.bias.get_vTES(TES_channel)
                        print("TES Bias Channel %d voltage: %.3f uV"%(TES_channel, TES_voltage))
                        print("\n\n")

                    elif bias_opt == 3:
                        try:
                            TES_channel = int(input('TES bias channel? (1-4): '))
                        except ValueError:
                            print("Error, not a valid Number")
                        except KeyboardInterrupt:
                            return
                        
                        try:        
                            TES_current = float(input('TES current in [mA]? (0-12.5): '))
                        except ValueError:
                            print("Error, not a valid Number")
                        except KeyboardInterrupt:
                            return
                        self.bias.iSHUNT(TES_channel, TES_current)

                    elif bias_opt == 4:
                        try:
                            TES_channel = int(input('TES bias channel? (1-4): '))
                        except ValueError:
                            print("Error, not a valid Number")
                        except KeyboardInterrupt:
                            return
                        
                        try:        
                            TES_voltage = float(input('TES voltage in [uV]? (0-5): '))
                        except ValueError:
                            print("Error, not a valid Number")
                        except KeyboardInterrupt:
                            return
                        self.bias.vTES(TES_channel, TES_voltage)

                    elif bias_opt == 5:
                        try:
                            LNA_channel = int(input('LNA bias channel? (1-2): '))
                        except ValueError:
                            print("Error, not a valid Number")
                        except KeyboardInterrupt:
                            return
                        
                        try:        
                            LNA_Drain_voltage = float(input('LNA Drain voltage in [V]? : '))
                        except ValueError:
                            print("Error, not a valid Number")
                        except KeyboardInterrupt:
                            return
                        if LNA_Drain_voltage>4:
                            print("High voltage warning! A lower value is preferred.")
                        else:
                            self.bias.vLNA_D(LNA_channel, LNA_Drain_voltage)

                    elif bias_opt == 6:
                        try:
                            LNA_channel = int(input('LNA bias channel? (1-2): '))
                        except ValueError:
                            print("Error, not a valid Number")
                        except KeyboardInterrupt:
                            return
                        
                        try:        
                            LNA_Gain_voltage = float(input('LNA drain voltage in [V]? : '))
                        except ValueError:
                            print("Error, not a valid Number")
                        except KeyboardInterrupt:
                            return
                        if LNA_Gain_voltage>4:
                            print("High voltage warning! A lower value is preferred.")
                        else:
                            self.bias.vLNA_G(LNA_channel, LNA_Gain_voltage)

                    elif bias_opt == 7:
                        try:
                            LNA_channel = int(input('LNA bias channel? (1-2): '))
                        except ValueError:
                            print("Error, not a valid Number")
                        except KeyboardInterrupt:
                            return
                        
                        try:        
                            LNA_Drain_current = float(input('LNA Drain current in [mA]? : '))
                        except ValueError:
                            print("Error, not a valid Number")
                        except KeyboardInterrupt:
                            return
                        if LNA_Drain_current>50:
                            print("High current warning! A lower value is preferred.")
                        else:
                            self.bias.iLNA_D(LNA_channel, LNA_Drain_current)

                    elif bias_opt == 8:
                        break
            
            if opt == 8:# control IF board
                #"Check connection",
                #"Get loopback",
                #"Set loopback",
                #"Get synthesizer frequency",
                #"Set synthesizer frequency",
                #"Get LO output frequency",
                #"Set LO output frequency",
                #"Return"
                while True:
                    if_opt = menu(self.if_caption, self.__if_opts)
                    
                    if if_opt == 0:
                        # a bool is returned
                        connection_status = self.udx1.check_connection()
                        if connection_status:
                            print("IF Transceiver is connected.\n\n")
                        else:
                            print("IF Transceiver is NOT connected.\n\n")

                    elif if_opt == 1:
                        loopback_status = self.udx1.get_loopback()
                        if loopback_status:
                            print("Loopback is on.\n\n")
                        else:
                            print("Loopback is off.\n\n")

                    elif if_opt == 2:
                        try:
                            loopback_str = input('Set loopback ON? (True or False): ')
                        except ValueError:
                            print("Error, not a valid Number")
                        except KeyboardInterrupt:
                            return
                        if loopback_str == "True" or loopback_str == "1":
                            loopback_bool = True
                        elif loopback_str == "False" or loopback_str == "0":
                            loopback_bool = False
                        else:
                            loopback_bool =False
                            print("Wrong parameter. Loopback is off.")
                        loopback_status = self.udx1.set_loopback(loopback_bool)
                        if loopback_status:
                            print("Loopback is on.\n\n")
                        else:
                            print("Loopback is off.\n\n")
                    
                    elif if_opt == 3:
                        synth_freq = self.udx1.get_synth_ref()
                        print("Synthesizer (LO) frequency: %.6f MHz\n\n"%synth_freq)

                    elif if_opt == 4:
                        try:
                            synth_freq = float(input('Set Synthesizer Frequency in unit of [MHz] (10-600): '))
                        except ValueError:
                            print("Error, not a valid Number")
                        except KeyboardInterrupt:
                            return
                        synth_freq = self.udx1.set_synth_ref(synth_freq)
                        print("Synthesizer frequency is set to %.6f MHz\n\n"%synth_freq)

                    elif if_opt == 5:
                        synth_out_freq = float(self.udx1.get_synth_out())
                        print("Synthesizer (LO) output frequency: %.6f MHz\n\n"%synth_out_freq)

                    elif if_opt == 6:
                        try:
                            synth_out= input('Set Synthesizer (LO) output frequency in unit of [MHz] (2500-8600)?: ')
                        except ValueError:
                            print("Error, not a valid Number")
                        except KeyboardInterrupt:
                            return
                        synth_out_freq = self.udx1.set_synth_out(synth_out)
                        print("Synthesizer output frequency: %.6f MHz\n\n"%synth_out_freq)

                    elif if_opt == 7:
                        break

            if opt == 9:  # get system state
                self.bias.end()
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
