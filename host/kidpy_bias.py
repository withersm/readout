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
np.seterr(divide = 'ignore') 
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
import calibration as cal
import serial

# from datetime import date
# from datetime import datetime
import pdb
import glob
import logging
from time import sleep

#offline demod (temporary)
import ali_offline_demod as dm

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

def connect_beam_mapper(direc="/dev/ttyUSB0"):
    x = os.path.exists(direc)	

    if x:
        ser = serial.Serial(
                port=direc,
                baudrate=9600,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS
        )
    else:
        print("Serial port could not be found.")

    return ser

def isfloat(N): # allows string -> float conversion for input parameters
		try:
			float(N)
			return True
		except ValueError:
			print('error thrown')
			return False

def convert_units(N, unit, rev = False): # convert inches, mm --> steps
    N = float(N)
    if unit == "in":
        if rev:
            return round(N * 0.25 / 1000) # steps --> in
        return round(N / 0.25 * 1000) # in --> steps
    elif unit == "mm":
        if rev:
            return round(N * 6.35 / 1000) # steps --> mm
        return round(N / 6.35 * 1000) # mm --> steps
    else:
        return int(N) # int since steps must be whole numbers


def menu(captions, options):
    """Creates menu for terminal interface
    inputs:
        list captions: List of menu captions
        list options: List of menu options
    outputs:
        int opt: Integer corresponding to menu option chosen by user"""
    log = logger.getChild("menu")
    print(captions[0] + "\n")
    for i in range(len(options)):
        print("\t" + "\033[32m" + str(i) + " ..... " "\033[0m" + options[i])
    print('')
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

def top_menu(captions, setup_options, ts_data_options, vna_data_options, hardware_control_options, other_options):
    """Creates top menu for terminal interface
    inputs:
        list captions: List of menu captions
        list options: List of menu options
    outputs:
        int opt: Integer corresponding to menu option chosen by user"""
    log = logger.getChild("menu")
    print(captions[0] + "\n")
    
    print("\033[0m"+'Setup:')
    for i in range(len(setup_options)):
        print("\t" + "\033[32m" + str(i) + " ..... " "\033[0m" + setup_options[i])
    
    print('Time Series Data:')
    for i in range(len(ts_data_options)):
        print("\t" + "\033[32m" + str(i+len(setup_options)) + " ..... " "\033[0m" + ts_data_options[i])

    print('VNA-like Data:')
    for i in range(len(vna_data_options)):
        print("\t" + "\033[32m" + str(i+len(setup_options)+len(ts_data_options)) + " ..... " "\033[0m" + vna_data_options[i])

    print('Hardware Control:')
    for i in range(len(hardware_control_options)):
        print("\t" + "\033[32m" + str(i+len(setup_options)+len(ts_data_options)+len(vna_data_options)) + " ..... " "\033[0m" + hardware_control_options[i])
    
    print('Other:')
    for i in range(len(other_options)):
        print("\t" + "\033[32m" + str(i+len(setup_options)+len(ts_data_options)+len(vna_data_options)+len(hardware_control_options)) + " ..... " "\033[0m" + other_options[i]+'\n')


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
        self.__decimation_factor = 2**19 #don't leave this hardcoded; move to config
        self.__fr_V_scaling_factor = 5.69#(3.76+3.84) / 2 #V; don't leave this hardcoded; move to config

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
        print("Connecting to Transceiver")
        self.udx1 = transceiver.Transceiver()
        self.udx1.connect("/dev/IFSLICE2")
        self.udx1.set_synth_out(default_f_center)
        # for v in self.__ValonPorts:
        #    self.valon = valon5009.Synthesizer(v.replace(' ', ''))

        print("Connecting to bias board")
        self.bias = bias_board.Bias("/dev/BIASBOARD")

        self.__udp = udpcap.udpcap()
        self.current_waveform = []
        self.current_amplitude = []
        caption1 = "\n\033[95mAliCPT-1 RFSoC Readout\033[95m"
        self.captions = [caption1]

        self.__main_opts = [
            "Upload firmware",
            "Initialize system & UDP conn",
            "Write test comb (single or multitone)",
            "Write stored comb from config file",
            "I <-> Q Phase offset [not functional yet]",
            "Tone Initalization",
            "Take Raw Data",
            "LO Sweep", 
            "Find Frequencies",
            "Compute Calibration (η)",           
            "Bias Board Control",
            "IF Slice Control",
            "Initialize PMOD registers",
            "Exit",
        ]

        self.__setup_options = [
            "Set attenuation",
            "Bias 4K LNA(s)",
            "Upload firmware",
            "Initalize system & UDP conn",
            "Preset Flux Ramp"
        ]
        
        self.__ts_data_options = [
            "New tone initalization",
            "Load tone initalization",
            "Take raw data (remember to turn on flux ramp first)",
            "Demod data (software)"
        ]

        self.__vna_data_options = [
            "Write test comb (single or multitone)",
            "Write comb from file",
            "LO sweep",
            "Find frequencies"
        ]

        self.__hardware_control_options = [
            'Bias board control',
            'IF slice control',
            'Flux ramp control'
        ]

        self.__other_options = [
            'Exit'
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
            "Get Input Attenuation",
            "Set Input Attenuation",
            "Get Output Attenuation",
            "Set Output Attenuation",
            "Return"
        ]

        self.fr_caption = ["Flux ramp board control."]
        self.__fr_opts = [
            "Get Flux Ramp Settings",
            "Basic Configuration (clk divisor and n packets)",
            "Advanced Configuration (voltage target and frequency target)",
            "Flux Ramp Off",
            "Return"
        ]

        #setup tagging for data management
        self.get_data_tag()
        print(f'Current tone initialization dir. / timestream datatag: {self.__dataTag}')

        #setup LO sweep parameters for auto tone initialization
        self.__freqStep = float(config['TONEINIT']['freqStep'])
        self.__nStep = int(config['TONEINIT']['nStep'])
        
    def get_data_tag(self):
        """
        grab the name of the most recent tone initialization set for use as a data tag in timestream data
        """
        last_tone_directory = max([dir for dir in glob.glob(f'{self.__saveData}/tone_initializations/*', recursive=False) if os.path.isdir(dir)],key=os.path.getmtime)
        self.__dataTag = last_tone_directory.split('/')[-1]
    
    def change_data_tag(self,path):
        self.__dataTag = path.split('/')[-1]
    
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
        
    def configure_pmod(self, clock_divisor, n_packets):
   
        #os.system("clear")
        print("Setting pmod registers.")
        cmd = {"cmd": "set_pmod", "args": [clock_divisor,n_packets]}
        cmdstr = json.dumps(cmd)
        self.r.publish("picard", cmdstr)
        success, _ = wait_for_reply(self.p, "set_pmod", max_timeout=2)

    def read_pmod(self):
        log = logger.getChild("kidpy.read_pmod")
        cmd = json.dumps({"cmd": "read_pmod", "args": []})
        self.r.publish("picard", cmd)
        status, data = wait_for_reply(self.p, "read_pmod", 3)
        if status:
            clk_divisor = int(data[0])
            n_packets = int(data[1])
            return clk_divisor, n_packets
        else:
            log.error("The rfsoc didn't return back our data.")
            return None
   
    def set_flux_ramp_advanced(self):
        print('Setting flux ramp.')

        try:
            target_V = float(input('Set ramp voltage (V): '))
        except ValueError:
            print("Error, not a valid number.")
        except KeyboardInterrupt:
            return
        
        try:
            target_f = float(input('Set ramp frequency (Hz): '))
        except ValueError:
            print("Error, not a valid number.")
        except KeyboardInterrupt:
            return
        
        #compute n packets
        n_packets = int(256e6 / (self.__decimation_factor * target_f))
        closest_available_freq = self.estimate_flux_ramp_freq(n_packets)
        print(f'Closest available frequency: {closest_available_freq} Hz')
        print(f'n packets: {n_packets}')
           
        n_bits_for_target_V = (2**12 / self.__fr_V_scaling_factor) * target_V
        f_clk = n_bits_for_target_V * closest_available_freq
        clock_divisor = int(125e6 / f_clk)
        closest_available_V = self.estimate_flux_ramp_V(clock_divisor, closest_available_freq)
        print(f'Closest available V: {closest_available_V} V')
        print(f'Clock divisor: {clock_divisor}')
        
        self.configure_pmod(clock_divisor, n_packets)

    def set_flux_ramp_basic(self):

        try:
            clock_divisor = float(input('Set clock divisor: '))
        except ValueError:
            print("Error, not a valid number.")
        except KeyboardInterrupt:
            return
        
        try:
            n_packets = float(input('Set number of packets: '))
        except ValueError:
            print("Error, not a valid number.")
        except KeyboardInterrupt:
            return

        self.configure_pmod(clock_divisor, n_packets)

    def estimate_flux_ramp_freq(self, n_packets):
        freq_estimated = 256e6 / (n_packets * self.__decimation_factor)

        return freq_estimated
    
    def estimate_flux_ramp_V(self, clk_divisor, freq_estimated = None):
        """
        Note that this is only as good as the value set in self.__fr_V_scaling_factor
        """

        f_clk_estimated = 125e6 / clk_divisor
        if freq_estimated != None:
            n_bits_estimated = f_clk_estimated / freq_estimated
        else: #if reset signal is off, max voltage is output at 2**12 bits
            n_bits_estimated = 2**12
        V_estimated = n_bits_estimated * (self.__fr_V_scaling_factor / 2**12)

        return V_estimated
   
    
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
                    + "\r\nCouldn't connect to redis-server double check its running and the generalConfig is correct"
                    + "\033[0m"
                )
            #opt = menu(self.captions, self.__main_opts)
            opt = top_menu(self.captions, self.__setup_options, self.__ts_data_options, self.__vna_data_options, self.__hardware_control_options, self.__other_options)
            if conStatus == False:
                resp = input(
                    "Can't connect to redis server, do you want to continue anyway? [y/n]: "
                )
                if resp != "y":
                    exit()
            if opt == 999999:
                pass
            
            if opt == 0: #set attenuation
                try:
                    new_atten_in = input('Set input attenuation (dB): ')
                except ValueError:
                    print("Error, not a valid number.")
                except KeyboardInterrupt:
                    return
                atten_in = self.udx1.set_rf_in(new_atten_in)
                resp = self.udx1.get_rf_in()
                print(f'Input Attenuation: {resp} dB')

                try:
                    new_atten_out = input('Set output attenuation (dB): ')
                except ValueError:
                    print("Error, not a valid number.")
                except KeyboardInterrupt:
                    return
                atten_in = self.udx1.set_rf_out(new_atten_out)
                resp = self.udx1.get_rf_out()
                print(f'Output Attenuation: {resp} dB')                

            if opt == 1: #bias 4K LNA(s)
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

                #display bias results
                self.bias.getAllIV()
            
            if opt == 2:  # upload firmware
                #os.system("clear")
                cmd = {"cmd": "ulBitstream", "args": []}
                cmdstr = json.dumps(cmd)
                self.r.publish("picard", cmdstr)
                self.r.set("status", "busy")
                print("Waiting for the RFSOC to upload its bitstream...")
                if wait_for_free(self.r, 0.75, 25):
                    print("Done")

            if opt == 3:  # Init System & UDP conn.
                #os.system("clear")
                print("Initializing System and UDP Connection")
                cmd = {"cmd": "initRegs", "args": []}
                cmdstr = json.dumps(cmd)
                self.r.publish("picard", cmdstr)
                if wait_for_free(self.r, 0.5, 5):
                    print("Done")

            if opt == 4:  # Set pmod
                self.set_flux_ramp_advanced()
                
                """
                os.system("clear")
                print("Setting pmod registers.")
                a = input("clock div?")
                b = input("n packets")  
                cmd = {"cmd": "set_pmod", "args": [a,b]}
                cmdstr = json.dumps(cmd)
                self.r.publish("picard", cmdstr)
                success, _ = wait_for_reply(self.p, "set_pmod", max_timeout=2)
                """
            if opt == 5: # new tone initalization
                print("Beginning auto calibration procedure.\n")
                start_time = time.strftime("%Y%m%d%H%M%S")
                
                #Ask for center frequency
                print('1) Set center frequency...')
                try:
                    freq_center = float(input('Set frequency center (MHz): '))
                except ValueError:
                    print("Error, not a valid Number")
                except KeyboardInterrupt:
                    return
                
                if freq_center < 4000 or freq_center > 8000:
                    print("Center frequency must be between 4000 and 8000 MHz.\nAborting...")
                    return
                
                #Create Directory
                os.mkdir(f'{self.__saveData}/tone_initializations/fcenter_{freq_center}_{start_time}')
                self.get_data_tag()

                #Load Test Comb
                print('\n2) Loading test comb...')
                
                print("Waiting for the RFSOC to finish writing the full comb")
                write_fList(self, [], [])
                print(self.get_last_flist())

                #Initial LO Sweep
                print('\n3) Performing initial LO sweep...')
                                
                initial_lo_filename = sweeps.loSweep(
                        self.udx1,
                        self.__udp,
                        self.get_last_flist(),
                        f_center=freq_center,
                        freq_step=self.__freqStep,
                        N_steps=self.__nStep,
                        file_path=f'{self.__saveData}/tone_initializations/{self.__dataTag}',
                        file_tag='initial'
                )

                # plot result
                print(initial_lo_filename)
                sweeps.plot_sweep(f"{initial_lo_filename}.npy")

                synth_freq = self.udx1.set_synth_out(freq_center)
                print("\nFinished sweep. Setting LO back to %.6f MHz\n\n"%synth_freq)
                

                #Find Freqs.
                print('\n4) Finding tones frequencies from initial lo sweep...')
                print(f'Loading: {initial_lo_filename}.npy')
                freqs, mags = cal.find_minima(f'{initial_lo_filename}.npy', plot=True, figpath=f'{self.__saveData}/tone_initializations/{self.__dataTag}', figname=f'peakfind_{initial_lo_filename.split("/")[-1]}')

                initial_freq_list_filename = f'{self.__saveData}/tone_initializations/{self.__dataTag}/freq_list_{initial_lo_filename.split("/")[-1]}'
                cal.save_array(freqs, initial_freq_list_filename)
                                
                #Load Targeted Comb
                print('\n5) Loading targeted comb...')
                print(f'Loading tones from: {initial_freq_list_filename}.npy')

                farray = cal.load_array(f'{initial_freq_list_filename}.npy')

                print(farray.real)

                lo = float(initial_freq_list_filename.split("_")[-2])*1e6
                print(lo)
                write_fList(self, farray.real - lo, []) #turned off for testing because I'm not looking at resonators and peak locations are incredibly random

                #Targeted LO Sweep
                print('\n6) Performing targeted LO sweep...')
                                
                targeted_lo_1_filename = sweeps.loSweep(
                        self.udx1,
                        self.__udp,
                        self.get_last_flist(),
                        f_center=freq_center,
                        freq_step=self.__freqStep,
                        N_steps=self.__nStep,
                        file_path=f'{self.__saveData}/tone_initializations/{self.__dataTag}',
                        file_tag='targeted_1'
                )

                # plot result
                print(targeted_lo_1_filename)
                sweeps.plot_sweep(f"{targeted_lo_1_filename}.npy")

                synth_freq = self.udx1.set_synth_out(freq_center)
                print("\nFinished sweep. Setting LO back to %.6f MHz\n\n"%synth_freq)

                #Find Freqs.
                print('\n7) Finding tones frequencies from initial lo sweep...')
                print(f'Loading: {targeted_lo_1_filename}.npy')
                freqs, mags = cal.find_minima(f'{targeted_lo_1_filename}.npy', plot=True, figpath=f'{self.__saveData}/tone_initializations/{self.__dataTag}', figname=f'peakfind_{targeted_lo_1_filename.split("/")[-1]}')

                targeted_freq_list_1_filename = f'{self.__saveData}/tone_initializations/{self.__dataTag}/freq_list_{targeted_lo_1_filename.split("/")[-1]}'
                cal.save_array(freqs, targeted_freq_list_1_filename)

                #Load Targeted Comb
                print('\n8) Loading targeted comb...')
                print(f'Loading tones from: {targeted_freq_list_1_filename}.npy')

                farray = cal.load_array(f'{targeted_freq_list_1_filename}.npy')

                print(farray.real)

                lo = float(targeted_freq_list_1_filename.split("_")[-2])*1e6
                print(lo)
                write_fList(self, farray.real - lo, []) #turned off for testing because I'm not looking at resonators and peak locations are incredibly random

                #Targeted LO Sweep
                print('\n9) Performing targeted LO sweep...')
                                
                targeted_lo_2_filename = sweeps.loSweep(
                        self.udx1,
                        self.__udp,
                        self.get_last_flist(),
                        f_center=freq_center,
                        freq_step=self.__freqStep,
                        N_steps=self.__nStep,
                        file_path=f'{self.__saveData}/tone_initializations/{self.__dataTag}',
                        file_tag='targeted_2'
                )

                # plot result
                print(targeted_lo_2_filename)
                sweeps.plot_sweep(f"{targeted_lo_2_filename}.npy")

                synth_freq = self.udx1.set_synth_out(freq_center)
                print("\nFinished sweep. Setting LO back to %.6f MHz\n\n"%synth_freq)                
                
                print(f'New tone initialization dir. / timestream datatag: {self.__dataTag}')

            if opt == 6: #load tone initalization
                try:
                    init_filepath = input('Enter directory path of initalization: ')
                except KeyboardInterrupt:
                    return
                
                freq_file = glob.glob(f'{init_filepath}/freq_list_lo_sweep_targeted_1_*')[0]


                print(f'Loading tones from: {freq_file}')



                farray = cal.load_array(f'{freq_file}')

                print(farray.real)

                lo = float(freq_file.split("_")[-2])*1e6
                print(lo)
                write_fList(self, farray.real - lo, []) #turned off for testing because I'm not looking at resonators and peak locations are incredibly random


                self.change_data_tag(init_filepath)
                print(f'Loaded tone initalization dir. / timestream datatag: {self.__dataTag}')

            if opt == 7: #take raw data
                try:
                    data_taking_opt = int(input("Data taking only [0], load curve [1], or beam map [2]?: "))
                    if data_taking_opt == 0:
                    # data taking without changing bias
                        t_length = 0
                        try:
                            t_length = int(input("How many seconds of data?: [0] for continuous data taking: "))
                            print(t_length)
                            if t_length == 0:
                                print("Starting continuous data taking... Press [y]+ENTER to stop...\n")
                        except ValueError:
                            print("Error, not a valid Number")
                        except KeyboardInterrupt:
                            return
                        
                        def data_taking_fn(t_length):
                            # once this function returns, the data taking will stop
                            # :t_length data taking length in unit of [seconds]
                            # if t_length <= 0: the data taking will stop with [y]+ENTER
                            # if t_length  > 0: the function run the sleep function
                            if t_length <= 0:
                                while True:
                                    trigger = str(input("Do you wish to stop data taking?:[y]"))
                                    if trigger =='y':
                                        return
                            else:
                                time.sleep(t_length)
                        
                        f = self.get_last_flist()
                        t = time.strftime("%Y%m%d%H%M%S")
                        rfsoc1 = data_handler.RFChannel(
                            f"{self.__saveData}/time_streams/ts_toneinit_{self.__dataTag}_t_{t}.hd5",  #f"./ALICPT_RDF_{t}.hd5"
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
                        
                        udp2.capture([rfsoc1], data_taking_fn, t_length)
                    if data_taking_opt == 1:
                        #print("Beginning auto calibration procedure.\n")
                        start_time = time.strftime("%Y%m%d%H%M%S")
                        
                        #Create Directory
                        directory = f'{self.__saveData}/IV_data/toneinit_{self.__dataTag}_t_{start_time}'
                        os.mkdir(directory)
                                       
                    
                        # load curve measurement
                        #bias_channel = list(input("Which channels?[1234]/[1]/[34]: "))
                        bias_channel = int(input("Enter channel [1,2,3,4]: "))
                        if bias_channel != 1 and bias_channel != 2 and bias_channel != 3 and bias_channel != 4:
                            print('Invalid bias channel.')
                            return
                        print(bias_channel)
                        print('Note: Ibias start > Ibias end for a standard IV curve.')
                        normal_bias = float(input("Set an initial high bias to drive the TES normal (mA): "))
                        normal_t = float(input('Set a time to remain at the high bias (s): '))
                        bias_start = float(input("Enter Ibias start (mA): "))
                        bias_end = float(input("Enter Ibias end (mA): "))
                        bias_step = float(input("Enter Ibias step (mA): "))
                        data_t = float(input("How long data taking at one bias step? [s]: "))
                
                        def loadcurve_fn(bias_channel, bias_start, bias_end, bias_step, normal_bias, normal_t, data_t):
                            
                            #compute pot step using bias_step
                            #pot_step = int(number_replace_this * bias_step)
                            pot_step = int(bias_step)

                            t = time.strftime("%Y%m%d%H%M%S")
                            #open datafile
                            bias_file = open(f'{directory}/bias_data_{t}.txt','w')
                            
                            #record start time with no current; wait for 10s to stabilize system
                            t = time.time()
                            f"{t} 0\n"                            
                            time.sleep(10)                            
                            
                            #set and record the normal_bias to drive the TES normal
                            self.bias.iSHUNT(int(bias_channel), float(normal_bias))
                            normal_bias_actual = self.bias.get_iTES(bias_channel)
                            t = time.time()
                            bias_file.write(f"{t} {normal_bias_actual}\n")
                            time.sleep(normal_t)
                            
                            #set and record the start bias
                            self.bias.iSHUNT(int(bias_channel), float(bias_start))
                            ibias_actual = float(self.bias.get_iTES(bias_channel))
                            t = time.time()
                            bias_file.write(f"{t} {ibias_actual}\n")
                            time.sleep(normal_t)

                            #get the pot position
                            pot_pos = int(self.bias.get_wiper(bias_channel)) #initial pot position
                            print(pot_pos)

                            #begin decreasing the pot
                            pot_pos -= pot_step
                            print('pot_pos: '+str(pot_pos))
                            print('ibias_actual: '+str(ibias_actual))
                            print('bias_end: '+str(bias_end))
                            while (ibias_actual >= bias_end):
                                print('triggered while loop')
                                print(f'ibias = {ibias_actual} mA')
                                self.bias.set_wiper(bias_channel,pot_pos)
                                ibias_actual = self.bias.get_iTES(int(bias_channel))
                                t = time.time()
                                print("recording bias")
                                bias_file.write(f"{t} {ibias_actual}\n")
                                time.sleep(data_t)

                                pot_pos -= pot_step
                            
                            bias_file.close()
                            self.bias.iSHUNT(bias_channel, 0)
                            return
                        
                        
                            
                        
                        """
                        def loadcurve_fn(bias_channel, bias_start, bias_end, bias_step, data_t):
                            # two for-loops for mapping channels and bias
                            # bias voltage is output to a txt file
                            if bias_start > bias_end:
                                #bias_points = np.linspace(bias_end, bias_start,int((bias_start-bias_end)/bias_step)+1, endpoint=True)
                                bias_points_reversed = np.arange(bias_end, bias_start+bias_step, bias_step)
                                bias_points = np.flip(bias_points_reversed)
                            elif bias_start < bias_end:
                                bias_points = np.arange(bias_start, bias_end+bias_step,bias_step)
                                #bias_points = np.linspace(bias_start, bias_end, int((bias_start-bias_end)/bias_step)+1, endpoint=True)
                            t = time.strftime("%Y%m%d%H%M%S")
                            #bias_file = open(f"./Bias_data_{t}.txt", 'w')
                            bias_file = open(f'{directory}/bias_data_{t}.txt','w')
                            for bias_i in bias_points:
                                for chan in bias_channel:
                                    self.bias.iSHUNT(int(chan), float(bias_i))
                                    actual_ibias = self.bias.get_iTES(int(chan))
                                    #print(f'setting bias to {int(bias_i)}')
                                    
                                    
                                    #self.bias.vTES(int(chan), bias_v*1e6)

                                t = time.time()
                                print("recording bias")
                                bias_file.write(f"{t} {actual_ibias}\n")
                                time.sleep(data_t)
                            bias_file.close()
                            print('taking ts data')
                            return
                        """

                        f = self.get_last_flist()
                        t = time.strftime("%Y%m%d%H%M%S")
                        rfsoc1 = data_handler.RFChannel(
                            f"{directory}/ts_toneinit_{self.__dataTag}_t_{t}.hd5",
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
                                
                        udp2.capture([rfsoc1], loadcurve_fn, bias_channel, bias_start, bias_end, bias_step, normal_bias, normal_t, data_t)
                        print('setting bias back to 0')

                    if data_taking_opt == 2:
                        
                        start_time = time.strftime("%Y%m%d%H%M%S")
                        
                        #Create Directory
                        directory = f'{self.__saveData}/beam_map_data/toneinit_{self.__dataTag}_t_{start_time}'
                        os.mkdir(directory)

                        ser = connect_beam_mapper()

                        try:
                            x_min = int(input("x_min: "))
                            y_min = int(input("y_min: "))
                            x_max = int(input("x_max: "))
                            y_max = int(input("y_max: "))
                            delta_x = int(input("delta_x: "))
                            delta_y = int(input("delta_y: "))
                            delta_t = int(input("delta_t: "))
                            
                        except ValueError:
                            print("Error, not a valid Number")
                        except KeyboardInterrupt:
                            return
                        
                        def raster(ser, x_min, y_min, x_max, y_max, delta_x, delta_y, delta_t):
                            # Initialize current position
                            current_x, current_y = x_min, y_min

                            # Open log file
                            with open(f'{directory}/beam_map_data_{t}.txt', 'a+') as logfile:
                                logfile.write(f"start, end, x, y\n")
                            # Iterate over y range
                                for y in range(y_min, y_max + delta_y, delta_y):
                                    # Determine x range: forward if y is even step away from y_min, reverse if odd
                                    if (y - y_min) // delta_y % 2 == 0:
                                        x_range = range(x_min, x_max + delta_x, delta_x)
                                    else:
                                        x_range = range(x_max, x_min - delta_x, -delta_x)

                            
                                
                                    for x in x_range:
                                        
                                        # Move to the next position (convert steps back to mm for position function)
                                        local_command = 'C '
                                        if isfloat(x):
                                            pos1_conv = convert_units(float(x), "mm")
                                            local_command += "IA1M" + str(pos1_conv) + ","
                                        if isfloat(x):
                                            pos2_conv = convert_units(float(y), "mm")
                                            local_command += "IA2M" + str(pos2_conv) + ","
                                        local_command += "R"
                                        local_command = local_command.encode("utf-8")

                                            # Log movement start time
                                        start_time = time.time()

                                        ser.write(local_command)

                                        while True: # read timestamp and position into log file while motor is moving
                                            ser.write(b'V')
                                            time.sleep(0.05)
                                            status = ""
                                            while ser.inWaiting() > 0:
                                                status = ser.read(ser.inWaiting()).decode("utf-8").strip()
                                            if status == "^" or status == "R":
                                                break

                                        # Log movement end time
                                        end_time = time.time()

                                        # Wait for delta_t seconds at the new position
                                        time.sleep(delta_t)
                                        
                                        # Log the movement
                                        logfile.write(f"{start_time}, {end_time}, {x}, {y}\n")
                                        logfile.flush()

                        f = self.get_last_flist()
                        t = time.strftime("%Y%m%d%H%M%S")
                        rfsoc1 = data_handler.RFChannel(
                            f"{directory}/ts_toneinit_{self.__dataTag}_t_{t}.hd5",
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
                                
                        udp2.capture([rfsoc1], raster, ser, x_min, y_min, x_max, y_max, delta_x, delta_y, delta_t)
                        print('setting bias back to 0')
                        
                except ValueError:
                    print("Error, not a valid Number")
                except KeyboardInterrupt:
                    return
                
            if opt == 8: #demod data (software)
                try:
                    file_opt = int(input('[0] Demod most recent time stream; [1] Demod user selected time stream '))
                    demod_opt = int(input('[0] Simple demod (removes flux ramp) [1] Chop demod (removes flux ramp and optical chop) ')                            )
                except ValueError:
                    print("Error, not a valid Number")
                except KeyboardInterrupt:
                    return
                
                if file_opt == 0: #most recent file
                    list_of_files = glob.glob(f'{self.__saveData}/time_streams/*.hd5') # * means all if need specific format then *.csv
                    
                    file = max(list_of_files, key=os.path.getctime)
                    print(f'Demoding {file}...')

                    tone_init_path = f'{self.__saveData}/tone_initializations'
                    ts_path = f'{self.__saveData}/time_streams'
                    filename = file.split('.')[-1]

     



                
                elif file_opt == 1:
                    print(f'Working directory: {self.__saveData}/time_streams')
                    try:
                        directory_opt = int(input('[0] Enter a file in this directory; [1] Enter a file in any directory (full path required )'))
                    except ValueError:
                        print("Error, not a valid Number")
                    except KeyboardInterrupt:
                        return
                    
                    if directory_opt == 0:
                        print('Available files: ')
                        list_of_files = glob.glob(f'{self.__saveData}/time_streams/*.hd5')



                        pretty_list = np.reshape(list_of_files, (len(list_of_files),1))

                        print(pretty_list)

                    elif directory_opt ==1:
                        pass

                    try:
                        file = input('File name (include full path): ')
                    except KeyboardInterrupt:
                        return
                    
                    split = file.split('/')
                    
                    tone_init_path = f'/{split[1]}/{split[2]}/{split[3]}/tone_initializations'

                    ts_path = f'/{split[1]}/{split[2]}/{split[3]}/time_streams'

                    filename = split[-1]

                    print(f'tone_init_path: {tone_init_path}')
                    print(f'ts_path: {ts_path}')
                    print(f'file name: {filename}')



                if demod_opt == 0:    
                    processed = dm.full_demod_process(filename, 
                                                    f_sawtooth=15, 
                                                    method='fft', 
                                                    n=0, 
                                                    channels='all',
                                                    start_channel=0,
                                                    stop_channel=1000,
                                                    tone_init_path = tone_init_path, 
                                                    ts_path = ts_path,
                                                    display_mode='terminal') 
                    


                elif demod_opt == 1:
                    print('Not implemented yet.')
                  
                """
                fig_fr, ax_fr = plt.subplots(1)
                ax_fr.plot(processed['fr t'], processed['fr data'])
                ax_fr.show()
                """

                fig_demod, ax_demod = plt.subplots(1)
                channel_count = 0
                for ch in range(len(processed['demod data'])):
                    ax_demod.plot(processed['demod t'],processed['demod data'][ch]-np.average(processed['demod data'][ch])+0.1*channel_count,'-')
                    ax_demod.set_xlabel('$t$ (s)')
                    ax_demod.set_ylabel('Phase ($n_{\\Phi_0})$')
                    ax_demod.set_title(file)
                    fig_demod.show()
                    channel_count += 1


            if opt == 9:  # Write test comb
                prompt = input("Full test comb? y/n ")
                #os.system("clear")
                if prompt == "y":
                    print("Waiting for the RFSOC to finish writing the full comb")
                    write_fList(self, [], [])
                    print(self.get_last_flist())
                    #print(self.get_tone_list())
                else:
                    print(
                        "Waiting for the RFSOC to write single {} MHz Tone".format(
                            float(self.__customSingleTone) / 1e6
                        )
                    )
                    write_fList(self, [float(self.__customSingleTone)], [])

            if opt == 10:  # write comb from file
                #os.system("clear")
                print("Waiting for the RFSOC to finish writing the full comb")
                try:
                    option = int(input('[0] Use most recent manual frequency list, [1] Input frequency list filename: '))  
                except ValueError:
                    print("Error, not a valid Number")
                except KeyboardInterrupt:
                    return

                if option == 0:
                    list_of_files = glob.glob(f'{self.__saveData}/frequency_lists_manual/*.npy')
                    latest_file = max(list_of_files, key=os.path.getctime)
                    print(f'Loading: {latest_file}')


                    farray = cal.load_array(latest_file)

                    print(farray.real)

                    lo = float(latest_file.split("_")[-2])*1e6
                    print(lo)
                    write_fList(self, farray.real - lo, [])


                elif option == 1:
                    filename = input('Filename (full path): ')

                    farray = cal.load_array(filename)

                    print(farray.real)

                    lo = float(filename.split("_")[-2])*1e6
                    print(lo)
                    write_fList(self, farray.real - lo, [])
                

                
                #cal.load_array()
                
                
                #write_fList(self, [100e6, 150e6, 175e6], [])
                # not used
                    
            if opt == 11: #LO sweep
                # valon should be connected and differentiated as part of bringing kidpy up.
                os.system("clear")
                print("LO Sweep")
                
                """
                try:
                    sweep_type = int(input('[0] Initial sweep, [1] Targeted sweep: '))
                except ValueError:
                    print("Error, not a valid Number")
                except KeyboardInterrupt:
                    return
                """
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
                #elif freq_step <= 0:
                #    print("Frequency step must be > 0.")
                elif nsteps <= 0:
                    print("Number of frequency steps must be > 0.")
                else:  
                    filename = sweeps.loSweep(
                            self.udx1,
                            self.__udp,
                            self.get_last_flist(),
                            f_center=freq_center,
                            #f_center=default_f_center,
                            freq_step=freq_step,
                            N_steps=nsteps,
                            file_path=f'{self.__saveData}/lo_sweeps_manual',
                            file_tag='manual'
                    )

                    # plot result
                    print(filename)
                    sweeps.plot_sweep(f"{filename}.npy")

                    synth_freq = self.udx1.set_synth_out(freq_center)
                    print("Finished sweep. Setting LO back to %.6f MHz\n\n"%synth_freq)

            if opt == 12: #find frequencies -> need to fix the filepathing
                """
                try:
                    find_type = int(input('[0] Initial Peak Find, [1] Targeted Peak Find: '))  
                except ValueError:
                    print("Error, not a valid Number")
                except KeyboardInterrupt:
                    return
                """
               
                try:
                    option = int(input('[0] Use most recent manual LO sweep, [1] Input LO sweep filename: '))  
                except ValueError:
                    print("Error, not a valid Number")
                except KeyboardInterrupt:
                    return

                #if find_type == 0:    
                if option == 0:
                    list_of_files = glob.glob(f'{self.__saveData}/lo_sweeps_manual/*.npy')
                    latest_file = max(list_of_files, key=os.path.getctime)
                    print(f'Loading: {latest_file}')
                    freqs, mags = cal.find_minima(latest_file, plot=True) #not saving plot right now

                    filename_split = latest_file.split("_")

                    cal.save_array(freqs, f'{self.__saveData}/frequency_lists_manual/freqs_fcenter_{filename_split[-2]}_{filename_split[-1]}')
                    
                    return


                elif option == 1:
                    try:
                        filename = input('File Name (full path): ')
                    except KeyboardInterrupt:
                        return

                    freqs, mags = cal.find_minima(filename, plot=True) #not saving plot right now

                    filename_split = filename.split("_")

                    cal.save_array(freqs, f'{self.__saveData}/frequency_lists_manual/freqs_fcenter_{filename_split[-2]}_{filename_split[-1]}')

                    return

                else:
                    print("Not a valid option.")
                    return
                
                """
                elif find_type == 1:
                    if option == 0:
                        list_of_files = glob.glob('./lo_sweeps/*.npy')
                        latest_file = max(list_of_files, key=os.path.getctime)
                        print(f'Loading: {latest_file}')
                        freqs = cal.find_targeted_minima(latest_file)

                        filename_split = latest_file.split("_")

                        cal.save_array(freqs, f'./frequency_lists/freqs_fcenter_{filename_split[-2]}_{filename_split[-1]}')
                        return


                    elif option == 1:
                        filename = input('File Name: ')

                        freqs = cal.find__targeted_minima('./lo_sweeps/'+filename)

                        filename_split = filename.split("_")

                        cal.save_array(freqs, f'./frequency_lists/freqs_fcenter_{filename_split[-2]}_{filename_split[-1]}')

                        return

                    else:
                        print("Not a valid option.")
                        return
                    
                else:
                    print("Not a valid option.")
                    return
                """
            
            if opt == 13: #bias board control
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
                    print('\n')
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
                        TES_voltage = self.bias.get_vTES (TES_channel)
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
                            TES_current = float(input('TES current in [mA]? (0-15): '))
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
                            TES_voltage = float(input('TES voltage in [uV]? (0-5V): '))
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
            
            if opt == 14:# control IF board
                #"Check connection",
                #"Get loopback",
                #"Set loopback",
                #"Get synthesizer frequency",
                #"Set synthesizer frequency",
                #"Get LO output frequency",
                #"Set LO output frequency",
                #"Return"
                while True:
                    print('\n')
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
                        """Get Input Attenuation"""
                        atten_in = float(self.udx1.get_rf_in())
                        print(f'Input attenuation: {atten_in} dB')

                    elif if_opt == 8:
                        """Set Input Attenuation"""
                        try:
                            new_atten_in = input('Set input attenuation (dB): ')
                        except ValueError:
                            print("Error, not a valid number.")
                        except KeyboardInterrupt:
                            return
                        atten_in = self.udx1.set_rf_in(new_atten_in)
                        resp = self.udx1.get_rf_in()
                        print(f'Input Attenuation: {resp} dB')

                    elif if_opt == 9:
                        """Get Output Attenuation"""
                        atten_out = float(self.udx1.get_rf_out())
                        print(f'Output attenuation: {atten_out} dB')

                    elif if_opt == 10:
                        """Set Output Attenuation"""
                        try:
                            new_atten_out = input('Set output attenuation (dB): ')
                        except ValueError:
                            print("Error, not a valid number.")
                        except KeyboardInterrupt:
                            return
                        atten_in = self.udx1.set_rf_out(new_atten_out)
                        resp = self.udx1.get_rf_out()
                        print(f'Output Attenuation: {resp} dB')

                    elif if_opt == 11:
                        break
                        
            if opt == 15:
                print('Flux Ramp Control')
                while True:
                    print('\n')
                    fr_opt = menu(self.fr_caption, self.__fr_opts)
                    if fr_opt == 0:
                        clk_divisor, n_packets = self.read_pmod()

                        print(f'Current PMOD settings:\nclock divisor: {clk_divisor}\nn packets: {n_packets}')

                        if clk_divisor != 0 and n_packets != 0:
                            
                            freq_estimated = self.estimate_flux_ramp_freq(n_packets)
                            V_estimated = self.estimate_flux_ramp_V(clk_divisor, freq_estimated)

                            print('Flux Ramp on Using Reset Signal')
                            print(f'Estimated V: {V_estimated} V\nEstimated freq: {freq_estimated} Hz')

                        elif clk_divisor == 0 and n_packets != 0:
                            
                            print('clock divider set to 0; flux ramp off')
                            """
                            freq_estimated = self.estimate_flux_ramp_freq(n_packets)
                            print(f'Flux ramp off; reset signal pulsing at {freq_estimated} Hz')
                            """
                            
                        elif clk_divisor != 0 and n_packets == 0:
                            
                            print('n packets set to 0; flux ramp off')
                            """
                            V_estimated = self.estimate_flux_ramp_V(clk_divisor, None) #will return the calibrated maximum voltate
                            freq_estimated = 256e6 / clk_divisor #ramp frequency set by the clock only (no reset)

                            print('Flux Ramp on without Reset Signal')
                            print(f'Estimated V: {V_estimated} V (max output)\nEstimated freq: {freq_estimated} Hz (set by clk)')
                            """

                        elif clk_divisor == 0 and n_packets == 0:
                            print('clock divider and n packets set to 0; flux ramp off')                         
                        
                        
                    elif fr_opt == 1:
                        self.set_flux_ramp_basic()
                    elif fr_opt == 2:
                        self.set_flux_ramp_advanced()
                    elif fr_opt == 3:
                        print('Turning flux ramp off.')

                        self.configure_pmod(1000000000,1)
                        clk_divisor, n_packets = self.read_pmod()

                        print(f'Clock divisor: {clk_divisor}\nN packets: {n_packets}')

                    elif fr_opt == 4:
                        break
                    
	    
            if opt == 16:  #exit
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


"""

if opt == 9: #find and save calibration (η)
                try:
                    delta_n = int(input('delta_n = '))  
                except ValueError:
                    print("Error, not a valid Number")
                except KeyboardInterrupt:
                    return
                
                try:
                    option = int(input('[0] Use most recent frequency list, [1] Input frequency list filename: '))  
                except ValueError:
                    print("Error, not a valid Number")
                except KeyboardInterrupt:
                    return
                
                if option == 0:
                    list_of_files = glob.glob('./frequency_lists/*.npy')
                    latest_file = max(list_of_files, key=os.path.getctime)
                    print(f'Loading: {latest_file}')
                    

                    f0 = cal.load_array(latest_file)

                    print(f0)                


                elif option == 1:
                    filename = input('File Name: ')
                    f0 = cal.load_array("./frequency_lists/"+filename)
                    print(f0)


                try:
                    option = int(input('[0] Use most recent LO sweep, [1] Input LO sweep filename: '))  
                except ValueError:
                    print("Error, not a valid Number")
                except KeyboardInterrupt:
                    return
                
                if option == 0:
                    list_of_files = glob.glob('./lo_sweeps/*.npy')
                    latest_file = max(list_of_files, key=os.path.getctime)
                    print(f'Loading: {latest_file}')

                    latest_file_split = latest_file.split("_")
                    ctime = latest_file_split[-1].split(".")

                    cal.find_calibration(latest_file, f0.real, delta_n, filename=f'./calibration/eta_fcenter_{latest_file_split[-2]}_{ctime[-2]}.txt')                               


                elif option == 1:
                    filename = input('File Name: ')
                    filename_split = filename.split("_")
                    ctime = filename_split[-1].split(".")

                    cal.find_calibration(filename, f0.real, delta_n, filename=f'./calibration/eta_fcenter_{filename_split[-2]}_{ctime[-2]}.txt')
"""
