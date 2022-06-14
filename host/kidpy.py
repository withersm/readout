# This software is a work in progress. It is a console interface designed 
# to operate the BLAST-TNG ROACH2 firmware. 
#
# Copyright (C) January, 2018  Gordon, Sam <sbgordo1@asu.edu>
# Author: Gordon, Sam <sbgordo1@asu.edu>
#
# Modified, May 2020: Stephenson, Ryan; Sinclair, Adrian; Roberson, Cody 
# (and Hacked by students of the ASU Astronomical Instruments lab for RFSoC)
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import numpy as np
import sys, os
import struct
from socket import *
import redis
#from gbeConfig import roachDownlink
import time
import matplotlib.pyplot as plt
from scipy import signal
################################################################
# Run in IPYTHON as: %run kidPy

# for plotting interface, run as: %run kidPy plot
################################################################

################################################################
# Config File Settings
################################################################
import configparser
config = configparser.ConfigParser()
config.read("generalConfig.conf")
__redis_host = config['REDIS']['host']


#######################################################################
# Captions and menu options for terminal interface
caption1 = '\n\t\033[95mKID-PY2 RFSoC Readout\033[95m'
captions = [caption1]

main_opts= ['Upload firmware',
            'Initialize system & UDP conn',
            'Write test comb (single or multitone)',
            #'Write stored comb',
            #'Get system state',
            #'VNA sweep and plot','Locate freqs from VNA sweep',
            #'Write found freqs',
            #'Target sweep and plot',
            #'Execute a script',
            'Exit']
#########################################################################

def menu(captions, options):
    """Creates menu for terminal interface
       inputs:
           list captions: List of menu captions
           list options: List of menu options
       outputs:
           int opt: Integer corresponding to menu option chosen by user"""
    print('\t' + captions[0] + '\n')
    for i in range(len(options)):
        print('\t' +  '\033[32m' + str(i) + ' ..... ' '\033[0m' +  options[i] + '\n')
    opt = eval(input())
    return opt

def main_opt(ri, fpga, udp, valon, upload_status):
    """Creates terminal interface
       inputs:
           casperfpga object fpga
           roachInterface object ri
           gbeConfig object udp
           valon synth object valon
           int upload_status: Integer indicating whether or not firmware is uploaded
        outputs:
          int  upload_status"""
    while 1:
        #if not fpga:
        #    print('\n\t\033[93mROACH link is down: Check PPC IP & Network Config\033[93m')
        #else:
        #    print('\n\t\033[92mROACH link is up\033[92m')
        #if not upload_status:
        #    print('\n\t\033[93mNo firmware onboard. If ROACH link is up, try upload option\033[93m')
        #else:
        #    print('\n\t\033[92mFirmware uploaded\033[92m')
        opt = menu(captions,main_opts)
        if opt == 0: # upload firmwarei
            # run init file
            if (ri.uploadOverlay() < 0):
                print("\nFirmware upload failed")
            else:
               upload_status = 1

        if opt == 1: # Init System & UDP conn.
            # run single script
            
            ri.initRegs()
            
            try:
                initValon(valon)
                print("Valon initiliazed")
            except (OSError, IndexError):
                print('\033[93mValon Synthesizer could not be initialized: Check comm port and power supply\033[93m')
                break
        
        if opt == 2: # Write test comb
            prompt = input('Full test comb? y/n ')
            if prompt == 'y':
                print("Writing VNA comb")
                LUT_I, LUT_Q, DDS_I, DDS_Q , freqs = ri.surfsUpDude(np.array([100e6]), vna=True)
                print("here")
                ri.load_bin_list(freqs)
                print("here2")
                ri.load_waveform_into_mem(LUT_I, LUT_Q, DDS_I, DDS_Q)
            else:
                print("Writing single 100MHz tone")
                LUT_I, LUT_Q, DDS_I, DDS_Q, freqs = ri.surfsUpDude(np.array([100e6]), vna=False)
                ri.load_bin_list(freqs)
                ri.load_waveform_into_mem(LUT_I, LUT_Q, DDS_I, DDS_Q)
        
        if opt == 3: # write stored comb
            if not fpga:
                print("\nROACH link is down")
                break
            try:
                freq_comb = np.load(freq_list)
                freq_comb = freq_comb[freq_comb != 0]
                freq_comb = np.roll(freq_comb, - np.argmin(np.abs(freq_comb)) - 1)
                ri.freq_comb = freq_comb
                ri.upconvert = np.sort(((ri.freq_comb + (ri.center_freq)*1.0e6))/1.0e6)
                print("RF tones =", ri.upconvert)
                if len(ri.freq_comb) > 400:
                    fpga.write_int(regs[np.where(regs == 'fft_shift_reg')[0][0]][1], 2**5 -1)
                    time.sleep(0.1)
                else:
                    fpga.write_int(regs[np.where(regs == 'fft_shift_reg')[0][0]][1], 2**9 -1)
                    time.sleep(0.1)
                ri.writeQDR(ri.freq_comb)
                #setAtten(27, 17)
                np.save("last_freq_comb.npy", ri.freq_comb)
            except KeyboardInterrupt:
                pass
        if opt == 4: # get system state
            if not fpga:
                print("\nROACH link is down")
                break
            if not np.size(ri.freq_comb):
                try:
                    ri.freq_comb = np.load("last_freq_comb.npy")
                except IOError:
                   print("\nFirst need to write a frequency comb with length > 1")
                   break
            try:
                ri.writeQDR(ri.freq_comb, transfunc = True)
                fpga.write_int(regs[np.where(regs == 'write_comb_len_reg')[0][0]][1], len(ri.freq_comb))
            except ValueError:
                print("\nClose Accumulator snap plot before calculating transfer function")
        if opt == 5: # VNA sweep and plot
            try:
                valonSweep(valon) 
                #plotVNASweep(str(np.load("last_vna_dir.npy")))
            except KeyboardInterrupt:
                print("AAAAHHHHHHHHHHHHHHHHHHHHHHHHHHH")

        if opt == 6: # Locate freqs from VNA Sweep
            try:
                path = str(np.load("last_vna_dir.npy"))
                print("Sweep path:", path)
                fk.main(path, center_freq, lo_step, smoothing_scale, peak_threshold, spacing_threshold)
                #findFreqs(str(np.load("last_vna_dir.npy")), plot = True)
            except KeyboardInterrupt:
                break
       
        if opt == 7: # Write found freqs 
            if not fpga:
                print("\nROACH link is down")
                break
            try:
                freq_comb = np.load(os.path.join(str(np.load('last_vna_dir.npy')), 'bb_targ_freqs.npy'))
                freq_comb = freq_comb[freq_comb != 0]
                freq_comb = np.roll(freq_comb, - np.argmin(np.abs(freq_comb)) - 1)
                ri.freq_comb = freq_comb
                print(ri.freq_comb)
                #ri.upconvert = np.sort(((ri.freq_comb + (center_freq)*1.0e6))/1.0e6)
                #print "RF tones =", ri.upconvert
                if len(ri.freq_comb) > 400:
                    fpga.write_int(regs[np.where(regs == 'fft_shift_reg')[0][0]][1], 2**5 -1)    
                    time.sleep(0.1)
                else:
                    fpga.write_int(regs[np.where(regs == 'fft_shift_reg')[0][0]][1], 2**9 -1)    
                    time.sleep(0.1)
                ri.writeQDR(ri.freq_comb)
                #setAtten(27, 17)
                np.save("last_freq_comb.npy", ri.freq_comb)
            except KeyboardInterrupt:
                pass
        if opt == 8: # Target Sweep and plot 
            if not fpga:
                print("\nROACH link is down")
                break
            try:
                targetSweep(ri, udp, valon)
                plotTargSweep(str(np.load("last_targ_dir.npy")))
            except KeyboardInterrupt:
                pass
        if opt == 9: # Execute Script
            if not fpga:
                print("\nROACH link is down")
                break
            try:
                prompt = input("what is the filename of the script to be executed: ")
                exec(compile(open("./scripts/"+prompt, "rb").read(), "./scripts/"+prompt, 'exec'))
            except KeyboardInterrupt:
                pass
        if opt == 10: # Exit
            sys.exit()
        return upload_status

############################################################################
# Interface for snap block plotting
plot_caption = '\n\t\033[95mKID-PY ROACH2 Snap Plots\033[95m'
plot_opts= ['I & Q ADC input',\
            'Firmware FFT',\
            'Digital Down Converter Time Domain',\
            'Downsampled Channel Magnitudes']
#############################################################################

def makePlotMenu(prompt,options):
    """Menu for plotting interface
       inputs:
           char prompt: a menu caption
           list options: List of menu options
       outputs:
           int opt: Integer corresponding to chosen option"""
    print('\t' + prompt + '\n')
    for i in range(len(options)):
        print('\t' +  '\033[32m' + str(i) + ' ..... ' '\033[0m' +  options[i] + '\n')
    print('\n' + "Run: ")
    opt = eval(input())
    return opt

def plot_opt(ri):
    """Creates terminal interface for plotting snap blocks
       inputs:
           roachInterface object ri"""
    while 1:
        opt = makePlotMenu(plot_caption, plot_opts)
        if opt == 0:
            try:
                ri.plotADC()
            except KeyboardInterrupt:
                #fig = plt.gcf()
                #plt.close(fig)
                pass
        if opt == 1:
            try:
                ri.plotFFT()
            except KeyboardInterrupt:
                fig = plt.gcf()
                plt.close(fig)
        if opt == 2:
            chan = eval(input('Channel = ? '))
            try:
                ri.plotMixer(chan, fir = False)
            except KeyboardInterrupt:
                fig = plt.gcf()
                plt.close(fig)
        if opt == 3:
            try:
                ri.plotAccum()
            except KeyboardInterrupt:
                fig = plt.gcf()
                plt.close(fig)
    return

def main():
    s = None
    # Valon synthesizer instance
    
    try:
        valon = valon_synth9.Synthesizer('/dev/ttyUSB0')
    except OSError:
        print("Valon could not be initialized. Check comm port and power supply.")
    
    ri = rfsocInterface.rfsocInterface() #fpga, gc )#, valon)
    
    os.system('clear')
    while 1:
        try:
            upload_status = 0
            #if fpga:
            #    if fpga.is_running():
            #        #firmware_info = fpga.get_config_file_info()
            #        upload_status = 1
            #time.sleep(0.1)
            #upload_status = main_opt(fpga, ri, udp, valon, upload_status)
            upload_status= main_opt(ri, None, None, valon, upload_status)
        except TypeError:
            pass
    return 

def plot_main():
    try:
        fpga = casperfpga.katcp_fpga.KatcpFpga(gc[np.where(gc == 'roach_ppc_ip')[0][0]][1], timeout = 3.)
    except RuntimeError:
        fpga = None
    # Roach interface
    ri = roachInterface(fpga, gc, regs, None)
    while 1:
        plot_opt(ri)
    return 

if __name__ == "__main__":
    if len(sys.argv) > 1:
        plot_main()
    else:
        main()
