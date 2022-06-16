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
import json
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
__customWaveform = config['DSP']['customWaveform']
__customSingleTone = config['DSP']['singleToneFrequency']
r = redis.Redis(__redis_host)



#######################################################################
# Captions and menu options for terminal interface
caption1 = '\n\t\033[95mKID-PY2 RFSoC Readout\033[95m'
captions = [caption1]

main_opts= ['Upload firmware',
            'Initialize system & UDP conn',
            'Write test comb (single or multitone)',
            'Write stored comb from config file',
            #'Get system state',
            #'VNA sweep and plot','Locate freqs from VNA sweep',
            #'Write found freqs',
            #'Target sweep and plot',
            #'Execute a script',
            'Exit']
#########################################################################
def testConnection(r):
    try:
        tr = r.set('testkey', '123')
        print('\033[0;36m' + "\r\nConnected" + '\033[0m')
        return True
    except redis.exceptions.ConnectionError as e:
        print('\033[0;31m' + "\r\nCouldn't connect to redis-server double check it's running and the generalConfig is correct" + '\033[0m')
        print(e)
        return False
        


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
        #TODO: implement a check here to ensure we are connected to the redis server and in turn
        # the RFSOC is connected to the redis server as well

        #TODO: implement a response routine
        os.system("clear") 
        conStatus = testConnection(r)
        opt = menu(captions,main_opts)
        if conStatus == False:
            resp = input("Can't connect to redis server, do you want to continue anyway? [y/n]: ")
            if resp != "y":
                exit()
        if opt == 0: # upload firmware
            cmd = {"cmd" : "ulBitstream", "args":[]}
            cmdstr = json.dumps(cmd)
            r.publish("picard", cmdstr)

        if opt == 1: # Init System & UDP conn.
            cmd = {"cmd" : "initRegs", "args":[]}
            cmdstr = json.dumps(cmd)
            r.publish("picard", cmdstr)
       
        if opt == 2: # Write test comb
            prompt = input('Full test comb? y/n ')
            if prompt == 'y':
                cmd = {"cmd" : "ulWaveform", "args":[]}
                cmdstr = json.dumps(cmd)
                r.publish("picard", cmdstr)

            else:
                print("Writing single {} Hz Tone".format(float(__customSingleTone)))
                cmd = {"cmd" : "ulWaveform", "args":[[float(__customSingleTone)]]}
                cmdstr = json.dumps(cmd)
                r.publish("picard", cmdstr)

       
        if opt == 3: # write stored comb
            fList = []

            # seperate values from config and remove ',' before converting to number and
            # sending the list of values upto the DAC
            for value in __customWaveform.split():
                s = value.replace(',', '')
                fList.append(float(s))

            cmd = {"cmd" : "ulWaveform", "args":[fList]}
            cmdstr = json.dumps(cmd)
            r.publish("picard", cmdstr)


        if opt == 4: # get system state
           exit()

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
            upload_status= main_opt(None, None, None, None, None)
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
