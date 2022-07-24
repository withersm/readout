# This software is a work in progress. It is a console interface designed 
# to operate the BLAST RFSOC firmware. 
#
# Copyright (C) 2022 <Authors listed below>
# Author: Cody Roberson, Ryan Stephenson Adrian Sinclair, Philip Mauskopf
#       (carobers@asu.edu)

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
import udpcap
import datetime

################################################################
# Config File Settings
################################################################
import configparser
config = configparser.ConfigParser()
config.read("generalConfig.conf")
__redis_host = config['REDIS']['host']
__customWaveform = config['DSP']['customWaveform']
__customSingleTone = config['DSP']['singleToneFrequency']
__saveData = config['DATA']['saveFolder']

#######################################################################
# Captions and menu options for terminal interface
caption1 = '\n\t\033[95mKID-PY2 RFSoC Readout\033[95m'
captions = [caption1]

main_opts= ['Upload firmware',
            'Initialize system & UDP conn',
            'Write test comb (single or multitone)',
            'Write stored comb from config file',
            'I <-> Q Phase offset',
            'Take Raw Data',
            #'',
            #'Write found freqs',
            #'Target sweep and plot',
            #'Execute a script',
            'Exit']
#########################################################################
def testConnection(r):
    try:
        tr = r.set('testkey', '123')
        return True
    except redis.exceptions.ConnectionError as e:
        print(e)
        return False
        

def waitForFree(r, delay = 0.25, timeout = 10):
    count = 0
    r.set("status", "busy")
    while(r.get("status") != b"free"):
        time.sleep(delay)
        count = count + 1
        if (count == timeout):
            print("TIMED OUT WHILE WAITING FOR RFSOC")
            return False
    return True

def checkBlastCli(r, p):
    r.publish("ping", "hello?")
    count = 1 
    delay = 0.5 
    timeout = 6
    m = None
    while(1):       
        m = p.get_message()
        if (m is not None and m['data'] == b"Hello World"):
            print("redisControl is running")
            return True
        if (count >= timeout):
            print("RFSOC didn't reply, is it running redisControl?")
            return False

        time.sleep(delay)
        count = count + 1
       

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
    opt = None
    try:
        opt = eval(input())
    except KeyboardInterrupt:
        exit()
    return opt

def main_opt(r, p, udp):
    while 1:
        #TODO: implement a check here to ensure we are connected to the redis server and in turn
        # the RFSOC is connected to the redis server as well

        #TODO: implement a response routine
        conStatus = testConnection(r)
        if conStatus:
            print('\033[0;36m' + "\r\nConnected" + '\033[0m')
        else:
            print('\033[0;31m' + "\r\nCouldn't connect to redis-server double check it's running and the generalConfig is correct" + '\033[0m')
        opt = menu(captions,main_opts)
        if conStatus == False:
            resp = input("Can't connect to redis server, do you want to continue anyway? [y/n]: ")
            if resp != "y":
                exit()
        if opt == 0: # upload firmware
            os.system("clear")
            cmd = {"cmd" : "ulBitstream", "args":[]}
            cmdstr = json.dumps(cmd)
            r.publish("picard", cmdstr)
            r.set("status", "busy")
            print("Waiting for the RFSOC to upload it's bitstream...")
            if waitForFree(r, 0.75, 25):
                print("Done")


        if opt == 1: # Init System & UDP conn.
            os.system("clear")
            print("Initializing System and UDP Connection")
            cmd = {"cmd" : "initRegs", "args":[]}
            cmdstr = json.dumps(cmd)
            r.publish("picard", cmdstr)
            if waitForFree(r, 0.5, 5):
                print("Done")

       
        if opt == 2: # Write test comb
            prompt = input('Full test comb? y/n ')
            os.system("clear")
            if prompt == 'y':
                print("Waiting for the RFSOC to finish writing the full comb")
                cmd = {"cmd" : "ulWaveform", "args":[]}
                cmdstr = json.dumps(cmd)
                r.publish("picard", cmdstr)

            else:
                print("Waiting for the RFSOC to write single {} MHz Tone".format(float(__customSingleTone)/1e6))
                cmd = {"cmd" : "ulWaveform", "args":[[float(__customSingleTone)]]}
                cmdstr = json.dumps(cmd)
                r.publish("picard", cmdstr)           
            if waitForFree(r, 0.75, 25):
                print("Done")

       
        if opt == 3: # write stored comb
            os.system("clear")
            print("Waiting for the RFSOC to finish writing the custom frequency list: \r\n{}".format(__customWaveform))
            fList = []

            # seperate values from config and remove ',' before converting to number and
            # sending the list of values upto the DAC
            for value in __customWaveform.split():
                s = value.replace(',', '')
                fList.append(float(s))

            cmd = {"cmd" : "ulWaveform", "args":[fList]}
            cmdstr = json.dumps(cmd)
            r.publish("picard", cmdstr)
            if waitForFree(r, 0.75, 25):
                print("Done")

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
                os.system('clear')
                print("Binding Socket")
                udp.bindSocket()
                print("Capturing packets") 
                fname = __saveData + "kidpyCaptureData{0:%Y%m%d%H%M%S}.h5".format(datetime.datetime.now())
                print(fname)
                if t < 60:
                    udp.shortDataCapture(fname, 488*t)
                else:
                    udp.LongDataCapture(fname, 488*t)
                print("Releasing Socket")
                udp.release()

                

        if opt == 6: # get system state
           exit()

        return 0

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


def main():
    r = redis.Redis(__redis_host)
    p = r.pubsub()
    p.subscribe("ping")
    time.sleep(1)
    if p.get_message()['data'] != 1:
        print("Failed to Subscribe to redis Ping Channel")
        exit()

    # check that the RFSOC is running redisControl.py 
    os.system('clear')
    if checkBlastCli(r, p) == False:
        exit()

    udp = udpcap.udpcap()
    while 1:
        try:
            upload_status = 0
            #if fpga:
            #    if fpga.is_running():
            #        #firmware_info = fpga.get_config_file_info()
            #        upload_status = 1
            #time.sleep(0.1)
            #upload_status = main_opt(fpga, ri, udp, valon, upload_status)
            upload_status = main_opt(r, p, udp)
        except TypeError:
            pass
    return 

if __name__ == "__main__":
    main()
