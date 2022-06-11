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
import valon_synth9
from socket import *
import rfsocInterface
#from gbeConfig import roachDownlink
import time
import matplotlib.pyplot as plt
from scipy import signal
import find_kids_interactive as fk
#import pygetdata as gd
import targplot
plt.ion()

################################################################
# Run in IPYTHON as: %run kidPy

# for plotting interface, run as: %run kidPy plot
################################################################


# Load general setting
# Load general settings
gc = np.loadtxt("./general_config", dtype = "str")
firmware = gc[np.where(gc == 'FIRMWARE_FILE')[0][0]][1]
vna_savepath = gc[np.where(gc == 'VNA_SAVEPATH')[0][0]][1]
targ_savepath = gc[np.where(gc == 'TARG_SAVEPATH')[0][0]][1]
dirfile_savepath = gc[np.where(gc == 'DIRFILE_SAVEPATH')[0][0]][1]

# Valon Synthesizer params
CLOCK = int(gc[np.where(gc == 'clock')[0][0]][1])
LO = int(gc[np.where(gc == 'lo')[0][0]][1])
ext_ref = int(gc[np.where(gc == 'ext_ref')[0][0]][1])
lo_step = np.float(gc[np.where(gc == 'lo_step')[0][0]][1])
center_freq = np.float(gc[np.where(gc == 'center_freq')[0][0]][1])

# Optional test frequencies
test_freq = np.float(gc[np.where(gc == 'test_freq')[0][0]][1])
test_freq = np.array([test_freq])
freq_list = gc[np.where(gc == 'freq_list')[0][0]][1]

# Parameters for resonator search
smoothing_scale = np.float(gc[np.where(gc == 'smoothing_scale')[0][0]][1])
peak_threshold = np.float(gc[np.where(gc == 'peak_threshold')[0][0]][1])
spacing_threshold  = np.float(gc[np.where(gc == 'spacing_threshold')[0][0]][1])


'''
def systemInit():
    if not fpga:
        print("\nROACH link is down")
        return
    # Valon object
    valon = getValon()
    # Roach PPC object
    fpga = getFPGA()
    # Roach interface 
    ri = roachInterface(fpga, gc, regs, valon)
    if (ri.uploadfpg() < 0):
        print("\nFirmware upload failed")
    time.sleep(0.3)
    # UDP socket
    s = socket(AF_PACKET, SOCK_RAW, htons(3))
    with open('/proc/sys/net/core/rmem_max', 'r') as f:
        buf_max = int(f.readline())
    s.setsockopt(sock.SOL_SOCKET, sock.SO_RCVBUF, buf_max)
    # UDP object
    udp = roachDownlink(ri, fpga, gc, regs, s, ri.accum_freq)
    try:
        initValon(valon)
        print("Valon initiliazed")
    except OSError:
        print('\033[93mValon Synthesizer could not be initialized: Check comm port and power supply\033[93m')
        return
    except IndexError:
        print('\033[93mValon Synthesizer could not be initialized: Check comm port and power supply\033[93m')
        return
    fpga.write_int(regs[np.where(regs == 'accum_len_reg')[0][0]][1], ri.accum_len - 1)
    time.sleep(0.1)
    fpga.write_int(regs[np.where(regs == 'dds_shift_reg')[0][0]][1], int(gc[np.where(gc == 'dds_shift')[0][0]][1]))
    time.sleep(0.1)
    #ri.lpf(ri.boxcar)
    if (ri.qdrCal() < 0):
        print('\033[93mQDR calibration failed... Check FPGA clock source\033[93m')
        return
    else:
        fpga.write_int(regs[np.where(regs == 'write_qdr_status_reg')[0][0]][1], 1)
    time.sleep(0.1)
    try:
        udp.configDownlink()
    except AttributeError:
        print("UDP Downlink could not be configured. Check ROACH connection.")
        return
    return

'''
def initValon(valon, ref_freq = 10):
    """Configures default parameters for a Valon 5009 Sythesizer
        inputs:
            valon synth object valon: See getValon()
            bool ext_ref: Use external ref?
            int ref_freq: Ext reference freq, MHz"""
    if ext_ref:
        valon.set_reference(ref_freq)
        valon.set_ref_select(1)
    else:
        valon.set_ref_select(0)
    valon.set_refdoubler(CLOCK, 0)
    valon.set_refdoubler(LO, 0)
    valon.set_pfd(CLOCK, 40.)
    valon.set_pfd(LO, 10.)
    valon.set_frequency(LO, center_freq) # LO
    valon.set_frequency(CLOCK, 512.) # Clock
    valon.set_rf_level(CLOCK, 7)
    valon.set_rf_level(LO, 10)
    return

def getValon():
    """Return a valon synthesizer object
       If there's a problem, return None"""
    try:
        valon = valon_synth9.Synthesizer(gc[np.where(gc == 'valon_comm_port')[0][0]][1])
        return valon
    except OSError:
        "Valon could not be initialized. Check comm port and power supply."
    return None

def setValonLevel(valon, chan, dBm):
    """Set the RF power level of a Valon channel
       inputs:
           valon synth object valon: See getValon()
           int chan: LO or CLOCK (see above)
           float dBm: The desired power level in dBm (***calibrate
                      with spectrum analyzer)"""
    valon.set_rf_level(chan, dBm)
    return
'''
def setAtten(inAtten, outAtten):
    """Set the input and output attenuation levels for a RUDAT MCL-30-6000
        inputs:
            float outAtten: The output attenuation in dB
            float inAtten: The input attenuation in dB"""
    command = "sudo ./set_rudats " + str(inAtten) + ' ' + str(outAtten)
    os.system(command)
    return

def readAtten():
    """Read the attenuation levels for both channels of a RUDAT MCL-30-6000
       outputs:
            float outAtten
            float inAtten"""
    os.system("sudo ./read_rudats > rudat.log")
    attens = np.loadtxt('./rudat.log', delimiter = ",")
    inAtten = attens[0][1]
    outAtten = attens[1][1]
    return inAtten, outAtten
'''

#######################################################################
# Captions and menu options for terminal interface
caption1 = '\n\t\033[95mKID-PY2 RFSoC Readout\033[95m'
captions = [caption1]

main_opts= ['Upload firmware',
            'Initialize system & UDP conn',
            'Write test comb (single or multitone)',
            'Write stored comb',
            'Get system state',
            'VNA sweep and plot','Locate freqs from VNA sweep',
            'Write found freqs',
            'Target sweep and plot',
            'Execute a script',
            'Exit']
#########################################################################

def valonSweep(valon_inst, step=1e-3, centerf=1000, span=.5):
    """
    vnaSweep sets the valon5009 to start frequency, and steps
    step size until the stopf frequency is reached.

    params
        valon_inst: Synthesizer
            Initialized Valon 5009 device
        step: float
            step size (in MHz) for the valon. Defaults to 1 KHz
        centerf: float
            Center Frequency for the valon lo (in MHz). Defaults to 1GHz
        span: float
            span around center frequency(in MHz) to sweep (+/-). Defaults to 
            500KHz0
    """
    valon_inst.set_frequency(LO, centerf)
    start = centerf - (span/2)
    stop = centerf + (span/2)
    sweep_freqs = np.arange(start, stop, step)
    sweep_freqs = np.round(np.round(sweep_freqs/step)*step,3)
    print(sweep_freqs)
    for i in range(len(sweep_freqs)):
        print('lo freq =', sweep_freqs[i])
        valon_inst.set_frequency(LO, sweep_freqs[i])
        time.sleep(.5)        
    valon_inst.set_frequency(LO, centerf) # lo
    return
'''
def writevnacomb(cw = false):                                                             
    # roach ppc object                                                          
    fpga = getfpga()                                                            
    if not fpga:                                                                
        print("\nroach link is down")                                            
        return                                                                  
    # roach interface                                                           
    ri = roachinterface(fpga, gc, regs, none)                                   
    try:                                                                        
        if cw:
            ri.freq_comb = test_freq    
        else:
            ri.makefreqcomb()
        if (len(ri.freq_comb) > 400):                                             
            fpga.write_int(regs[np.where(regs == 'fft_shift_reg')[0][0]][1], 2**5 -1)
            time.sleep(0.1)                                                     
        else:                                                                   
            fpga.write_int(regs[np.where(regs == 'fft_shift_reg')[0][0]][1], 2**9 -1)
            time.sleep(0.1)                                                     
        ri.upconvert = np.sort(((ri.freq_comb + (center_freq)*1.0e6))/1.0e6)    
        print("rf tones =", ri.upconvert)                                        
        ri.writeqdr(ri.freq_comb, transfunc = false)                            
        np.save("last_freq_comb.npy", ri.freq_comb)                             
        if not (fpga.read_int(regs[np.where(regs == 'dds_shift_reg')[0][0]][1])):
            if regs[np.where(regs == 'ddc_mixerout_bram_reg')[0][0]][1] in fpga.listdev():
                shift = ri.return_shift(0)                                      
                if (shift < 0):                                                 
                    print("\nerror finding dds shift: try writing full frequency comb (n = 1000), or single test frequency. then try again")
                    return                                                       
                else:                                                           
                    fpga.write_int(regs[np.where(regs == 'dds_shift_reg')[0][0]][1], shift)
                    print("wrote dds shift (" + str(shift) + ")")                
            else:                                                               
                fpga.write_int(regs[np.where(regs == 'dds_shift_reg')[0][0]][1], ri.dds_shift)
    except keyboardinterrupt:                                                   
        return 
    return
'''

def vnasweepconsole():
    """does a wideband sweep of the rf band, saves data in vna_savepath
       as .npy files"""
    # # udp socket
    # s = socket(af_packet, sock_raw, htons(3))
    # valon object
    valon = getvalon()
    # roach ppc object
    #fpga = getfpga()
    # roach interface 
    #ri = roachinterface(fpga, gc, regs, valon)
    # udp object
    udp = roachdownlink(ri, fpga, gc, regs, s, ri.accum_freq)
    udp.configsocket()
    navg = np.int(gc[np.where(gc == 'navg')[0][0]][1])
    if not os.path.exists(vna_savepath):
        os.makedirs(vna_savepath)
    sweep_dir = vna_savepath + '/' + \
       str(int(time.time())) + '-' + time.strftime('%b-%d-%y-%h-%m-%s') + '.dir'
    os.mkdir(sweep_dir)
    np.save("./last_vna_dir.npy", sweep_dir)
    print(sweep_dir)
    valon.set_frequency(lo, center_freq)
    span = ri.pos_delta
    print("sweep span =", 2*np.round(ri.pos_delta,2), "hz")
    start = center_freq*1.0e6 - (span)
    stop = center_freq*1.0e6 + (span)
    sweep_freqs = np.arange(start, stop, lo_step)
    sweep_freqs = np.round(sweep_freqs/lo_step)*lo_step
    if not np.size(ri.freq_comb):
        ri.makefreqcomb()
    np.save(sweep_dir + '/bb_freqs.npy', ri.freq_comb)
    np.save(sweep_dir + '/sweep_freqs.npy', sweep_freqs)
    nchan = len(ri.freq_comb)
    if not nchan:
        nchan = fpga.read_int(regs[np.where(regs == 'read_comb_len_reg')[0][0]][1])
    idx = 0
    while (idx < len(sweep_freqs)):
        print('lo freq =', sweep_freqs[idx]/1.0e6)
        valon.set_frequency(lo, sweep_freqs[idx]/1.0e6)
        time.sleep(0.2)
        #time.sleep(0.1)
        if (udp.savesweepdata(navg, sweep_dir, sweep_freqs[idx], nchan,skip_packets = 25) < 0):
            continue
        else:
            idx += 1
        #time.sleep(0.1)
    valon.set_frequency(lo, center_freq) # lo
    return
'''


def targetsweep(valon,**keywords):
    """does a sweep centered on the resonances, saves data in targ_savepath
       as .npy files
       inputs:
           roachinterface object ri
           roach udp object udp
           valon synth object valon
           bool write: write test comb before sweeping?
           float span: sweep span, hz
           navg = number of data points to average at each sweep step
       keywords are:    
            span --specifies custom span rather than from general config
            lo_step --specifies custom lo step rather than from general config"""
    if ('span' in keywords):
        span = keywords['span']
    else:
        span = np.float(gc[np.where(gc == 'targ_span')[0][0]][1])
    if ('lo_step' in keywords):
        lo_step_targ = keywords['lo_step']
    else:
        lo_step_targ = lo_step
    navg = np.int(gc[np.where(gc == 'navg')[0][0]][1])
    vna_savepath = str(np.load("last_vna_dir.npy"))
    if not os.path.exists(targ_savepath):
        os.makedirs(targ_savepath)
    sweep_dir = targ_savepath + '/' + \
       str(int(time.time())) + '-' + time.strftime('%b-%d-%y-%h-%m-%s') + '.dir'
    os.mkdir(sweep_dir)
    np.save("./last_targ_dir.npy", sweep_dir)
    print(sweep_dir)
    target_freqs = np.load(vna_savepath + '/bb_targ_freqs.npy')
    #target_freqs = np.load("last_freq_comb.npy")
    np.save(sweep_dir + '/bb_target_freqs.npy', target_freqs)
    start = center_freq*1.0e6 - (span/2.)
    stop = center_freq*1.0e6 + (span/2.) 
    sweep_freqs = np.arange(start, stop, lo_step_targ)
    sweep_freqs = np.round(sweep_freqs/lo_step_targ)*lo_step_targ
    np.save(sweep_dir + '/bb_freqs.npy', target_freqs)
    np.save(sweep_dir + '/sweep_freqs.npy',sweep_freqs)
    first = true
    for freq in sweep_freqs:
        print('lo freq =', freq/1.0e6, ' mhz')
        valon.set_frequency(lo, freq/1.0e6)
        #time.sleep(0.1)
        udp.savesweepdata(navg, sweep_dir, freq, len(target_freqs),skip_packets = 25)
        #time.sleep(0.1)
    valon.set_frequency(LO, center_freq)
    return
'''


def openStoredSweep(savepath):
    """Opens sweep data
       inputs:
           char savepath: The absolute path where sweep data is saved
       ouputs:
           numpy array Is: The I values
           numpy array Qs: The Q values"""
    files = sorted(os.listdir(savepath))
    I_list, Q_list = [], []
    for filename in files:
        if filename.startswith('I'):
            I_list.append(os.path.join(savepath, filename))
        if filename.startswith('Q'):
            Q_list.append(os.path.join(savepath, filename))
    Is = np.array([np.load(filename) for filename in I_list])
    Qs = np.array([np.load(filename) for filename in Q_list])
    return Is, Qs

def plotVNASweep(path):
    plt.figure()
    print('Loading Data.\n')
    Is, Qs = openStoredSweep(path)
    sweep_freqs = np.load(path + '/sweep_freqs.npy')
    bb_freqs = np.load(path + '/bb_freqs.npy')
    rf_freqs = np.zeros((len(bb_freqs),len(sweep_freqs)))
    print('Reshaping Data.\n')
    for chan in range(len(bb_freqs)):
        rf_freqs[chan] = (sweep_freqs + bb_freqs[chan])/1.0e6
    Q = np.reshape(np.transpose(Qs),(len(Qs[0])*len(sweep_freqs)))
    I = np.reshape(np.transpose(Is),(len(Is[0])*len(sweep_freqs)))
    mag = np.sqrt(I**2 + Q**2)
    mag = 20*np.log10(mag/np.max(mag))
    mag = np.concatenate((mag[int(len(mag)/2):],mag[:int(len(mag)/2)]))
    rf_freqs = np.hstack(rf_freqs)
    rf_freqs = np.concatenate((rf_freqs[int(len(rf_freqs)/2):],rf_freqs[:int(len(rf_freqs)/2)]))
    print('Plotting Data.\n')
    plt.plot(rf_freqs, mag)
    plt.title(path, size = 16)
    plt.xlabel('frequency (MHz)', size = 16)
    plt.ylabel('dB', size = 16)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(path,'vna_sweep.png'), dpi = 100, bbox_inches = 'tight')
    plt.close()
    return

def plotTargSweep(path,interactive = True):
    """Plots the results of a TARG sweep
       inputs:
           path: Absolute path to where sweep data is saved"""
   
    Is, Qs = openStoredSweep(path)
    sweep_freqs = np.load(path + '/sweep_freqs.npy')
    bb_freqs = np.load(path + '/bb_freqs.npy')
    channels = len(bb_freqs)
    mags = np.zeros((channels, len(sweep_freqs))) 
    chan_freqs = np.zeros((channels, len(sweep_freqs)))
    new_targs = np.zeros((channels))
    for chan in range(channels):
        mags[chan] = np.sqrt(Is[:,chan]**2 + Qs[:,chan]**2)
        mags[chan] = 20*np.log10(mags[chan]/np.max(mags[chan]))
        chan_freqs[chan] = (sweep_freqs + bb_freqs[chan])/1.0e6
    mags = np.concatenate((mags[len(mags)/2:],mags[:len(mags)/2]))
    #bb_freqs = np.concatenate(bb_freqs[len(b_freqs)/2:],bb_freqs[:len(bb_freqs)/2]))
    #chan_freqs = np.concatenate((chan_freqs[len(chan_freqs)/2:],chan_freqs[:len(chan_freqs)/2]))
    #new_targs = [chan_freqs[chan][np.argmin(mags[chan])] for chan in range(channels)]
    if interactive:
        ip = targplot.interactive_plot(Is,Qs,chan_freqs)
    else:
        plt.figure()
        for chan in range(channels):
            plt.plot(chan_freqs[chan],mags[chan])
        plt.title(path, size = 16)
        plt.xlabel('frequency (MHz)', size = 16)
        plt.ylabel('dB', size = 16)
        plt.tight_layout()
        plt.savefig(os.path.join(path,'targ_sweep.png'), dpi = 100, bbox_inches = 'tight')
    return

def plotLastVNASweep():
    plotVNASweep(str(np.load('last_vna_dir.npy')))
    return

def plotLastTargSweep():
    plotTargSweep(str(np.load('last_targ_dir.npy')))
    return


'''
def plotPhasePSD(chan, udp, ri, time_interval):
    """Plots a channel phase noise power spectral density using Welch's method
       inputs:
           int chan: Detector channel
           gbeConfig object udp
           roachInterface object ri
           float time_interval: The integration time interval, seconds"""
    plt.ion()
    I, Q, phases = udp.streamChanPhase(chan, time_interval)
    f, Sii = signal.welch(I, ri.accum_freq, nperseg=len(I)/4)
    f, Sqq = signal.welch(Q, ri.accum_freq, nperseg=len(Q)/4)
    f, Spp = signal.welch(phases, ri.accum_freq, nperseg=len(phases)/4)
    Spp = 10*np.log10(Spp[1:]) 
    Sii = 10*np.log10(Sii[1:]) 
    Sqq = 10*np.log10(Sqq[1:]) 
    #plt.figure(figsize = (10.24, 7.68))
    #plt.title(r' $S_{\phi \phi}$', size = 18)
    plt.suptitle('Chan ' + str(chan))
    plt.subplot(3,1,1)
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_ylabel('dBc/Hz', size = 16)
    #ax.set_xlabel('log Hz', size = 16)
    ax.set_ylim((np.min(Sii) - 10, np.max(Sii) + 10))
    ax.plot(f[1:], Sii, linewidth = 1, label = 'I', alpha = 0.7)
    plt.legend(loc = 'upper right')
    plt.grid()
    plt.subplot(3,1,2)
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_ylabel('dBc/Hz', size = 16)
    #ax.set_xlabel('log Hz', size = 16)
    ax.set_ylim((np.min(Sqq) - 10, np.max(Sqq) + 10))
    ax.plot(f[1:], Sqq, linewidth = 1, label = 'Q', alpha = 0.7)
    plt.legend(loc = 'upper right')
    plt.grid()
    plt.subplot(3,1,3)
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_ylabel('dBc/Hz', size = 16)
    ax.set_xlabel('log Hz', size = 16)
    ax.set_ylim((np.min(Spp) - 10, np.max(Spp) + 10))
    ax.plot(f[1:], Spp, linewidth = 1, label = 'Phase', alpha = 0.7)
    plt.legend(loc = 'upper right')
    plt.grid()
    return
'''

def plotAllPSD(dirfile): 
    if dirfile == None:
        dirfile = str(np.load('last_data_path.npy'))
    firstframe = 0
    firstsample = 0
    d = gd.dirfile(dirfile, gd.RDWR|gd.UNENCODED)
    print("Number of frames in dirfile =", d.nframes)
    nframes = d.nframes
    vectors = d.field_list()
    ifiles = [i for i in vectors if i[0] == "I"]
    qfiles = [q for q in vectors if q[0] == "Q"]
    ifiles.remove("INDEX")
    wn = []
    plt.figure()
    plt.title(r' $S_{\phi \phi}$', size = 16)
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_ylabel('dBc/Hz', size = 16)
    ax.set_xlabel('log Hz', size = 16)
    for n in range(len(ifiles)):
        ivals = d.getdata(ifiles[n], gd.FLOAT32, first_frame = firstframe, first_sample = firstsample, num_frames = nframes)
        qvals = d.getdata(qfiles[n], gd.FLOAT32, first_frame = firstframe, first_sample = firstsample, num_frames = nframes)
        ivals = ivals[~np.isnan(ivals)]
        Qvals = qvals[~np.isnan(qvals)]
        f, Spp = signal.welch(np.arctan2(qvals,ivals), 488.28125)
        Spp = Spp[Spp != 0.]
        if not np.size(Spp):
            mean_wn = np.nan
            pass
        else:
            Spp = 10*np.log10(Spp) 
        mean_wn = np.mean(Spp[3*len(Spp)/4:])
        ax.plot(f, Spp, linewidth = 1)
        wn.append(mean_wn)
    plt.grid()
    plt.tight_layout()
    d.close()
    wn = np.array(wn)
    plt.figure()
    plt.plot(wn)
    plt.scatter(list(range(len(wn))), wn)
    plt.xlabel('Chan', size = 18)
    plt.ylabel('dBc/Hz', size = 18)
    plt.grid()
    plt.tight_layout()
    return

def filter_trace(path, bb_freqs, sweep_freqs):
    """Loads RF frequencies and magnitudes from TARG sweep data
       inputs:
           char path: Absolute path to sweep data
           bb_freqs: Array of baseband frequencies used during sweep
           sweep_freqs: Array of LO frequencies used during sweep
       outputs:
           array chan_freqs: Array of RF frequencies covered by each channel
           array mags: Magnitudes, in dB, of each channel sweep"""
    chan_I, chan_Q = openStoredSweep(path)
    channels = np.arange(np.shape(chan_I)[1])
    mag = np.zeros((len(bb_freqs),len(sweep_freqs)))
    chan_freqs = np.zeros((len(bb_freqs),len(sweep_freqs)))
    for chan in channels:
        mag[chan] = (np.sqrt(chan_I[:,chan]**2 + chan_Q[:,chan]**2))
        chan_freqs[chan] = (sweep_freqs + bb_freqs[chan])/1.0e6
    N= len(chan_freqs)
    mag= np.concatenate((mag[int(N/2):],mag[:int(N/2)]))
    mags = 20*np.log10(mag/np.max(mag))
    mags = np.hstack(mags)
    chan_freqs= np.concatenate((chan_freqs[int(N/2):],chan_freqs[:int(N/2)]))
    chan_freqs = np.hstack(chan_freqs)
    return chan_freqs, mags

def findFreqs(path, plot = False):
    """Open target sweep data stored at path and identify resonant frequencies
       inputs:
           char path: Absolute path to sweep data
           bool plot: Option to plot results"""
    bb_freqs = np.load(path + '/bb_freqs.npy')
    sweep_freqs = np.load(path + '/sweep_freqs.npy')
    chan_freqs, mags = filter_trace(path, bb_freqs, sweep_freqs)
    chan_freqs *= 1.0e6
    filtermags = lowpass_cosine(mags, lo_step, 1./smoothing_scale, 0.1 * (1.0/smoothing_scale))
    ilo = np.where((mags-filtermags) < -1.0*peak_threshold)[0]
    iup = np.where( (mags-filtermags) > -1.0*peak_threshold)[0]
    new_mags = mags - filtermags
    new_mags[iup] = 0
    labeled_image, num_objects = ndimage.label(new_mags)
    indices = ndimage.measurements.minimum_position(new_mags,labeled_image,np.arange(num_objects)+1)
    kid_idx = np.array(indices, dtype = 'int')
    del_idx = []
    for i in range(len(kid_idx) - 1):
        spacing = (chan_freqs[kid_idx[i + 1]] - chan_freqs[kid_idx[i]])
        if (spacing < spacing_threshold):
            if (new_mags[kid_idx[i + 1]] < new_mags[kid_idx[i]]):
                del_idx.append(i)
            else:
                del_idx.append(i + 1)

    del_idx = np.array(del_idx)
    kid_idx = np.delete(kid_idx, del_idx)

    del_again = []
    for i in range(len(kid_idx) - 1):
        spacing = (chan_freqs[kid_idx[i + 1]] - chan_freqs[kid_idx[i]])
        if (spacing < spacing_threshold):
            if (new_mags[kid_idx[i + 1]] < new_mags[kid_idx[i]]):
                del_again.append(i)
            else:
                del_again.append(i + 1)

    del_again = np.array(del_again)
    kid_idx = np.delete(kid_idx, del_again)
    # list of kid frequencies
    rf_target_freqs = np.array(chan_freqs[kid_idx])
    bb_target_freqs = ((rf_target_freqs*1.0e6) - center_freq)

    if len(bb_target_freqs) > 0:
        bb_target_freqs = np.roll(bb_target_freqs, - np.argmin(np.abs(bb_target_freqs)) - 1)
        np.save(path + '/bb_targ_freqs.npy', bb_target_freqs)
        print(len(rf_target_freqs), "KIDs found:\n")
        print(rf_target_freqs)
    else:
        print("No freqs found...")

    if plot:
        plt.figure(1)
        plt.plot(chan_freqs, mags,'b', label = 'no filter')
        plt.plot(chan_freqs, filtermags,'g', label = 'filtered')
        plt.xlabel('frequency (Hz)')
        plt.ylabel('dB')
        plt.legend()
        plt.savefig(path + 'MagsFiltermags.png')
        plt.figure(2)
        plt.plot(chan_freqs, mags - filtermags, 'b')
        plt.plot(chan_freqs[ilo],mags[ilo]-filtermags[ilo],'r*')
        plt.savefig(path + 'KidsMagsMinusFiltermags.png')
        plt.figure(4)
        plt.plot(chan_freqs, mags, 'b')
        plt.plot(chan_freqs[kid_idx], mags[kid_idx], 'r*')
        plt.xlabel('frequency (Hz)')
        plt.ylabel('dB')
        plt.savefig(path + 'KidsMags.png')
    return

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
