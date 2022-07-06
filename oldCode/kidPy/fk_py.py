import numpy as np
import sys, os
import scipy.ndimage
from despike import despike

# setting output_on to True will enable debugging statements and plots
# DO NOT ENABLE ON FLIGHT COMP
output_on = True
if output_on:
    import matplotlib.pyplot as plt
    plt.ion()
    plt.rcParams['axes.grid'] = True

"""
vna_path = sys.argv[1]
targ_path = sys.argv[2]
smoothing_scale = float(sys.argv[3])
peak_threshold = float(sys.argv[4])
spacing_threshold = float(sys.argv[5])
lo_centerfreq = float(sys.argv[6])
sweep_step = float(sys.argv[7])
"""

#########################################
# Channel window limits
#########################################
pos_win = 0.05 # MHz above f_res
neg_win = 0.05 # MHz below f_res

def open_stored(path, vna_freqs, sweep_freqs):
    files = sorted(os.listdir(path))[:-3]
    chan_I = np.zeros((len(sweep_freqs),len(vna_freqs)))
    chan_Q = np.zeros((len(sweep_freqs),len(vna_freqs)))
    for i in range(len(files)):
        chan_I[i], chan_Q[i] = np.loadtxt(os.path.join(path,files[i]), dtype = "float", usecols = (1,2), unpack = True)
    return chan_I, chan_Q

def open_stored_py(path):
    files = sorted(os.listdir(path))
    trim_file_list(files)
    I_list, Q_list = [], []
    for filename in files:
        if filename.startswith('I'):
            I_list.append(os.path.join(path, filename))
        if filename.startswith('Q'):
            Q_list.append(os.path.join(path, filename))
    Is = np.array([np.load(filename) for filename in I_list])
    Qs = np.array([np.load(filename) for filename in Q_list])
    return Is, Qs

def filter_trace(path, vna_freqs, sweep_freqs):
    chan_I, chan_Q = open_stored_py(path)
    channels = np.arange(np.shape(chan_I)[1])
    mag = np.zeros((len(vna_freqs),len(sweep_freqs)))
    chan_freqs = np.zeros((len(vna_freqs),len(sweep_freqs)))
    for chan in channels:
        mag[chan] = (np.sqrt(chan_I[:,chan]**2 + chan_Q[:,chan]**2))
        chan_freqs[chan] = (sweep_freqs + vna_freqs[chan])/1.0e6
    mag = np.concatenate((mag[len(mag)/2:], mag[:len(mag)/2]))
    mags = np.hstack(mag)
    mags = 20*np.log10(mags/np.max(mags))
    chan_freqs = np.hstack(chan_freqs)
    chan_freqs = np.concatenate((chan_freqs[len(chan_freqs)/2:],chan_freqs[:len(chan_freqs)/2]))
    return chan_freqs, mags

def lowpass_cosine( y, tau, f_3db, width, padd_data=True):
        # padd_data = True means we are going to symmetric copies of the data to the start and stop
    # to reduce/eliminate the discontinuities at the start and stop of a dataset due to filtering
    #
    # False means we're going to have transients at the start and stop of the data
    # kill the last data point if y has an odd length
    if np.mod(len(y),2):
        y = y[0:-1]
    # add the weird padd
    # so, make a backwards copy of the data, then the data, then another backwards copy of the data
    if padd_data:
        y = np.append( np.append(np.flipud(y),y) , np.flipud(y) )
    # take the FFT
    import scipy
    import scipy.fftpack
    ffty=scipy.fftpack.fft(y)
    ffty=scipy.fftpack.fftshift(ffty)
    # make the companion frequency array
    delta = 1.0/(len(y)*tau)
    nyquist = 1.0/(2.0*tau)
    freq = np.arange(-nyquist,nyquist,delta)
    # turn this into a positive frequency array
    pos_freq = freq[(len(ffty)/2):]
    # make the transfer function for the first half of the data
    i_f_3db = min( np.where(pos_freq >= f_3db)[0] )
    f_min = f_3db - (width/2.0)
    i_f_min = min( np.where(pos_freq >= f_min)[0] )
    f_max = f_3db + (width/2);
    i_f_max = min( np.where(pos_freq >= f_max)[0] )
    transfer_function = np.zeros(len(y)/2)
    transfer_function[0:i_f_min] = 1
    transfer_function[i_f_min:i_f_max] = (1 + np.sin(-np.pi * ((freq[i_f_min:i_f_max] - freq[i_f_3db])/width)))/2.0
    transfer_function[i_f_max:(len(freq)/2)] = 0
    # symmetrize this to be [0 0 0 ... .8 .9 1 1 1 1 1 1 1 1 .9 .8 ... 0 0 0] to match the FFT
    transfer_function = np.append(np.flipud(transfer_function),transfer_function)
    # apply the filter, undo the fft shift, and invert the fft
    filtered=np.real(scipy.fftpack.ifft(scipy.fftpack.ifftshift(ffty*transfer_function)))
    # remove the padd, if we applied it
    if padd_data:
        filtered = filtered[(len(y)/3):(2*(len(y)/3))]
    # return the filtered data
    return filtered

# lower and upper frequency cutoffs
low_freq_cut = 620.
high_freq_cut = 1040.
# mask lower and upper frequency cutoffs
mask_low = []
mask_high = []

def trim_file_list(files):
    if 'bb_targ_freqs.npy' in files:
        files.remove('bb_targ_freqs.npy')
    if 'sweep_freqs.npy' in files:
        files.remove('sweep_freqs.npy')
    if 'gradient_freqs.npy' in files:
        files.remove('gradient_freqs.npy')
    if 'first_targ_trf.npy' in files:
        files.remove('first_targ_trf.npy')
    if 'bb_freqs.npy' in files:
        files.remove('bb_freqs.npy')
    if 'vna_sweep.png' in files:
        files.remove('vna_sweep.png')
    return

def fitFreqs(vna_path, lo_centerfreq, sweep_step, smoothing_scale, peak_threshold, spacing_threshold):
    print vna_path, lo_centerfreq, sweep_step, smoothing_scale, peak_threshold, spacing_threshold
    vna_freqs = np.load(os.path.join(vna_path, "bb_freqs.npy"))
    sweep_freqs = np.load(os.path.join(vna_path, "sweep_freqs.npy"))
    rf_freqs = vna_freqs + lo_centerfreq
    chan_freqs, mags = filter_trace(vna_path, vna_freqs, sweep_freqs)
    mags = despike(mags)
    # first done in log space for frequency identification
    filtermags = lowpass_cosine(mags, sweep_step, 1./smoothing_scale, 0.1 * (1.0/smoothing_scale))
    
    if output_on:
        plt.ion()
        plt.figure(1)
        plt.clf()
        plt.plot(chan_freqs,mags,label='Raw')
        plt.plot(chan_freqs,filtermags,'g',label='Filtered')
        #plt.scatter(chan_freqs[np.where(sweep_freqs == lo_centerfreq)[0][0]::2*np.where(sweep_freqs == lo_centerfreq)[0][0]],\
        #           filtermags[np.where(sweep_freqs == lo_centerfreq)[0][0]::2*np.where(sweep_freqs == lo_centerfreq)[0][0]], color = 'red')
        plt.xlabel('frequency [MHz]', fontsize = 16)
        plt.ylabel('|S21| [dB]', fontsize = 16)
        plt.legend()
        plt.tight_layout()
    
        plt.figure(2)
        plt.clf()
        plt.plot(chan_freqs,mags-filtermags, label = 'TRF corrected')
        plt.xlabel('frequency [MHz]', fontsize = 16)
        plt.ylabel('|S21| dB', fontsize = 16)
        plt.legend()
        plt.tight_layout()
    
    iup = np.where( (mags-filtermags) > -1.0*peak_threshold)[0]
    new_mags = mags - filtermags
    new_mags[iup] = 0
    labeled_image, num_objects = scipy.ndimage.label(new_mags)
    indices = scipy.ndimage.measurements.minimum_position(new_mags,labeled_image,np.arange(num_objects)+1)
    kid_idx = indices
    
    #############################################################################
    # Enforce minimum frequency spacing condition
    #############################################################################
    del_idx = []
    for i in range(len(kid_idx) - 1):
        #if output_on:
        #    print kid_idx[i + 1], kid_idx[i]
        #    print "freqs:", chan_freqs[kid_idx[i + 1]], chan_freqs[kid_idx[i]]
        #    print "mags:", new_mags[kid_idx[i + 1]], new_mags[kid_idx[i]]
        spacing = (chan_freqs[kid_idx[i + 1]] - chan_freqs[kid_idx[i]]) * 1.0e3
        #if output_on:
        #    print "diff:", spacing
        if (spacing < spacing_threshold):
            #if output_on:
            #    print "Spacing collision"
            if (new_mags[kid_idx[i + 1]] < new_mags[kid_idx[i]]):
                if (kid_idx[i][0] in del_idx):
                    pass
                else:
                    del_idx.append(kid_idx[i][0])
                    #if output_on:
                    #    print "Removing", chan_freqs[kid_idx[i]]
            else:
                if (kid_idx[i + 1][0] in del_idx):
                    pass
                else:
                    del_idx.append(kid_idx[i + 1][0])
                    #if output_on:
                    #    print "Removing", chan_freqs[kid_idx[i + 1]]
    
    #if output_on:
    #    print np.min(kid_idx), np.max(kid_idx)
    #    print del_idx
    for idx in del_idx:
        kid_idx.remove(idx)
    
    #########################################################################
    # Enforce low and high frequency cutoffs to avoid picking noise at band edges
    #########################################################################
    
    del_freqs = []
    for idx in kid_idx:
        if chan_freqs[idx] < low_freq_cut:
            #if output_on:
            #    print "low freq cut"
            del_freqs.append(idx)
        if chan_freqs[idx] > high_freq_cut:
            #if output_on:
            #    print "high_freq_cut"
            del_freqs.append(idx)
    for idx in del_freqs:
        kid_idx.remove(idx)
    
    #########################################################################
    # Enforce low and high frequency cutoffs for a mask window
    #########################################################################
    """
    del_freqs = []
    for i in range(len(mask_low)):
        for idx in kid_idx:
            if mask_low[i] <= chan_freqs[idx] <= mask_high[i]:
                #if output_on:
                #    print "mask cut"
                del_freqs.append(idx)
    for idx in del_freqs:
        kid_idx.remove(idx)
    """
    kid_idx = np.asarray(kid_idx)
    kid_idx = np.hstack(kid_idx)
    kid_idx = kid_idx.tolist()
    print chan_freqs[kid_idx], lo_centerfreq
    bb_target_freqs = ((chan_freqs[kid_idx]*1.0e6) - lo_centerfreq)
    print bb_target_freqs
    
    ############################################################################
    # Save target freqs to file
    ############################################################################
    
    # List of channel IDs
    chan_id = []
    if len(bb_target_freqs) > 0:
        bb_target_freqs = np.roll(bb_target_freqs, - np.argmin(np.abs(bb_target_freqs)))
        print bb_target_freqs
        for idx in kid_idx:
            f1 = np.round((bb_target_freqs + lo_centerfreq)/1.0e6, 3)
            f2 = np.round(chan_freqs[idx], 3)
            #print f1, f2
            #print np.where(f1 == f2)[0][0]
            chan_id.append(np.where(f1 == f2)[0][0])
        np.save(vna_path + '/bb_targ_freqs.npy', bb_target_freqs)
        print len(kid_idx), "KIDs found"
    else:
        print "No targ freqs found!"
    
    if output_on:
        plt.figure()
        plt.plot(chan_freqs, mags-filtermags, alpha = 0.7)
        plt.scatter(chan_freqs[kid_idx], new_mags[kid_idx], color = 'r', marker = '*')
        plt.xlabel('frequency [MHz]', fontsize = 16)
        plt.ylabel('|S21| [dB]', fontsize = 16)
        # Add channel windows
        for idx in kid_idx:
            f1 = np.round((bb_target_freqs + lo_centerfreq)/1.0e6, 3)
            f2 = np.round(chan_freqs[idx], 3)
            plt.axvspan(chan_freqs[idx] - neg_win, chan_freqs[idx] + pos_win, ymin=0.0, ymax=0.9, alpha=0.5, color = 'lightblue')
        # Add channel indices
            plt.text(chan_freqs[idx], new_mags[idx] - 2., str(np.where(f1 == f2)[0][0]), rotation = 90., size = 10, clip_on = True)
        plt.legend(loc = 'upper right', fontsize = 14)
        plt.tight_layout()
    ###############################################################################
    # Calculate inverse transfer function to flatten transfer function for VNA sweep
    ###############################################################################
    
    # Repeat filtering, but on mags in linear space (better for transfer function correction)
    filtermags = lowpass_cosine(10**(mags/20.), sweep_step, 1./smoothing_scale, 0.1 * (1.0/smoothing_scale))
    raw_mags = filtermags[np.where(sweep_freqs == lo_centerfreq)[0][0]::2*np.where(sweep_freqs == lo_centerfreq)[0][0]]
    norm_mags = np.max(raw_mags)/raw_mags
    norm_mags /= np.max(norm_mags)
    # reorder frequencies into firmware order
    ordered_mags = np.concatenate((norm_mags[len(norm_mags)/2:], norm_mags[:len(norm_mags)/2]))
    np.save(os.path.join(vna_path,'vna_amps.npy'), ordered_mags)
    if output_on:
        plt.figure()
        plt.scatter(chan_freqs[np.where(sweep_freqs == lo_centerfreq)[0][0]::2*np.where(sweep_freqs == lo_centerfreq)[0][0]], norm_mags)
        plt.xlabel('frequency [MHz]', fontsize = 16)
        plt.ylabel('Normalized Ampl.', fontsize = 16)
        plt.tight_layout()
    
    ###############################################################################
    # Create initial list of target tone amplitudes
    ###############################################################################
    
    norm_mags = np.max(filtermags)/filtermags
    norm_mags /= np.max(norm_mags)
    ordered_mags = np.concatenate((norm_mags[len(norm_mags)/2:], norm_mags[:len(norm_mags)/2]))
    targ_mags = ordered_mags[kid_idx]
    print kid_idx, targ_mags
    if output_on:
        plt.figure()
        plt.scatter(chan_freqs[kid_idx], norm_mags[kid_idx])
        plt.xlabel('frequency [MHz]', fontsize = 16)
        plt.ylabel('Normalized Ampl.', fontsize = 16)
        plt.tight_layout()
    np.save(os.path.join(vna_path, 'targ_amps.npy'), norm_mags[kid_idx])
    return
