# adrian sinclair
import numpy as np

def despike(mags):
    # despike magnitudes - set spike threshold after taking diff(mags)
    thresh = 0.5
    diffmags = np.diff(mags)
    spike_check_pos = diffmags[diffmags >= 0.0]
    spike_check_neg = diffmags[diffmags < 0.0] 
    spikes = 0 
    m = mags
    if (len(spike_check_pos) == 0) and (len(spike_check_neg) > 0):
        spikes = np.where(diffmags<=-1*thresh)[0]
        print "only negative spikes found"
    elif (len(spike_check_pos) > 0) and (len(spike_check_neg) == 0):
        spikes = np.where(diffmags>=thresh)[0]
        print "only positive spikes found"
    elif (len(spike_check_pos) == 0) and (len(spike_check_neg) == 0):
        print "no spikes found"
    elif (len(spike_check_pos) >= 0) and (len(spike_check_neg) >= 0):
        spikepos = np.where(diffmags>= thresh)[0]
        spikeneg = np.where(diffmags<=-1.0*thresh)[0]
        spikes = np.concatenate((spikepos,spikeneg))
        print "both pos+neg spikes found"
    else:
        print "indeterminate despike"
    
    if type(spikes) != int:
        for i in range(len(spikes)):
            if spikes[i] < 2:
                m[spikes[i]] = (mags[spikes[i]+2]+mags[spikes[i]+3])/2.0 # check for spikes at beginning
            elif spikes[i] > (len(mags)-3): #(spikes[len(spikes)-1]) < (spikes[len(spikes)-1]-2): # check for spikes at end
                m[spikes[i]] = (mags[spikes[i]-2]+mags[spikes[i]-3])/2.0
            else: # spikes in between
                m[spikes[i]] = (mags[spikes[i]-1]+mags[spikes[i]+1])/2.0
    return m 
