"""
Sweeps module is where lo sweep code can be accessed.
"""


import numpy as np
import valon5009


#######################################################
# Temporary Home for DSP Functions
# These should get a dedicated DSP python file
#######################################################
def sweep(loSource, udp, f_center, freqs, N_steps=500, freq_step=0.0):
    """
    Actually perform an LO Sweep using valon 5009's and save the data

    :param loSource:
        Valon 5009 Device Object instance
    :type loSource: valon5009.Synthesizer
    :param f_center:
        Center frequency of upconverted tones
    :param freqs: List of Baseband Frequencies returned from rfsocInterface.py's writeWaveform()
    :type freqs: List

    :param udp: udp data capture utility. This is our bread and butter for taking data from ethernet
    :type udp: udpcap.udpcap object instance

    :param N_steps: Number of steps with which to do the sweep.
    :type N_steps: Int

    Credit: Dr. Adrian Sinclair (adriankaisinclair@gmail.com)
    """
    tone_diff = np.diff(freqs)[0] / 1e6  # MHz
    if freq_step > 0:
        flo_step = freq_step
    else:
        flo_step = tone_diff / N_steps
    flo_start = f_center - flo_step * N_steps / 2.0  # 256
    flo_stop = f_center + flo_step * N_steps / 2.0  # 256

    flos = np.arange(flo_start, flo_stop, flo_step)
    udp.bindSocket()

    def temp(lofreq):
        # self.set_ValonLO function here
        loSource.set_frequency(valon5009.SYNTH_B, lofreq)
        # Read values and trash initial read, suspecting linear delay is cause..
        Naccums = 50
        I, Q = [], []
        for i in range(10):  # toss 10 packets in the garbage
            udp.parse_packet()

        for i in range(Naccums):
            # d = udp.parse_packet()
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
        start_ind = np.min(np.argwhere(Imed != 0.0))
        Z = Z[start_ind : start_ind + len(freqs)]

        print(".", end="")

        return Z

    sweep_Z = np.array([temp(lofreq) for lofreq in flos])

    f = np.zeros([np.size(freqs), np.size(flos)])
    for itone, ftone in enumerate(freqs):
        f[itone, :] = flos * 1.0e6 + ftone
    #    f = np.array([flos * 1e6 + ftone for ftone in freqs]).flatten()
    sweep_Z_f = sweep_Z.T
    #    sweep_Z_f = sweep_Z.T.flatten()
    udp.release()
    ## SAVE f and sweep_Z_f TO LOCAL FILES
    # SHOULD BE ABLE TO SAVE TARG OR VNA
    # WITH TIMESTAMP

    # set the LO back to the original frequency
    loSource.set_frequency(valon5009.SYNTH_B, f_center)

    return (f, sweep_Z_f)


def loSweep(
    loSource,
    udp,
    freqs=[],
    f_center=400.0,
    N_steps=500,
    freq_step=1.0,
    savefile="s21",
):
    """Perform a stepped frequency sweep centered at f_center and save result as s21.npy file

    f_center: center frequency for sweep in [MHz], default is 400
    """
    #    print(freqs)
    f, sweep_Z_f = sweep(
        loSource,
        udp,
        f_center,
        np.array(freqs) / 2,
        N_steps=N_steps,
        freq_step=freq_step,
    )
    np.save(savefile + ".npy", np.array((f, sweep_Z_f)))
    print("LO Sweep s21 file saved.")
