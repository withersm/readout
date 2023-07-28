"""
Overview
________

udp2 is the Next iteration of udpcap. Here, we want to facilitate the process of pulling data
from multiple channels from multiple RFSOC's in a multiprocessing environment. 

.. note::
    A key part of python multiprocessing library is 'pickling'. This is a funny name to describe object serialization. Essentially, our code needs
    to be convertable into a stream of bytes that can be passed intoa new python interpreter process.
    Certain typs of variables such as h5py objects or sockets can't be pickled. We therefore have to create the h5py/socket objects we need post-pickle. 

:Authors: Cody
:Date: 2023-07-26
:Version: 1.0.0
"""
import concurrent.futures
import logging
import data_handler
import numpy as np
import socket
import data_handler
import socket
import time
from data_handler import RFChannel

logger = logging.getLogger(__name__)


def __workerprocess(chan: RFChannel):
    """
    Worker Process
    Given properties through chan, the worker process creates a new socket and binds to it. A RawDataFile is then created
    and formated. Following this, the raw file is filled out with the aquired data and then closed.
    """
    log = logger.getChild(__name__)

    log.debug(f"__worker process for {chan.name} started. ")
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind((chan.ip, chan.port))
        s.settimeout(1.0)
    except Exception as e:
        log.error(e)
        return
    log.debug(f"{chan.name} loading HDF5")
    raw = data_handler.RawDataFile(chan.raw_filename)
    raw.format(chan.n_sample, chan.n_resonator, chan.n_attenuator)

    log.debug(f"{chan.name} begin data collection")

    i = np.zeros((1024, chan.n_sample))
    q = np.zeros((1024, chan.n_sample))
    ts = np.zeros((2, chan.n_sample))
    try:
        log.debug("start collection loop")
        for k in range(chan.n_sample):
            data = np.frombuffer(bytearray(s.recv(8208)), dtype="<i")[0:2048]
            i[:, k] = data[0::2]
            q[:, k] = data[1::2]
            ts[1, k] = time.time_ns() /1e6
        log.debug("finished collection loop")
        raw.adc_i[...] = i
        raw.adc_q[...] = q
        raw.timestamp[...] = ts
        s.close()
        raw.close()
    except Exception as e2:
        log.error(str(e2))


def capture(channels: list):
    """
    For each RFChannel provided, capture(...) spawns a worker process that does the following:
        * Generates a RawDataFile
        * Creates a socket connection on the network
        * Downlinks data and populates the raw file
    """
    log = logger.getChild(__name__)
    log.info("Starting Capture")

    if channels is None or channels is []:
        log.warning("Specified list of rfsoc connections is empy/None")
        return

    with concurrent.futures.ProcessPoolExecutor() as exec:
        log.debug("submitting jobs to process pool")
        results = exec.map(__workerprocess, channels)
        log.debug("Worker Processes executed, waiting for jobs to complete.")
        exec.shutdown(wait=True)
    log.info("Capture Finished")


def test():
    import os
    t = 10
    NSAMP = 488 * t
    os.system("clear")
    # print("pretending to collect data")
    savefile = "TESTTEST.hdf"

    rfsoc1 = data_handler.RFChannel(savefile, "192.168.5.40", 4096, "rfso1", NSAMP)
    # capture([rfsoc1])
    __workerprocess(rfsoc1)

if __name__ == "__main__":
    test()