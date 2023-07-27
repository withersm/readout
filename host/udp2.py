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
    iter = np.arange(0, chan.n_sample, 1)
    try:
        for i in iter:
            data = np.frombuffer(s.recv(8208), dtype="<i")
            raw.adc_i[:, i] = data[0::2]
            raw.adc_q[:, i] = data[1::2]
            raw.timestamp[1, i] = time.time_ns() / 1e3
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
