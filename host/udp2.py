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
from ctypes import c_bool
import ctypes
import logging
import data_handler
import numpy as np
import socket
import data_handler
from data_handler import getdtime
import socket
from data_handler import RFChannel
import multiprocessing as mp

__LOGFMT = "%(asctime)s|%(levelname)s|%(filename)s|%(lineno)d|%(funcName)s|%(message)s"

# logging.basicConfig(format=__LOGFMT, level=logging.DEBUG)
logging.basicConfig(format=__LOGFMT, level=logging.INFO)
logger = logging.getLogger(__name__)
__logh = logging.FileHandler("./kidpy.log")
logging.root.addHandler(__logh)
logger.log(100, __LOGFMT)
__logh.flush()
__logh.setFormatter(logging.Formatter(__LOGFMT))


def __data_writer_process(dataqueue, chan: RFChannel):
    """ """
    log = logger.getChild(__name__)
    log.debug(f"began data writer process <{chan.name}>")

    # Create HDF5 Datafile
    raw = data_handler.RawDataFile(chan.raw_filename)
    raw.format(0, chan.n_resonator, chan.n_attenuator)

    # Pass in the last LO sweep hhere

    while True:
        # we're done if the queue closes
        obj = dataqueue.get()
        if obj is None:
            break

        # re-Allocate Dataset
        indx, adci, adcq, timestamp = obj
        raw.resize(indx)
        # Get Data
        raw.adc_i[:, indx - 488 : indx] = adci
        raw.adc_q[:, indx - 488 : indx] = adcq
        raw.timestamp[indx - 488 : indx] = timestamp

    raw.close()
    log.debug("Queue closed, closing file and exiting...")


def __data_collector_process(dataqueue, chan: RFChannel, runFlag):
    """"""
    log = logger.getChild(__name__)
    log.debug(f"began data collector process <{chan.name}>")
    # Creae Socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind((chan.ip, chan.port))
        s.settimeout(1.0)
    except Exception as e:
        dataqueue.close()
        return

    # Take Data
    idx = 488
    i = np.zeros((1024, 488))
    q = np.zeros((1024, 488))
    ts = np.zeros(488)

    while runFlag.value:
        if (
            idx >= 2440
        ):  # prevent runaway, just in case KeyboardInterrupt doesn't work as intended
            break
        i[...] = 0
        q[...] = 0
        ts[...] = 0
        for k in range(488):
            data = np.frombuffer(bytearray(s.recv(8208)), dtype="<i")[0:2048]
            i[:, k] = data[0::2]
            q[:, k] = data[1::2]
            ts[k] = (getdtime(), 1)
        dataqueue.put((idx, i, q, ts))
        idx = idx + 488

    dataqueue.close()
    s.close()


def capture(channels: list):
    """
    For each RFChannel provided, capture(...) spawns a worker process that does the following:
        * Generates a RawDataFile
        * Creates a socket connection on the network
        * Downlinks data and populates the raw file
    """
    log = logger.getChild(__name__)

    if channels is None or channels is []:
        log.warning("Specified list of rfsoc connections is empy/None")
        return

    # mmmmmmmmmmmmmmmmmmmmm
    manager = mp.Manager()
    pool = mp.Pool()

    log.info("Starting Capture Processes")
    runFlag = manager.Value(ctypes.c_bool, False)
    try:
        for chan in channels:
            dataqueue = manager.Queue()
            pool.apply_async(__data_writer_process, (dataqueue, chan))
            log.debug(f"Spawned data collector process: {chan.name}")
            pool.apply_async(__data_collector_process, (dataqueue, chan, runFlag))
            log.debug(f"Spawned data writer process: {chan.name}")

        pool.close()
        log.info("Waiting on capture to complete")
        pool.join()
    except KeyboardInterrupt:
        log.info("Ending Data Capture")
        runFlag.value = False

    log.info("Finished...")


if __name__ == "__main__":
    # lets test this thing, shall we?
    rfsoc = data_handler.RFChannel(
        "./test_raw.h5", "192.168.5.40", 4096, "rfsoc1-test", 488, 1024, 1
    )
    capture([rfsoc])
