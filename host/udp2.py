"""
Overview
________

udp2 is the Next iteration of udpcap. Here, we want to facilitate the process of pulling data
from multiple channels from multiple RFSOC's in a multiprocessing environment. 

.. note::
    A key part of python multiprocessing library is 'pickling'. This is a funny name to describe object serialization. Essentially, our code needs
    to be convertable into a stream of bytes that can be passed intoa new python interpreter process.
    Certain typs of variables such as h5py objects or sockets can't be pickled. We therefore have to create the h5py/socket objects we need post-pickle. 


:Authors: Cody Roberson
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
import time
import multiprocessing as mp

# logging.basicConfig(format=__LOGFMT, level=logging.DEBUG)

logger = logging.getLogger(__name__)


def __data_writer_process(dataqueue, chan: RFChannel, runFlag):
    """ """
    log = logger.getChild(__name__)
    log.debug(f"began data writer process <{chan.name}>")

    # Create HDF5 Datafile
    raw = data_handler.RawDataFile(chan.raw_filename)
    raw.format(0, chan.n_resonator, chan.n_attenuator)

    # Pass in the last LO sweep hhere

    while True:
        # we're done if the queue closes or we don't get any day within 10 seconds
        try:
            obj = dataqueue.get(True, 10)
        except:
            obj = None
        if obj is None:
            log.debug(f"obj is None <{chan.name}>")
            break
        t1 = time.perf_counter_ns()
        log.debug(f"Received a queue object<{chan.name}>")
        # re-Allocate Dataset
        indx, adci, adcq, timestamp = obj
        raw.resize(indx)
        log.debug("resized")
        # Get Data
        raw.adc_i[:, indx - 488 : indx] = adci
        raw.adc_q[:, indx - 488 : indx] = adcq
        raw.timestamp[indx - 488 : indx] = timestamp
        t2 = time.perf_counter_ns()
        log.debug(f"Parsed in this loop's data <{chan.name}>")
        log.debug(f"Data Writer deltaT = {(t2-t1)*1e-6} ms for <{chan.name}>")

    raw.close()
    log.debug(f"Queue closed, closing file and exiting for <{chan.name}>")


def __data_collector_process(dataqueue, chan: RFChannel, runFlag):
    """"""
    log = logger.getChild(__name__)
    log.debug(f"began data collector process <{chan.name}>")
    # Creae Socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind((chan.ip, chan.port))
        s.settimeout(10)
    except Exception as e:
        return
    # log.debug(f"Socket bound - <{chan.name}>")
    # Take Data
    idx = 488
    i = np.zeros((1024, 488))
    q = np.zeros((1024, 488))
    ts = np.zeros(488)
    log.debug(f"runflag is {runFlag.value}")

    while runFlag.value:
        t1 = time.perf_counter_ns()
        try:
            i[...] = 0
            q[...] = 0
            ts[...] = 0
            for k in range(488):
                data = np.frombuffer(bytearray(s.recv(8208)), dtype="<i")[0:2048]
                i[:, k] = data[0::2]
                q[:, k] = data[1::2]
                ts[k] = getdtime()
            dataqueue.put((idx, i, q, ts))
        except TimeoutError:
            log.warning(f"Timed out waiting for data <{chan.name}>")
            break
        idx = idx + 488
        t2 = time.perf_counter_ns()
        log.debug(f"datacollector deltaT = {(t2-t1)*1e-6} ms")
    log.debug(f"exited while loop, putting None in dataqueue for <{chan.name}> ")
    dataqueue.put(None)
    s.close()


def exceptionCallback(e):
    raise e


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
    runFlag = manager.Value(ctypes.c_bool, True)

    for chan in channels:
        dataqueue = manager.Queue()
        pool.apply_async(
            __data_writer_process,
            (dataqueue, chan, runFlag),
            error_callback=exceptionCallback,
        )
        log.debug(f"Spawned data collector process: {chan.name}")
        pool.apply_async(
            __data_collector_process,
            (dataqueue, chan, runFlag),
            error_callback=exceptionCallback,
        )
        log.debug(f"Spawned data writer process: {chan.name}")

    pool.close()
    log.info("Waiting on capture to complete")
    _ = input("Press enter to end data collection")
    log.info("Ending Data Capture; Waiting for child processes to finish")
    runFlag.value = False
    pool.join()
    log.info("Finished...")


if __name__ == "__main__":
    __LOGFMT = (
        "%(asctime)s|%(levelname)s|%(filename)s|%(lineno)d|%(funcName)s|%(message)s"
    )
    logging.basicConfig(format=__LOGFMT, level=logging.DEBUG)
    __logh = logging.FileHandler("./udp2.log")
    logging.root.addHandler(__logh)
    logger.log(100, __LOGFMT)
    __logh.flush()
    __logh.setFormatter(logging.Formatter(__LOGFMT))

    # lets test this thing, shall we?
    rfsoc = data_handler.RFChannel(
        "./rfsoc1_fakedata.h5", "127.0.0.1", 4096, "Stuffed Crust Pizza", 488, 1024, 1
    )
    rfsoc2 = data_handler.RFChannel(
        "./rfsoc2_fakedata.h5", "127.0.0.1", 4097, "Salad", 488, 1024, 1
    )
    capture([rfsoc, rfsoc2])
