import concurrent.futures
import multiprocessing as mlpr
import logging
import socket
from dataclasses import dataclass
import numpy as np
import time
import data_handler


@dataclass
class Connection:
    raw_df: data_handler.RawDataFile
    ip_addr: str = ""
    port: int = 0000


def data_writer(raw_df: data_handler.RawDataFile, queue):
    """
    data_writer child process. the receive_data parent process will
    pull down udp packets and pass it off to this process using a shared queue.
    :param raw_df: raw observation data file
    :param queue: process queue data
    :return: None
    """
    active = True

    # While active, get data from the queue and place into specified hdf file.
    while active:
        queuedata = queue.get()
        if queuedata is not None:
            spec_data, i, t = queuedata
            raw_df.adc_i[:, i] = spec_data[0::2]
            raw_df.adc_q[:, i] = spec_data[1::2]
            raw_df.timestamp[0, i] = t

            # hardware counter left out of firmware for the time being.
            raw_df.timestamp[1, i] = 12345678
        else:
            active = False
            raw_df.fh.flush()


def receive_udp(conn: Connection, n_samples: int):
    """
    receive_udp child process. The capture function will generate N Connection subprocesses
    of this function. Here, udp data is captured and passed off to the data_writer dataprocess.
    This is where the connection is actually bound

    :param conn: Connection
    :param n_samples: Number of samples to take
    :return: None
    """
    # create a socket and bind it to the connection
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(1.0)

    host = conn.ip_addr
    port = conn.port

    try:
        sock.bind((host, port))
    except socket.timeout:
        logging.getLogger(__name__).error(
            f"socket.timeout -> Could not bind to the socket for {host}:{port}"
        )
    except socket.error:
        logging.getLogger(__name__).error(
            f"socket.error -> Could not bind to the socket for {host}:{port}"
        )

    # create an async datawriter process
    manager = mlpr.Manager()
    pool = mlpr.Pool(1)
    queue = manager.Queue()
    pool.apply_async(data_writer, (conn.raw_df, queue))

    # Lets get to work grabbing data
    i = 0
    while i < n_samples:
        data = sock.recv(8208)
        spec_data = np.frombuffer(data, dtype="<i", offset=0)
        t = time.time_ns() / 1e3

        if data is None:
            break

        queue.put((spec_data, i, t))
        i = i + 1

    pool.close()


def capture(connection_list: list[Connection], n_samples: int):
    """
    Capture Data. Spawns N connection receive_udp processes for data taking
    :param connection_list: List of Connections
    :param n_samples: n samples to record.
    :return: None
    """
    # create subprocesses for each udp connection.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for con in connection_list:
            executor.submit(receive_udp, con, n_samples)
