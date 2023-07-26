"""
Description
___________

udp2 is the Next iteration of udpcap. The idea here is to facilitate taking data in a multiprocess enviornment
from multiple channels from multiple RFSOC's in our mKID readout system.

:Authors: Cody
:Date: 2023-07-26
:Version: 1.0.0
"""
import concurrent.futures
import logging
import socket
from dataclasses import dataclass
import numpy as np
import time
import data_handler

logger = logging.getLogger(__name__)


@dataclass
class Connection:
    raw_df: data_handler.RawDataFile
    ip_addr: str = ""
    port: int = 0000


def __receive_udp(conn: Connection, n_samples: int):
    """
    UDP data receiver process. *This function is intended to be executed from a ProcessPoolExecutor*

    Parameters
    __________

    :param conn: RFSOC udp connection to obtain data from.
    :type conn: Connection
    :param n_samples: Number of samples to take
    :type n_samples: int
    :return: Does not return anything
    """
    log = logger.getChild(__name__)
    # create a socket and bind it to the connection
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(1.0)

    host = conn.ip_addr
    port = conn.port
    raw_df = conn.raw_df

    try:
        sock.bind((host, port))
    except socket.timeout:
        log.error(f"socket.timeout -> Could not bind to the socket for {host}:{port}")
    except socket.error:
        log.error(f"socket.error -> Could not bind to the socket for {host}:{port}")

    # Take data and add into HDF File
    i = 0
    log.debug(f"Beginning data taking loop for {conn.ip_addr}:{conn.port}")

    while i < n_samples:
        data = sock.recv(8208)
        spec_data = np.frombuffer(data, dtype="<i", offset=0)
        t = time.time_ns() / 1e3

        if data is None:
            break

        raw_df.adc_i[:, i] = spec_data[0::2]
        raw_df.adc_q[:, i] = spec_data[1::2]
        raw_df.timestamp[0, i] = t

        i = i + 1


def capture(connection_list: list, n_samples: int):
    """
    Capture Data. Spawns N connection receive_udp processes for data taking

    Parameters
    __________

    :param connection_list: List of Connections
    :param n_samples: n samples to record.
    :return: None
    """
    log = logger.getChild(__name__)
    # create subprocesses for each udp connection.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for con in connection_list:
            log.debug(f"Spawning receive_udp process for {con.ip_addr}:{con.port}")
            executor.submit(__receive_udp, con, n_samples)
