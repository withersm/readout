"""
Handles udp packets and the low level connection between the rfsoc data ports and our
computer system
"""

import logging

log = logging.getLogger(__name__)

import socket


class DataConnection:
    def __init__(self):
        pass
        self.ip = ""
        self.port = ""
        self.mac = ""

    def establish_connection(self):
        pass

    def close_connection(self):
        pass

    def get_streaming_data(self):
        pass

    def test_connection(self):
        pass
