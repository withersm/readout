"""
this class will represent an zcu111 rfsoc running kidpy and the subsequent controls
offered for them. Tasks such as uploading firmware and configuring software registers
will be controlled by the library through here

does the rfsoc class then take control of the udp connection or shall it stay seperate? `
"""

import logging
import numpy as np

__all__ = []
__version__ = "0.1"
__author__ = "Cody Roberson"
__email__ = "carobers@asu.edu"

log = logging.getLogger(__name__)


class RFSOC:
    """

    """
    def __init__(self):
        self.identifier = ""
        self.ip_adress = ""

    def upload_bitstream(self):
        pass

    def set_fw_registers(self):
        pass

    def set_waveform(self):
        pass

