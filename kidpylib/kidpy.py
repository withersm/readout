"""
Main kidpy library.
"""
__all__ = []
__version__ = "0.1"
__author__ = "Cody Roberson"
__email__ = "carobers@asu.edu"

import numpy as np
import logging
import kidpylib.rfsoc as rf
import kidpylib.config as conf

log = logging.getLogger(__name__)


class Kidpy:
    """
    <kidpy docstring>
    """

    def __init__(self):
        log.info("kidpy init")
        # Initialize subcomponents
        self.g = rf.rfsoc()
        self.c = conf.GeneralConfig("/home/cody/Desktop/myconf.ini")

    def find_rfsocs(self):
        pass

    def establish_rfsoc_connections(self):
        """Find RFSOCs and establish connections

        Enumerates rfsocs on the network and creates
        :return:
        """
        pass
