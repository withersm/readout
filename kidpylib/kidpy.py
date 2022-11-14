"""
Main Library
"""

import logging


_log = logging.getLogger(__name__)


class kidpy:
    def __init__(self):
        logging.basicConfig(level=logging.DEBUG)
        log = logging.getLogger(__name__)
        log_fh = logging.FileHandler("kidpydata/kidpylib.log")
        log_format = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s][%(funcName)s] - %(message)s')
        log_fh.setFormatter(log_format)
        log.addHandler(log_fh)
        _log.info("kidpy lib object initialized")


