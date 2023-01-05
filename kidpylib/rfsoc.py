"""
this class will represent zcu111 rfsoc running kidpy and the subsequent controls
offered for them. Tasks such as uploading firmware and configuring software registers
will be controlled by the library through here

* does the rfsoc class then take control of the udp connection or shall it stay seperate. `
"""

import logging

log = logging.getLogger(__name__)

class rfsoc():
    def __init__(self):
        pass

