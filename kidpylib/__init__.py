"""
Kidpy Lib

Kidpy is now taking the form as an extendable library that offers control of the mkid readout
on the rfsoc. The main difference to old kidpy is that this offers a form of unified control over
entire readout systems with minimal effor on the part of the scientest. This project shall be considered a success
if this cause is met.
"""

import sys

# Ensure compatibility
assert (sys.version_info.major == 3), "Python 3.9 or greater required"
assert (sys.version_info.minor >= 9), "Python 3.9 or greater required"

# Import main kidpylib and add it to the top level mainspace relative to the package
from . import kidpy
from .kidpy import *