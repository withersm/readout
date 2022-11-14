"""
Handles the process of loading and creating the relative configuration to the readout system

credit: Alexandra Zaharia on Python Configuration and Data Classes.
    https://alexandra-zaharia.github.io/posts/python-configuration-and-dataclasses/
"""
import configparser
import numpy as np
from dataclasses import dataclass

@dataclass
class GeneralConfig():
    redis_host: str
    redis_port: int
    data_folder: str



