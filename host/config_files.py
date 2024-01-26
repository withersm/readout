#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 14:52:58 2023

@author: matthew
"""

import yaml

class config_files():
    """
    A class with methods for reading configuration files.
    """
    
    def read(filename):
        """
       Read the configuration file and load the data.

       :return: The loaded configuration data.
       :rtype: dict
       """
        with open(filename,'r') as file:
            config = yaml.safe_load(file)
            
        return config
    
if __name__ == '__main__':
    pass
    """
    config = config_files.read('run_1_control_config.yaml')
    print(type(config))
    """