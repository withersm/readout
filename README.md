
<h2 style="text-align: center"> The Kinetic Inductance Detector Python Library </h2>
<p style="text-align: center">
<a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

A library for controlling and taking data from Kinetic Inductance Detector readout electronics.
Users will use this base library to build up an application specific to their project. The idea
beging that most of the low level details are taken care of. Users should focus on the minutia of
their integration and the science they would like to perform. A nominal system consists of a
ZCU111 RFSOC connected to an Intermediate Frequency Up/Down rf converter which is in turn, 
connected to detectors in a cryostat. 

Control over ethernet is facilitated via a Redis server. 

---

## Installation

### Virtual Environment
It's best practice to use a virtual environmet instead of your system python installation.
With virtual enviornments, you can ensure there are no conflicting packages or package versions
with other projects. You can utilize Anaconda or Pythons basic `venv` package to this end.


Create an environment `python3 -m venv kp3env`

Activate the environment `source kp3env/bin/activate`

### Dependency Requirements
1. Python 3.8+
2. Redis 6.2+ and it's build requirements

#### Other Reqs
1. Ethernet card supporting jumbo frames
2. Ubuntu 20.04+ LTS


Please use requirements.txt to satisfy dependencies. 
( Instructions shall assume you are using a Python Virtual Enviornment)

`pip install -r requrements.txt`

---

## Usage
_how will users approach using the library to take data_


### Contributing
_how and what kind of contributions will be added to the library_


### Modifications
_How users will further implement their application within or utilizing this library_

---

## Documentation
_YES, IT IS **REAL**_

### Library Structure
```
kidpylib
├── config.py
├── data_handler.py
├── devices
│   ├── rudat.py
│   ├── udx.py
│   └── valon5009.py
├── __init__.py
├── kidpy.py
├── plot.py
├── rfsoc.py
├── science
│   └── sweeps.py
└── udp.py

config      //Loads configuration data from specified config file. Uses the python builtint configparser library
rudat       //Controls MiniCircuits Rudat digitally controlled variable attenuator.
udx         //Controls ASU/Alphacore Up-Down Converter (IF Slices)
valon5009   //Controls Valon 5009 Digital Synthesizers
kidpy       //Main Library interface for controlling the RFSOC
science/    //Contains User provided Science python files. Ideally this is where a majority of kidpy will be modified
udp         //Handles UDP Connections
dataHander  //Handles many if not most of our data needs throughout the library.
```

## License
_text here_