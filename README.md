
<h2 style="text-align: center"> The Kinetic Inductance Detector Python Library </h2>
<p style="text-align: center" align="center">
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
with other projects. You can utilize Anaconda or Python's `venv` package to this end.

Example:

Create an environment `python3 -m venv kp3env`

Activate the environment `source kp3env/bin/activate`

### Dependency Requirements
1. Python 3.8+
2. Redis 6.2+ and it's build requirements

#### Other Reqs
1. Ethernet card supporting jumbo frames of 8208 bytes or more
2. Ubuntu 20.04+ LTS
3. Ubuntu network configured for a mtu of 9000


Please use requirements.txt to satisfy dependencies. 
( Instructions shall assume you are using a Python Virtual Enviornment)

`pip install -r requrements.txt`

### bitstream

Git does not include the bitstream required by the software.  
[Please download it here](https://www.dropbox.com/s/sogkt112b25eoxk/202306091243_silver_blast_fixedeth_vivado2020.2_bit.zip?dl=1)
 and unzip it into ./rfsoc/  before uploading the folder to the RFSOC.

---

## Usage
_how will users approach using the library to take data_


### Contributing
Contributions should be documented using reStructuredText Markup. This is 
for the future to help keep things clearer in terms of documentation. Examples
for documention functions and classes are placed in the docs folder.


### Modifications
_How users will further implement their application within or utilizing this library_

---

## Documentation
[SEE Read The Docs](https://asu-astronomical-instrumentation.github.io/readout/docs/build/html/index.html)

## License
_text here_

## Development Notes

1. rfsocInterface:writeWaveform Function signature has changed, breaks rfsoc/rediscontrol 20230612-1857





