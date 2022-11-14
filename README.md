# READOUT SYSTEM
This is a pre-alpha repository for full system readout implementation. The goal is to get something going outside of
our standard jupyter notebook flow.

# Documentation and Notes - Development

## Library File Structure
```
kidpylib
├── config.py
├── devices
│   ├── rudat.py
│   ├── udx.py
│   └── valon5009.py
├── kidpy.py
├── plot.py
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


___
## Components

Config:
  - what configurations would be handy to access?
    - [ ] redis host
    - [ ] redis port
    - [ ] data_folder
    - [ ] gateware image
    - [ ] arbitrary N waveforms?
    - [ ] arbirary M rf systems LO frequencies?
    - [ ] something to make IF controls generic?

Kidpy:
  - What would the primary function here be?
    - Enumerate available rfsoc's
    - Connect to the rfsocs
      - upload bitstreams
      - initialize their connections

### TODO

Data Handler: 
- [ ] Add attributes to the misc variables section
  - [ ] bbfreq
  - [ ] chan mask
  - [ ] detector xdelta
  - [ ] detector ydelta
  - [ ] global azimuths
  - [ ] sample rate
  - [ ] tilenum
  - [ ] tonepower
  - [ ] collectiontype

- [ ] guess I have some questions to investigate


___


## Run Procedure
1. Power on the RFSOC and readout server
2. open up 2 terminals and ssh them both into the readout server
3. ping the rfsoc to ensure the connection is established
4. in terminal 0, ssh into the rfsoc `ssh xilinx@rfsoc`, password is xilinx
5. in terminal 1, start the redis server `redis-server /etc/redis/redis.conf &`
6. in terminal 0, navigate to /home/xilinx/readout/ and run `sudo python3 redisControl.py`
7. Once you see the following in terminal 0, proceed to the next step

```
library loaded!
Successfully subscribed to Captain Picard, Awaiting Commands...
init
Successfully subscribed to PING, Awaiting Commands...
Starting Listener
```

8. in terminal 1, navigate to kidpy `cd ~/readout/host/`
9. in terminal 1, activate python `py3` and run kidpy `python kidpy.py`
10. upload the bitstream, setup udp and registers, write your waveform
11. raw data is currently configured to save to /data/, this can be modified in generalConfig.conf in the readout
folder.


### READOUT SERVER NOTE
interface config 
ifconfig \<interface> \<ip> netmask \<addr> mtu 9000 hw ether \<mac>

ex: 
    ifconfig ens1f3 192.168.3.40 netmask 255.255.255.0 mtu 9000 hw ether 80:3f:5d:09:2b:b0


