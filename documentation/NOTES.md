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



___
## TODO MILESTONES

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

