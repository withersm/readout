# READOUT SYSTEM
This is a pre-alpha repository for full system readout implementation. The goal is to get something going outside of
our standard jupyter notebook flow.

## File Structure

### host
This contains all notebooks and python files that will run on a host computer such as a server.
* host/kidpy.py
    This is the main tui based control of the KIDPY firmware on the rfsoc. This connects to the redis server 
    and publishes data as a means to control the rfsoc. Data streams taken shall use the HDF5 file standard.

### rfsoc
This contains all notebooks and python files which will run on the RFSOC
 * rfsoc/rfsocInterface.py
    Pynq interface for our firmware. Waveforms are generated and uploaded here. Snapshots of the various DSP components can be taken here however, this
    file is not intended to plot or display them. Instead the data should be handed off to another process such as through redis.

* rfsoc/redisControl.py 
    Connect to a redis server on some host pc / server. It shall subscribe/join command and data message channels.
    Apon receiving for example an "upload bitstream command", it shall utilize functions in the rfsocInterface to accomplish that task.
    This will towards later revisions be part of the startup process such that it's ready and listening for commands


