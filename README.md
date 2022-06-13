# READOUT SYSTEM
This is a pre-alpha repository for full system readout implementation. The goal is to get something going outside of
our standard jupyter notebook flow.

## File Structure

### host
This contains all notebooks and python files that will run on a host computer such as a server.

### rfsoc
This contains all notebooks and python files which will run on the RFSOC
 * rfsoc/rfsocInterface.py
    Pynq interface for our firmware. Waveforms are generated and uploaded here. Snapshots of the various DSP components can be taken here however, this
    file is not intended to plot or display them. Instead the data should be handed off to another process such as through redis.

* rfsoc/redisControl.py (Planned)
    This is a planned redis client. It shall connect to a redis server on some host pc / server. It shall subscribe/join command and data message channels.
    Apon receiving for example an "upload bitstream command", it shall utilize functions in the rfsocInterface to accomplish that task.
