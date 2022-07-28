# READOUT SYSTEM
This is a pre-alpha repository for full system readout implementation. The goal is to get something going outside of
our standard jupyter notebook flow.

## File Structure

### host
This contains all notebooks and python files that will run on a host computer such as a server.

* host/kidpy.py
    This is the main tui based control of the KIDPY firmware on the rfsoc. This connects to the redis server 
    and publishes data as a means to control the rfsoc. 

* host/udpcap.py
    data is captured and saved here from the ethernet port on the readout server. This library features two main
    user functions of note. Short Data Capture and Long Data Capture. The short data capture runs on a single core processor however
    the long data capture requires a multicore processor. 

* host/generalConfig.conf
    Settings are stored here such as redis server configuration, data save path, etc.
    **Note that this assumed there is a folder in the root directory named 'data' and has it's permissions set to allow read/write from all users**

### rfsoc
This contains all notebooks and python files which will run on the RFSOC
 * rfsoc/rfsocInterface.py
    Pynq interface for our firmware. Waveforms are generated and uploaded here. Snapshots of the various DSP components can be taken here however, this
    file is not intended to plot or display them. Instead the data should be handed off to another process such as through redis.

* rfsoc/redisControl.py 
    Connect to a redis server on some host pc / server. It shall subscribe/join command and data message channels.
    Apon receiving for example an "upload bitstream command", it shall utilize functions in the rfsocInterface to accomplish that task.
    This will towards later revisions be part of the startup process such that it's ready and listening for commands

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
