import numpy as np
import json
import redis

r = redis.Redis(host="192.168.2.10")

def ulFirmware():
    cmdDict = {"cmd": "ulBitstream", "args":[]}
    cmdstr = json.dumps(cmdDict)
    r.publish("picard", cmdstr)

def setRegs():
    cmdDict = {"cmd": "initRegs", "args":[[]]}
    cmdstr = json.dumps(cmdDict)
    r.publish("picard", cmdstr)


"""
In [80]: cmdDict = {"cmd": "ulBitstream", "args":[[]]}

In [81]: cmdstr = json.dumps(cmdDict)

In [82]: r.publish("picard", cmdstr)
Out[82]: 2

In [83]: cmdDict = {"cmd": "initRegs", "args":[[]]}

In [84]: cmdstr = json.dumps(cmdDict)

In [85]: r.publish("picard", cmdstr)
Out[85]: 2

In [86]: cmdDict = {"cmd": "ulWaveform", "args":[]}

In [87]: cmdstr = json.dumps(cmdDict)

In [88]: r.publish("picard", cmdstr)
Out[88]: 2

In [89]: cmdDict = {"cmd": "ulWaveform", "args":[[]]}

In [90]: cmdstr = json.dumps(cmdDict)

In [91]: r.publish("picard", cmdstr)
Out[91]: 2

In [92]: cmdDict = {"cmd": "ulBitstream", "args":[[]]}

In [93]: cmdstr = json.dumps(cmdDict)

In [94]: r.publish("picard", cmdstr)
"""
