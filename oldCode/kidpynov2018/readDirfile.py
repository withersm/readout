# read dirfile to parse IQ data
import numpy as np
from glob import glob

def getIQ( dirfile, chan_num):
  """
  getIQ(dirfile,chan_num)
  params: dirfile - as string corresponding to the .dir file
          chan_num - channel to plot
  """
  I = np.fromfile(dirfile+"/I_"+str(chan_num),dtype=float) 
  Q = np.fromfile(dirfile+"/Q_"+str(chan_num),dtype=float) 
  I, Q = np.array(I).flatten(), np.array(Q).flatten() # convert list to array and flatten
  return I,Q

def getAllIQ( dirfile):
  """
  getAllIQ(dirfile,chan_num)
  params: dirfile - as string corresponding to the .dir file
  """
  If = glob(dirfile+"/I_*")
  Qf = glob(dirfile+"/Q_*")
  Iall,Qall = [],[]
  for i in range(len(If)): # loop over all channels collecting timestreams
    I, Q = np.fromfile(If[i],dtype=float), np.fromfile(Qf[i],dtype=float)
    Iall.append(I); Qall.append(Q);
  Iall, Qall = np.array(Iall).flatten(), np.array(Qall).flatten() # convert list to array and flatten
  return Iall, Qall
