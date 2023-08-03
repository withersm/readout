"""
:Authors: - Cody Roberson
          - Jack Sayers
          - Daniel Cunnane

:Date: 2023-08-01

:Version: 2.0.0

Brief overview
--------------
Here we define the data types and format that is utilized throughout the project.
Our primary observation data is stored using HDF5  `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_
via the `h5py python library <https://www.h5py.org/>`_


RawDataFile
--------------
The RawDataFile class is analogous to standard camera's raw file. Detector data is captured, unprocessed into this file.


Dimensions
--------------
n_resonator


ObservationDataFile
-------------------
000



"""

import h5py
import os
import time
import logging
import numpy as np
from dataclasses import dataclass
from datetime import date
from datetime import datetime
import glob

logger = logging.getLogger(__name__)


@dataclass
class RFChannel:
    """
    **Dataclass**; contains the state information relevant to an observation on a
    per-RF-chain basis going into a/the cryostat. Each rf channel gets it's own RF chain and
    subsequently it's own UDP connection. The information used here is pulled into the RawDataFile.


    :param str raw_filename: path to where the HDF5 file shall exist.

    :param str ip: ip address of UDP stream to listen to

    :param int port: port of UDP to listen to, default is 4096

    :param str name: Friendly Name to call the channel. This is relevant to logs, etc

    :param int n_sample: n_samples to take. While this is used to format the
        rawDataFile, it should be
        left as 0 for now since udp2.capture() dynamically resizes/allocates it's datasets.

    :param int n_resonator: Number of resonators we're interested in / Tones we're generating
        from the RFSOC dac mapped to this Channel.

    :param int n_attenuator: Dimension N Attenuators

    :param np.ndarray baseband_freqs: array of N resonator tones

    :param np.ndarray attenuator_settings: Array of N Attenuator settings

    :param int sample_rate: Data downlink sample rate ~488 Samples/Second

    :param int tile_number: Which tile this rf channel belongs to

    :param int rfsoc_number: Which rfsoc unit is used

    :param int chan_number: channel # of the rfsoc being used

    :param int ifslice_number: Which IF slice the rfsoc channel # is using.

    :param str lo_sweep_filename: path to the LO sweep data with which to append to the rawDataFile.

    """

    raw_filename: str
    ip: str
    baseband_freqs: np.ndarray
    tone_powers: np.ndarray
    attenuator_settings: np.ndarray
    port: int = 4096
    name: str = "Untitled"
    n_sample: int = 488
    n_resonator: int = 1000
    n_attenuators: int = 2
    sample_rate: np.ndarray = 488
    tile_number: int = 0
    rfsoc_number: int = 0
    chan_number: int = 0
    ifslice_number: int = 0
    lo_sweep_filename: str = "/path/to/some_s21_file_here.npy"


class RawDataFile:
    """A raw hdf5 data file object for incoming rfsoc-UDP data streams.

    UDP packets containing our downsampled data streaming from the RFSOC to the readout computer
    will be captured and saved to this hdf5 filetype.

    :param path: /file/path/here/file.h5
    :param n_sample: (Dimension) The number of data samples collected (i.e., sample_rate * length_of_file_in_seconds)

    .. note::
        As of a recent change to the data collection process, n_samples is defaulted to 0 and
        repeatedly resized during data collection. This is due to the fact that an indefinite quantity of data was
        desired.

    :param n_resonator: (Dimension) The number of resonance tones
    :param overwrite(optional): When true, signals that if an hdf5 file exists, overwrite its contents completely and
        do not contents. By default, this is false and the file is opened in append mode and subsequently read.
    """

    def __init__(self, path, overwrite=False):
        log = logger.getChild(__name__)

        self.filename = path
        if not os.path.exists(path):
            log.debug("File not found, creating file...")
            try:
                self.fh = h5py.File(self.filename, "a")
            except Exception as e:
                log.error(e)
                raise
        else:
            log.debug("File already exists.")
            if overwrite:
                log.debug("Overwriting file")
                try:
                    self.fh = h5py.File(self.filename, "w")
                except Exception as e:
                    log.error(e)
                    raise
            else:
                log.debug("Opening File as append")
                try:
                    self.fh = h5py.File(self.filename, "a")
                except Exception as e:
                    log.error(e)
                    raise
                self.read()

    def format(self, n_sample: int, n_resonator: int, n_attenuator: int):
        """
        When called, populates the hdf5 file with our desired datasets
        """
        # ********************************* Dimensions *******************************
        self.n_sample = self.fh.create_dataset(
            "dimension/n_sample",
            (1,),
            dtype=h5py.h5t.NATIVE_UINT64,
        )
        self.n_resonator = self.fh.create_dataset(
            "dimension/n_resonator",
            (1,),
            dtype=h5py.h5t.NATIVE_UINT64,
        )
        self.n_attenuators = self.fh.create_dataset(
            "dimension/n_attenuators",
            (1,),
            dtype=h5py.h5t.NATIVE_UINT64,
        )
        # ******************************** Global Data ******************************
        self.attenuator_settings = self.fh.create_dataset(
            "global_data/attenuator_settings",
            (2,),
            dtype=h5py.h5t.NATIVE_DOUBLE,
        )
        self.baseband_freqs = self.fh.create_dataset(
            "global_data/baseband_freqs", (n_resonator,)
        )
        self.sample_rate = self.fh.create_dataset("global_data/sample_rate", (1,))
        self.tile_number = self.fh.create_dataset(
            "global_data/tile_number", (n_resonator,), dtype=h5py.h5t.NATIVE_INT32
        )
        self.tone_powers = self.fh.create_dataset(
            "global_data/tone_powers", (n_resonator,), dtype=h5py.h5t.NATIVE_DOUBLE
        )
        self.rfsoc_number = self.fh.create_dataset(
            "global_data/rfsoc_number", (1,), dtype=h5py.h5t.NATIVE_INT32
        )
        self.chan_number = self.fh.create_dataset(
            "global_data/chan_number", (1,), dtype=h5py.h5t.NATIVE_INT32
        )
        self.chan_number.attrs.create(
            "info", "possibility of multiple raw files per channel per RFSOC"
        )
        self.ifslice_number = self.fh.create_dataset(
            "global_data/ifslice_number", (1,), dtype=h5py.h5t.NATIVE_INT32
        )

        # ****************************** Time Ordered Data *****************************
        self.adc_i = self.fh.create_dataset(
            "time_ordered_data/adc_i",
            (1024, n_sample),
            chunks=(1024, 488),
            maxshape=(1024, None),
            dtype=h5py.h5t.STD_I32LE,
        )
        self.adc_q = self.fh.create_dataset(
            "time_ordered_data/adc_q",
            (1024, n_sample),
            chunks=(1024, 488),
            maxshape=(1024, None),
            dtype=h5py.h5t.STD_I32LE,
        )
        self.lo_freq = self.fh.create_dataset(
            "time_ordered_data/lo_freq",
            (488,),
            chunks=(488,),
            maxshape=(None,),
        )

        self.timestamp = self.fh.create_dataset(
            "time_ordered_data/timestamp", (n_sample,), chunks=(488,), maxshape=(None,)
        )
        self.fh.flush()

    def resize(self, n_sample: int):
        """
        resize the dynamically allocated datasets. This will mostly be used by udp2.py to
        expand the data file to accomodate more data.
        """
        self.n_sample[0] = n_sample
        self.adc_i.resize((1024, n_sample))
        self.adc_q.resize((1024, n_sample))
        self.timestamp.resize((n_sample,))

    def set_global_data(self, chan: RFChannel):
        self.attenuator_settings[:] = chan.attenuator_settings
        self.baseband_freqs[:] = chan.baseband_freqs
        self.sample_rate[0] = chan.sample_rate
        self.tile_number[:] = chan.tile_number
        self.tone_powers[:] = chan.tone_powers
        self.tile_number[0] = chan.tile_number
        self.rfsoc_number[0] = chan.rfsoc_number
        self.ifslice_number[0] = chan.ifslice_number
        self.n_resonator[0] = chan.n_resonator
        self.n_attenuators[0] = chan.n_attenuators

    def read(self):
        """
        When called, the hdf5 file is read into this class instances variables to give them a nicer handle to work with.
        read() is called when a RawDataFile object is initialized and a datafile bearing the same name exists. That file may not
        have the same data or be from an older version of this file in which case an error may occur. Future iterations should be made
        more robust.

        The code used below to read the datasets from the RawDataFile was actually generated from gen_read given a blank RawDataFile.

        .. warning::
            changes to the name of a dataset must be identical to the instance variable identifier with which that dataset belongs.
            self.identifier = self.fh["/some/path/here/identifier"]
        """
        log = logger.getChild(__name__)

        try:
            self.n_attenuators = self.fh["/dimension/n_attenuators"]
            self.n_resonator = self.fh["/dimension/n_resonator"]
            self.n_sample = self.fh["/dimension/n_sample"]

            self.attenuator_settings = self.fh["/global_data/attenuator_settings"]
            self.baseband_freqs = self.fh["/global_data/baseband_freqs"]
            self.chan_number = self.fh["/global_data/chan_number"]
            self.ifslice_number = self.fh["/global_data/ifslice_number"]
            self.rfsoc_number = self.fh["/global_data/rfsoc_number"]
            self.sample_rate = self.fh["/global_data/sample_rate"]
            self.tile_number = self.fh["/global_data/tile_number"]
            self.tone_powers = self.fh["/global_data/tone_powers"]
            self.lo_freq = self.fh["/time_ordered_data/lo_freq"]
            if "/global_data/lo_sweep" in self.fh:
                self.lo_sweep = self.fh["/global_data/lo_sweep"]
            else:
                self.lo_sweep = None

            self.timestamp = self.fh["/time_ordered_data/timestamp"]
            self.adc_i = self.fh["/time_ordered_data/adc_i"]
            self.adc_q = self.fh["/time_ordered_data/adc_q"]

        except Exception as e:
            log.error(e)

    def append_lo_sweep(self, sweeppath: str):
        """
        Call this function to provide the RawDataFile with
        """
        log = logger.getChild(__name__)
        log.debug(f"Checking for file {sweeppath}")
        if os.path.exists(sweeppath):
            log.debug("found sweep file, appending.")
            sweepdata = np.load(sweeppath)
            self.fh.create_dataset("/global_data/lo_sweep", data=sweepdata)
        else:
            log.info("Specified sweep file does not exist. Will not append.")

    def close(self):
        """
        Close the RawDataFile
        """
        self.fh.close()


class TelescopeDataFile:
    """
    Telescope Data File.
    """

    def __init__(self, path, overwrite=False) -> None:
        log = logger.getChild(__name__)

        self.filename = path
        if not os.path.exists(path):
            log.debug("File not found, creating file...")
            try:
                self.fh = h5py.File(self.filename, "a")
            except Exception as e:
                log.error(e)
                raise
        else:
            log.debug("File already exists.")
            if overwrite:
                log.debug("Overwriting file")
                try:
                    self.fh = h5py.File(self.filename, "w")
                except Exception as e:
                    log.error(e)
                    raise
            else:
                log.debug("Opening File as append")
                try:
                    self.fh = h5py.File(self.filename, "a")
                except Exception as e:
                    log.error(e)
                    raise
                self.read()

    def format():
        """
        format TBD
        """
        pass

    def read():
        """
        depends on format....
        """
        pass


class ObservationDataFile:
    """
    The main Observation datafile which will encompass the data gatherd during an observation
    period. raw rfsoc data is migrated into this data file after the observation has concluded.

    :param n_rfsoc: the number of RFSOCs that will be merged together
    :param n_sample: the number of data samples collected
    :param n_sample_lo: the number of data samples collected during the LO sweep prior to the observation
    :param n_resonator:  the total number of resonances spanning all RFSOCs
    :param n_attenuator: the number of attenuators on each RFSOC -
    """

    def __init__(self, path, overwrite=False):
        log = logger.getChild(__name__)

        self.filename = path
        if not os.path.exists(path):
            log.debug("File not found, creating file...")
            try:
                self.fh = h5py.File(self.filename, "a")
            except Exception as e:
                log.error(e)
                raise
        else:
            log.debug("File already exists.")
            if overwrite:
                log.debug("Overwriting file")
                try:
                    self.fh = h5py.File(self.filename, "w")
                except Exception as e:
                    log.error(e)
                    raise
            else:
                log.debug("Opening File as append")
                try:
                    self.fh = h5py.File(self.filename, "a")
                except Exception as e:
                    log.error(e)
                    raise
                self.read()

    def format(
        self,
        n_rfsoc: int,
        n_sample: int,
        n_sample_lo: int,
        n_resonator: int,
        n_attenuator: str,
    ):
        # ***************************** Time Ordered data *************************************
        self.adc_i = self.df.create_dataset(
            "time_ordered_data/adc_i",
            (n_resonator, n_sample),
            dtype=h5py.h5t.NATIVE_UINT32,
            chunks=True,
        )
        self.adc_i.attrs.create("units", "volts")
        self.adc_i.attrs.create("dimension_names", "bin_number, packet_number")

        self.adc_q = self.df.create_dataset(
            "time_ordered_data/adc_i",
            (n_resonator, n_sample),
            dtype=h5py.h5t.NATIVE_UINT32,
            chunks=True,
        )
        self.adc_q.attrs.create("units", "volts")
        self.adc_q.attrs.create("dimension_names", "bin_number, packet_number")

        self.delta_freq = self.df.create_dataset(
            "time_ordered_data/delta_freq",
            (n_resonator, n_sample),
            dtype=h5py.h5t.NATIVE_DOUBLE,
        )

        self.delta_oq = self.df.create_dataset(
            "time_ordered_data/delta_oq",
            (n_resonator, n_sample),
            dtype=h5py.h5t.NATIVE_DOUBLE,
        )

        self.detector_delta_azimuth = self.df.create_dataset(
            "time_ordered_data/detector_delta_azimuth",
            (n_resonator, n_sample),
            dtype=h5py.h5t.NATIVE_DOUBLE,
        )
        self.detector_delta_elevation = self.df.create_dataset(
            "time_ordered_data/detector_delta_elevation",
            (n_resonator, n_sample),
            dtype=h5py.h5t.NATIVE_DOUBLE,
        )
        self.lo_freq = self.df.create_dataset(
            "time_ordered_data/lo_freq",
            (1, n_sample),
            dtype=h5py.h5t.NATIVE_DOUBLE,
        )
        self.telescope_azimuth = self.df.create_dataset(
            "time_ordered_data/telescope_azimuth",
            (1, n_sample),
            dtype=h5py.h5t.NATIVE_DOUBLE,
        )
        self.telescope_elevation = self.df.create_dataset(
            "time_ordered_data/telescope_elevation",
            (1, n_sample),
            dtype=h5py.h5t.NATIVE_DOUBLE,
        )
        self.telescope_elevation = self.df.create_dataset(
            "time_ordered_data/telescope_elevation",
            (1, n_sample),
            dtype=h5py.h5t.NATIVE_DOUBLE,
        )
        self.timestamp = self.df.create_dataset(
            "time_ordered_data/timestamp",
            (n_rfsoc, n_sample),
            dtype=h5py.h5t.NATIVE_UINT64,
        )
        # *********************************  GLOBAL DATA ****************************************

        self.attenuator_settings = self.df.create_dataset(
            "global_data/attenuator_settings",
            (n_rfsoc, n_attenuator),
            dtype=h5py.h5t.NATIVE_DOUBLE,
        )

        self.baseband_freq = self.df.create_dataset(
            "global_data/baseband_freq", (1, n_resonator), dtype=h5py.h5t.NATIVE_DOUBLE
        )

        self.channel_mask = self.df.create_dataset(
            "global_data/channel_mask", (1, n_resonator), dtype=h5py.h5t.NATIVE_DOUBLE
        )
        self.detector_delta_x = self.df.create_dataset(
            "global_data/detector_delta_x",
            (1, n_resonator),
            dtype=h5py.h5t.NATIVE_DOUBLE,
        )
        self.detector_delta_x = self.df.create_dataset(
            "global_data/detector_delta_x",
            (1, n_resonator),
            dtype=h5py.h5t.NATIVE_DOUBLE,
        )

        self.df_di = self.df.create_dataset(
            "global_data/df_di", (1, n_resonator), dtype=h5py.h5t.NATIVE_DOUBLE
        )
        self.df_dq = self.df.create_dataset(
            "global_data/df_dq", (1, n_resonator), dtype=h5py.h5t.NATIVE_DOUBLE
        )
        self.df_dt = self.df.create_dataset(
            "global_data/df_dt", (1, n_resonator), dtype=h5py.h5t.NATIVE_DOUBLE
        )
        self.doq_di = self.df.create_dataset(
            "global_data/doq_di", (1, n_resonator), dtype=h5py.h5t.NATIVE_DOUBLE
        )
        self.doq_df = self.df.create_dataset(
            "global_data/doq_df", (1, n_resonator), dtype=h5py.h5t.NATIVE_DOUBLE
        )
        self.doq_dt = self.df.create_dataset(
            "global_data/doq_dt", (1, n_resonator), dtype=h5py.h5t.NATIVE_DOUBLE
        )
        self.lo_sweep_adc_i = self.df.create_dataset(
            "global_data/lo_sweep_adc_i",
            (n_resonator, n_sample_lo),
            dtype=h5py.h5t.NATIVE_DOUBLE,
        )
        self.lo_sweep_adc_q = self.df.create_dataset(
            "global_data/lo_sweep_adc_q",
            (n_resonator, n_sample_lo),
            dtype=h5py.h5t.NATIVE_DOUBLE,
        )
        self.lo_sweep_baseband_freq = self.df.create_dataset(
            "global_data/lo_sweep_baseband_freq",
            (1, n_resonator),
            dtype=h5py.h5t.NATIVE_DOUBLE,
        )
        self.lo_sweep_freq = self.df.create_dataset(
            "global_data/lo_sweep_freq", (1, n_sample_lo), dtype=h5py.h5t.NATIVE_DOUBLE
        )
        self.sample_rate = self.df.create_dataset(
            "global_data/sample_rate", (1,), dtype=h5py.h5t.NATIVE_DOUBLE
        )
        self.tile_number = self.df.create_dataset(
            "global_data/tile_number", (1, n_resonator), dtype=h5py.h5t.NATIVE_INT32
        )
        self.tone_power = self.df.create_dataset(
            "global_data/tone_power", (1, n_resonator), dtype=h5py.h5t.NATIVE_INT32
        )
        self.rfsoc_number = self.df.create_dataset(
            "global_data/rfsoc_number", (1, n_resonator), dtype=h5py.h5t.NATIVE_UINT32
        )
        self.ifslice_number = self.df.create_dataset(
            "global_data/ifslice_number", (1, n_resonator), dtype=h5py.h5t.NATIVE_UINT32
        )
        # ********************************** Dimensions ****************************************
        # Dimensions
        self.dim_n_rfsoc = self.df.create_dataset(
            "Dimension/n_rfsoc",
            (1,),
            dtype=h5py.h5t.NATIVE_UINT64,
        )
        self.dim_n_sample = self.df.create_dataset(
            "Dimension/n_sample",
            (1,),
            dtype=h5py.h5t.NATIVE_UINT64,
        )
        self.dim_n_sample_lo = self.df.create_dataset(
            "Dimension/n_sample_lo",
            (1,),
            dtype=h5py.h5t.NATIVE_UINT64,
        )
        self.dim_n_resonator = self.df.create_dataset(
            "Dimension/n_resonator",
            (1,),
            dtype=h5py.h5t.NATIVE_UINT64,
        )
        self.dim_n_attenuator = self.df.create_dataset(
            "Dimension/n_attenuator",
            (1,),
            dtype=h5py.h5t.NATIVE_UINT64,
        )


def gen_read(h5: str):
    """
    Cheat function. Reads an hdf5 file and generates read statements to then be pasted back into
    the class. This method worked because the dataset name was the same as the instance variable name.
    ie.  /group/group/adc_i translates to self.fh.adc_i
    """
    f = h5py.File(h5, "r")

    def somefunc(name, object):
        if isinstance(object, h5py.Dataset):
            prop = object.name.split("/").pop()
            print(f"self.{prop} = self.fh['{object.name}']")

    for k, v in f.items():
        if isinstance(v, h5py.Dataset):
            print(f"self.{k} = self.fh['{k}']")
        elif isinstance(v, h5py.Group):
            v.visititems(somefunc)


def getdtime():
    """
    Gets the time in fractional hours since midnight
    of the current day with a precision down to seconds.

    :return: Fraction of hours since midnight

    :rtype: Float

    """
    t = time.localtime()
    t1 = time.mktime(t)  # current time

    t2 = time.struct_time(
        (
            t.tm_year,
            t.tm_mon,
            t.tm_mday,
            0,
            0,
            0,
            t.tm_wday,
            t.tm_yday,
            t.tm_isdst,
            t.tm_zone,
            t.tm_gmtoff,
        )
    )
    t2 = time.mktime(t2)
    return (t1 - t2) / 3600


def get_yymmdd():
    """
    Duplicate from onrkidpy
    """
    # get today's date string
    yy = "{}".format(date.today().year)
    mm = "{}".format(date.today().month)
    if date.today().month < 10:
        mm = "0" + mm
    dd = "{}".format(date.today().day)
    if date.today().day < 10:
        dd = "0" + dd
    yymmdd = yy + mm + dd
    return yymmdd


def get_last_lo(name: str):
    """
    Modified function to get the laster sweep file from data.
    this function expects a general file format consisting of the
    following.

    .. code::

        "/data/{yymmdd}/{yymmdd}_{name}_LO_Sweep_*.npy"
        example.
        /data/20230730/20230730_rfsoc1_LO_Sweep_hour15p4622.npy
        /data/20230730/20230730_rfsoc1_LO_Sweep_hour15p4625.npy
        /data/20230730/20230730_rfsoc1_LO_Sweep_hour15p4628.npy
    """
    # see if we already have the parent folder for today's date
    yymmdd = get_yymmdd()
    date_folder = "/data/" + yymmdd + "/"
    check_date_folder = glob.glob(date_folder)
    if np.size(check_date_folder) == 0:
        return ""

    fstring = f"/data/{yymmdd}/{yymmdd}_{name}_LO_Sweep_*.npy"
    g = glob.glob(fstring)

    if len(g) == 0:
        return ""

    g.sort()
    return g[-1]


def get_last_rdf(name: str):
    """
    Modified function to get the latest RawDataFile
    following.

    .. code::

        "/data/{yymmdd}/{yymmdd}_{name}_TOD_set*.hd5"
        example.
        /data/20230731/20230731_rfsoc1_TOD_set1001.hd5
        /data/20230731/20230731_rfsoc1_TOD_set1002.hd5

    """
    # see if we already have the parent folder for today's date
    yymmdd = get_yymmdd()
    date_folder = "/data/" + yymmdd + "/"
    check_date_folder = glob.glob(date_folder)
    if np.size(check_date_folder) == 0:
        return ""

    fstring = f"/data/{yymmdd}/{yymmdd}_{name}_TOD_set*.hd5"
    g = glob.glob(fstring)

    if len(g) == 0:
        return ""

    g.sort()
    return g[-1]


# 20230731_TOD_set1002
if __name__ == "__main__":
    pass
