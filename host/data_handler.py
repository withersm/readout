"""
Overview
________

data_handler defines an interface with whichi readout data is obtained, stored, and manipulated.
The format specified here is a standard 

:Authors: - Cody
          - Jack Sayers
          - Daniel Cunnane
:Date: 2023-07-26
:Version: 1.0.0
"""

import h5py
import os
import datetime
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RFChannel:
    raw_filename: str = ""  # HDF5 file to open/create
    ip: str = ""  # udp ip to listen on
    port: int = 0000  # udp port to listen on
    name: str = ""  # Friendly Name to call the channel
    n_sample: int = 488
    n_resonator: int = 1024
    n_attenuator: int = 1


class RawDataFile:
    """A raw hdf5 data file object for incoming rfsoc-UDP data streams.

    UDP packets containing our downsampled data streaming from the RFSOC to the readout computer
    will be captured and saved to this hdf5 filetype.

    :param path: /file/path/here/file.h5
    :param n_sample: The number of data samples collected (i.e., sample_rate * length_of_file_in_seconds)

    :param n_resonator: The number of resonance tones
    :param n_attenuator: The number of attenuators in the IF slice or elsewhere in the system - likely = 2 for now
    """

    def __init__(self, path):
        log = logger.getChild(__name__)

        self.filename = path
        if not os.path.isfile(path):
            log.debug("File not found, creating file...")
            try:
                self.fh = h5py.File(self.filename, "a")
            except Exception as e:
                log.error(e)
                raise
        else:
            log.debug("File exists, attempting to load")
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
            "Dimension/n_sample",
            (1,),
            dtype=h5py.h5t.NATIVE_UINT64,
        )
        self.n_resonator = self.fh.create_dataset(
            "Dimension/n_resonator",
            (1,),
            dtype=h5py.h5t.NATIVE_UINT64,
        )
        self.n_attenuator = self.fh.create_dataset(
            "Dimension/n_attenuator",
            (1,),
            dtype=h5py.h5t.NATIVE_UINT64,
        )
        # ******************************** Global Data ******************************
        self.attenuator_settings = self.fh.create_dataset(
            "global_data/attenuator_settings",
            (1, n_attenuator),
            dtype=h5py.h5t.NATIVE_DOUBLE,
        )
        self.baseband_freq = self.fh.create_dataset(
            "global_data/baseband_freq", (1, n_resonator), dtype=h5py.h5t.NATIVE_DOUBLE
        )
        self.sample_rate = self.fh.create_dataset(
            "global_data/sample_rate", (1,), dtype=h5py.h5t.NATIVE_DOUBLE
        )
        self.tile_number = self.fh.create_dataset(
            "global_data/tile_number", (1, n_resonator), dtype=h5py.h5t.NATIVE_INT32
        )
        self.tone_power = self.fh.create_dataset(
            "global_data/tone_power", (1, n_resonator), dtype=h5py.h5t.NATIVE_DOUBLE
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
            (n_resonator, n_sample),
            dtype=h5py.h5t.NATIVE_UINT32,
        )
        self.adc_q = self.fh.create_dataset(
            "time_ordered_data/adc_q",
            (n_resonator, n_sample),
            dtype=h5py.h5t.NATIVE_UINT32,
        )
        self.lo_freq = self.fh.create_dataset(
            "time_ordered_data/lo_freq", (1, n_sample), dtype=h5py.h5t.NATIVE_INT32
        )
        timestamp_compound_datatype = [
            ("time_us", h5py.h5t.NATIVE_UINT64),
            ("packet_number", h5py.h5t.NATIVE_UINT64),
        ]
        self.timestamp = self.fh.create_dataset(
            "time_ordered_data/timestamp",
            (2, n_sample),
            dtype=timestamp_compound_datatype,
        )
        self.fh.flush()

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
            self.n_attenuator = self.fh["/Dimension/n_attenuator"]
            self.n_resonator = self.fh["/Dimension/n_resonator"]
            self.n_sample = self.fh["/Dimension/n_sample"]
            self.attenuator_settings = self.fh["/global_data/attenuator_settings"]
            self.baseband_freq = self.fh["/global_data/baseband_freq"]
            self.chan_number = self.fh["/global_data/chan_number"]
            self.ifslice_number = self.fh["/global_data/ifslice_number"]
            self.rfsoc_number = self.fh["/global_data/rfsoc_number"]
            self.sample_rate = self.fh["/global_data/sample_rate"]
            self.tile_number = self.fh["/global_data/tile_number"]
            self.tone_power = self.fh["/global_data/tone_power"]
            self.adc_i = self.fh["/time_ordered_data/adc_i"]
            self.adc_q = self.fh["/time_ordered_data/adc_q"]
            self.lo_freq = self.fh["/time_ordered_data/lo_freq"]
            self.timestamp = self.fh["/time_ordered_data/timestamp"]
        except Exception as e:
            log.error(e)

    def close(self):
        self.fh.close()


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

    def __init__(
        self,
        n_rfsoc: int,
        n_sample: int,
        n_sample_lo: int,
        n_resonator: int,
        n_attenuator: str,
        filename,
    ):
        self.df = h5py.File(filename, "w")

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

    that is /group/group/adc_i translates to self.fh.adc_i
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

