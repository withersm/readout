"""
:author: Cody Roberson
:date: 02/13/2023
:file: data_handler
:copyright: To Be Determined in the Interest of Collaboration
:description: DataHandler takes care of generating our hdf5 ovservation data files

02/17/2023:
    HDF5 arrays are (row,col), scalars are (1,)
02/21/2023
    while hdf datasets can have infinite dimensions, they do need to be resized:
        dataset (1,4) to store [0,1,2,3]
    then dataset.resize((2,4))
        dataset (2,4) to store [0,1,2,3], [4,8,12,16] becomes possible
    While it would be nice to dynamically allocate n_samples as needed, the hdf5 lib implementation
    makes a call to emalloc which leads to speculation that a mem copy takes place and the same inefficiencies
    of array copies takes place under the hood. This is problematic for large, efficient, datasets.
04/11/23

    Data organization:
----------
"""

import h5py
import os
import datetime

class RawDataFile:
    """A raw hdf5 data file object for incoming rfsoc-UDP data streams.

    UDP packets containing our downsampled data streaming from the RFSOC to the readout computer
    will be captured and saved to this hdf5 filetype.

    :param path: /file/path/here/file.h5
    :param n_sample: The number of data samples collected (i.e., sample_rate * length_of_file_in_seconds)

    :param n_resonator: The number of resonance tones
    :param n_attenuator: The number of attenuators in the IF slice or elsewhere in the system - likely = 2 for now
    """

    def __init__(self, path: str, n_sample: int, n_resonator: int, n_attenuator: int):
        self.filename = path
        try:
            self.fh = h5py.File(self.filename, "w")
        except IOError:
            print(
                "FATA: Could not create hdf5 file, please check the path exists and"
                "that read/write permissions are allowed"
            )
            raise
        except Exception as except_rdf:  # TODO: REFINE EXCEPTION CASES
            raise except_rdf

        # ********************************* Dimensions *******************************
        self.dim_n_sample = self.fh.create_dataset(
            "Dimension/n_sample",
            (1,),
            dtype=h5py.h5t.NATIVE_UINT64,
        )
        self.dim_n_resonator = self.fh.create_dataset(
            "Dimension/n_resonator",
            (1,),
            dtype=h5py.h5t.NATIVE_UINT64,
        )
        self.dim_n_attenuator = self.fh.create_dataset(
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
            "global_data/sample_Rate", (1,), dtype=h5py.h5t.NATIVE_DOUBLE
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


class DataHandler:
    """
    Used to handle the interaction between obtaining a packet and
    subsequently storing it.

    Users shall use a DataHandler class to take some amount of packets
    """

    def __init__(self):
        self.rawdataset = []

        # Polulated int he create_odc
        self.raw_data_folder = None
        self.housekeeping_folder = None
        self.plots_folder = None
        self.raw_data_dirfile = None
        self.obs_df = None  # observation file
        self.raw_dfs = []  # raw data files

    def create_odh(self, dataroot: str):
        """
        Creates an Observation data collection folder with the structure specified in the
        Implementation and Dataformat Design Document

        :param dataroot:
        :return:
        """
        # create folder structure:
        self.data_root_folder = (
            dataroot
            + "/observation_"
            + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        dr = self.data_root_folder

        self.raw_data_folder = dr + "/raw_data/"
        self.housekeeping_folder = dr + "/housekeeping/"
        self.plots_folder = dr + "/plots/"

        try:
            os.mkdir(self.data_root_folder)
            os.mkdir(self.raw_data_folder)
            os.mkdir(self.housekeeping_folder)
            os.mkdir(self.plots_folder)
        except IOError:
            print("Failed to write directory")
            raise
        


    def extend_datalen(self):
        """
        Extends the available dataspace in *ALL* dof the data files
        :return:
        """

    def record_data(self, n_samples):
        """
        record a specified number of data samples to hdf5 document
        :param n_samples:
        :return:
        """
        pass

    def finalize(self, rawdf: list[RawDataFile], obs_df: ObservationDataFile):
        """
        Merges the raw data files collected during an observation period into the Observation DataFile

        Interest:
        # GD
        attenuator_settings

        # TOD
        adc_i: n_resonator x n_sample
        adc_q: n_resonator x n_sample
        lo_freq: n_sample
        timestamp: n_sample

        :param rawdf:
        :param obs_df:
        :return:
        """
        pass

