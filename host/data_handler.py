"""
@author: Cody Roberson
@date: 02/13/2023
@file: data_handler
@copyright: To Be Determined in the Interest of Collaboration
@description:
    DataHandler takes care of generating our hdf5 ovservation data files


@Revisions:

@Dev Notes:
    02/17/2023
    HDF5 arrays are (row,col), scalars are (1,)

"""
import logging
import multiprocessing

import numpy as np
from time import sleep
import h5py
import multiprocessing as mproc


class RawDataFile:
    """A raw hdf5 data file object for incoming rfsoc-UDP data streams.

    udp packets containing our downsampled data streaming from the RFSOC to the readout computer
    will be captured and saved to this hdf5 filetype.

    Dimmensions:
        - **n_sample:** the number of data samples collected (i.e., sample_rate * length_of_file_in_seconds)
        - **n_resonator:** the number of resonance tones
        - **n_attenuator:** the number of attenuators in the IF slice or elsewhere in the system - likely = 2 for now

    :param path: /file/path/here/file.h5
    :param n_sample:
    :param n_resonator:
    :param n_attenuator:
    """

    def __init__(self, path: str, n_sample: int, n_resonator: int, n_attenuator: int):
        """"""
        self.filename = path
        self.fh = None
        try:
            self.fh = h5py.File(filename, "w")
        except IOError:
            print(
                "FATA: Could not create hdf5 file, please check the path exists and"
                "that read/write permissions are allowed"
            )
            raise
        except Exception as except_rdf:  # TODO: REFINE EXCEPTION CASES
            raise except_rdf
        # ********************* DATAFIELD ***********************

        # Global Data
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
        self.ifslice_number = self.fh.create_dataset(
            "global_data/ifslice_number", (1,), dtype=h5py.h5t.NATIVE_INT32
        )

        # Time Ordered Data
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
            ("time_ms", h5py.h5t.NATIVE_UINT64),
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
    """The main Observation datafile which will encompass the data gatherd during an observation
    period. raw rfsoc data is migrated into this data file after the observation has concluded.
    """

    def __init__(
        self,
        n_rfsoc: int,
        n_sample: int,
        n_sample_lo: int,
        n_resonator: int,
        n_attenuator: str,
    ):
        """
        Initialize the DataFile object. Captures given parameters in to instance variables
        and passes them to internal file template method

        n_rfsoc:
        n_sample:
        n_sample_lo:
        n_resonator:
        n_attenuator:
            i.e., not the total number of attenuators spanning all RFSOCs
        :param n_rfsoc: the number of RFSOCs that will be merged together
        :param n_sample: the number of data samples collected
        :param n_sample_lo: the number of data samples collected during the LO sweep prior to the observation
        :param n_resonator:  the total number of resonances spanning all RFSOCs
        :param n_attenuator: the number of attenuators on each RFSOC -
        """

        DETECTOR_DATA_LENGTH = 2052  # CONST

        self.df = h5py.File(filename, "w")

        # Time Ordered data

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
        # ****************  GLOBAL DATA ***********************

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


class DataHandler:
    def __process_data_handoff(
        queue: multiprocessing.queues.Queue, filename: str, nPackets: int
    ):
        """
        This process should run independantly to any sort of data capture processes
        since disk io-ops are slow. The operation is as follows:

            while running in a seperate process,
                check for items being placed in the queue,
                    if the queue contains object, begin copying data out of the queue and into
                    the relevant hdf5 datasets.
        :param queue: Multiprocessing queue containing our data relayed from the depacketizer
        :param filename: (DEPRECATED)
        :param nPackets:
        :return:
        """
        dFile = h5py.File(filename, "w")
        data = dFile.create_dataset(
            "PACKETS",
            (2052, nPackets),
            dtype=h5py.h5t.NATIVE_INT32,
            chunks=True,
            maxshape=(None, None),
        )
        active = True
        while active:
            rawData = queue.get()
            if rawData is not None:
                d, c = rawData
                data[:, c] = d
            else:
                active = False
                dFile.flush()
        dFile.close()


def DoTestRoutine():
    pass


if __name__ == "__main__":
    DoTestRoutine()
