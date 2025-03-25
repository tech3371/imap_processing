"""
Perform CoDICE l1a processing.

This module processes CoDICE L0 files and creates L1a data products.

Notes
-----
    from imap_processing.codice.codice_l1a import process_codice_l1a
    processed_datasets = process_codice_l1a(path_to_l0_file)
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.codice import constants
from imap_processing.codice.codice_l0 import decom_packets
from imap_processing.codice.decompress import decompress
from imap_processing.codice.utils import CODICEAPID
from imap_processing.spice.time import met_to_ttj2000ns

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# TODO: Determine what should go in event data CDF and how it should be
#       structured.


class CoDICEL1aPipeline:
    """
    Contains methods for processing L0 data and creating L1a data products.

    Parameters
    ----------
    table_id : int
        A unique ID assigned to a specific table configuration. This field is
        used to link the overall acquisition and processing settings to a
        specific table configuration.
    plan_id : int
        The plan table that was in use.  In conjunction with ``plan_step``,
        describes which counters are included in the data packet.
    plan_step : int
        Plan step that was active when the data was acquired and processed. In
        conjunction with ``plan_id``, describes which counters are included
        in the data packet.
    view_id : int
        Provides information about how data was collapsed and/or compressed.

    Methods
    -------
    calculate_epoch_values()
        Calculate and return the values to be used for `epoch`.
    decompress_data(science_values)
        Perform decompression on the data.
    define_coordinates()
        Create ``xr.DataArrays`` for the coords needed in the final dataset.
    define_data_variables()
        Define and add the appropriate data variables to the dataset.
    define_dimensions()
        Define the dimensions of the data arrays for the final dataset.
    define_support_variables()
        Define and add 'support' CDF data variables to the dataset.
    get_acquisition_times()
        Retrieve the acquisition times via the Lo stepping table.
    get_energy_table()
        Retrieve the ESA sweep values.
    get_hi_energy_table_data(species)
        Retrieve energy table data for CoDICE-Hi products
    reshape_data()
        Reshape the data arrays based on the data product being made.
    set_data_product_config()
        Set the various settings for defining the data products.
    """

    def __init__(self, table_id: int, plan_id: int, plan_step: int, view_id: int):
        """Initialize a ``CoDICEL1aPipeline`` class instance."""
        self.table_id = table_id
        self.plan_id = plan_id
        self.plan_step = plan_step
        self.view_id = view_id

    def calculate_epoch_values(self) -> NDArray[int]:
        """
        Calculate and return the values to be used for `epoch`.

        On CoDICE, the epoch values are derived from the `acq_start_seconds` and
        `acq_start_subseconds` fields in the packet. Note that the
        `acq_start_subseconds` field needs to be converted from microseconds to
        seconds

        Returns
        -------
        epoch : NDArray[int]
            List of epoch values.
        """
        epoch = met_to_ttj2000ns(
            self.dataset.acq_start_seconds + self.dataset.acq_start_subseconds / 1e6
        )

        return epoch

    def decompress_data(self, science_values: list[str]) -> None:
        """
        Perform decompression on the data.

        The science data within the packet is a compressed byte string of
        values. Apply the appropriate decompression algorithm to get an array
        of decompressed values.

        Parameters
        ----------
        science_values : list[str]
            A list of byte strings representing the science values of the data
            for each packet.
        """
        # The compression algorithm depends on the instrument and view ID
        if self.config["instrument"] == "lo":
            compression_algorithm = constants.LO_COMPRESSION_ID_LOOKUP[self.view_id]
        elif self.config["instrument"] == "hi":
            compression_algorithm = constants.HI_COMPRESSION_ID_LOOKUP[self.view_id]

        self.raw_data = []
        for packet_data, byte_count in zip(
            science_values, self.dataset.byte_count.data
        ):
            # Convert from numpy array to byte object
            values = ast.literal_eval(str(packet_data))

            # Only use the values up to the byte count. Bytes after this are
            # used as padding and are not needed
            values = values[:byte_count]

            decompressed_values = decompress(values, compression_algorithm)
            self.raw_data.append(decompressed_values)

    def define_coordinates(self) -> None:
        """
        Create ``xr.DataArrays`` for the coords needed in the final dataset.

        The coordinates for the dataset depend on the data product being made.
        """
        self.coords = {}

        coord_names = ["epoch", *list(self.config["output_dims"].keys())]

        # These are labels unique to lo-counters products coordinates
        if self.config["dataset_name"] in [
            "imap_codice_l1a_lo-counters-aggregated",
            "imap_codice_l1a_lo-counters-singles",
        ]:
            coord_names.append("spin_sector_pairs_label")

        # Define the values for the coordinates
        for name in coord_names:
            if name == "epoch":
                values = self.calculate_epoch_values()
            elif name in [
                "esa_step",
                "inst_az",
                "spin_sector",
                "spin_sector_pairs",
                "spin_sector_index",
                "ssd_index",
            ]:
                values = np.arange(self.config["output_dims"][name])
            elif name == "spin_sector_pairs_label":
                values = np.array(
                    [
                        "0-30 deg",
                        "30-60 deg",
                        "60-90 deg",
                        "90-120 deg",
                        "120-150 deg",
                        "150-180 deg",
                    ]
                )

            coord = xr.DataArray(
                values,
                name=name,
                dims=[name],
                attrs=self.cdf_attrs.get_variable_attributes(name),
            )

            self.coords[name] = coord

    def define_data_variables(self) -> xr.Dataset:
        """
        Define and add the appropriate data variables to the dataset.

        The data variables included in the dataset depend on the data product
        being made. The method returns the ``xarray.Dataset`` object that can
        then be written to a CDF file.

        Returns
        -------
        processed_dataset : xarray.Dataset
            The 'final' ``xarray`` dataset.
        """
        # Create the main dataset to hold all the variables
        dataset = xr.Dataset(
            coords=self.coords,
            attrs=self.cdf_attrs.get_global_attributes(self.config["dataset_name"]),
        )

        # Stack the data so that it is easier to reshape and iterate over
        all_data = np.stack(self.data)

        # The dimension of all_data is something like (epoch, num_counters,
        # num_energy_steps, num_positions, num_spin_sectors) (or may be slightly
        # different depending on the data product). In any case, iterate over
        # the num_counters dimension to isolate the data for each counter so
        # each counter's data can be placed in a separate CDF data variable.
        for counter, variable_name in zip(
            range(all_data.shape[1]), self.config["variable_names"]
        ):
            # Extract the counter data
            counter_data = all_data[:, counter, ...]

            # Get the CDF attributes
            descriptor = self.config["dataset_name"].split("imap_codice_l1a_")[-1]
            cdf_attrs_key = f"{descriptor}-{variable_name}"
            attrs = self.cdf_attrs.get_variable_attributes(cdf_attrs_key)

            # For most products, the final CDF dimensions always has "epoch" as
            # the first dimension followed by the dimensions for the specific
            # data product
            dims = ["epoch", *list(self.config["output_dims"].keys())]

            # However, CoDICE-Hi products use specific energy bins for the
            # energy dimension
            # TODO: This will be expanded to all CoDICE-Hi products once I
            #       can validate them. For now, just operate on hi-sectored
            if self.config["dataset_name"] == "imap_codice_l1a_hi-sectored":
                dims = [
                    f"energy_{variable_name}" if item == "esa_step" else item
                    for item in dims
                ]

            # Create the CDF data variable
            dataset[variable_name] = xr.DataArray(
                counter_data,
                name=variable_name,
                dims=dims,
                attrs=attrs,
            )

        # Add support data variables based on data product
        dataset = self.define_support_variables(dataset)

        # For CoDICE-Hi products, since energy dimension was replaced, we no
        # longer need the "esa_step" coordinate
        # TODO: This will be expanded to all CoDICE-Hi products once I
        #       can validate them. For now, just operate on hi-sectored
        if self.config["dataset_name"] == "imap_codice_l1a_hi-sectored":
            dataset = dataset.drop_vars("esa_step")

        return dataset

    def define_support_variables(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Define and add 'support' CDF data variables to the dataset.

        These variables include instrument metadata, energies, times, etc. that
        help further define the L1a CDF data product. The variables included
        depend on the data product being made.

        Parameters
        ----------
        dataset : xarray.Dataset
            ``xarray`` dataset for the data product.

        Returns
        -------
        dataset : xarray.Dataset
            ``xarray`` dataset for the data product, with added support variables.
        """
        # These variables can be gathered from the packet data
        packet_data_variables = [
            "rgfo_half_spin",
            "nso_half_spin",
            "sw_bias_gain_mode",
            "st_bias_gain_mode",
        ]

        hi_energy_table_variables = [
            "energy_h",
            "energy_he3",
            "energy_he4",
            "energy_c",
            "energy_o",
            "energy_ne_mg_si",
            "energy_fe",
            "energy_uh",
            "energy_junk",
            "energy_he3he4",
            "energy_cno",
        ]

        for variable_name in self.config["support_variables"]:
            # CoDICE-Hi energy tables are treated differently because values
            # are binned and we need to record the energies _and_ their deltas
            if variable_name in hi_energy_table_variables:
                centers, deltas = self.get_hi_energy_table_data(
                    variable_name.split("energy_")[-1]
                )

                # Add bin centers and deltas to the dataset
                dataset[variable_name] = xr.DataArray(
                    centers,
                    dims=[variable_name],
                    attrs=self.cdf_attrs.get_variable_attributes(
                        f"{self.config['dataset_name'].split('_')[-1]}-{variable_name}"
                    ),
                )
                dataset[f"{variable_name}_delta"] = xr.DataArray(
                    deltas,
                    dims=[f"{variable_name}_delta"],
                    attrs=self.cdf_attrs.get_variable_attributes(
                        f"{self.config['dataset_name'].split('_')[-1]}-{variable_name}_delta"
                    ),
                )

            # Otherwise, support variable data can be gathered from nominal
            # lookup tables or packet data
            else:
                # These variables require reading in external tables
                if variable_name == "energy_table":
                    variable_data = self.get_energy_table()
                    dims = ["esa_step"]
                    attrs = self.cdf_attrs.get_variable_attributes("energy_table")

                elif variable_name == "acquisition_time_per_step":
                    variable_data = self.get_acquisition_times()
                    dims = ["esa_step"]
                    attrs = self.cdf_attrs.get_variable_attributes(
                        "acquisition_time_per_step"
                    )

                # These variables can be gathered straight from the packet data
                elif variable_name in packet_data_variables:
                    variable_data = self.dataset[variable_name].data
                    dims = ["epoch"]
                    attrs = self.cdf_attrs.get_variable_attributes(variable_name)

                # Data quality is named differently in packet data and needs to be
                # treated slightly differently
                elif variable_name == "data_quality":
                    variable_data = self.dataset.suspect.data
                    dims = ["epoch"]
                    attrs = self.cdf_attrs.get_variable_attributes("data_quality")

                # Spin period requires the application of a conversion factor
                # See Table B.5 in the algorithm document
                elif variable_name == "spin_period":
                    variable_data = (
                        self.dataset.spin_period.data * constants.SPIN_PERIOD_CONVERSION
                    ).astype(np.float32)
                    dims = ["epoch"]
                    attrs = self.cdf_attrs.get_variable_attributes("spin_period")

                # Add variable to the dataset
                dataset[variable_name] = xr.DataArray(
                    variable_data,
                    dims=dims,
                    attrs=attrs,
                )

        return dataset

    def get_acquisition_times(self) -> list[float]:
        """
        Retrieve the acquisition times via the Lo stepping table.

        Get the acquisition times from the lookup table based on the values of
        ``plan_id`` and ``plan_step``

        The Lo stepping table defines how many voltage steps and which steps are
        used during each spacecraft spin. A full cycle takes 16 spins. The table
        provides the "acquisition time", which is the acquisition time, in
        milliseconds, for the energy step.

        Returns
        -------
        acquisition_times : list[float]
            The list of acquisition times from the Lo stepping table.
        """
        # Determine which Lo stepping table is needed
        lo_stepping_table_id = constants.LO_STEPPING_TABLE_ID_LOOKUP[
            (self.plan_id, self.plan_step)
        ]

        acquisition_times: list[float] = constants.ACQUISITION_TIMES[
            lo_stepping_table_id
        ]

        return acquisition_times

    def get_energy_table(self) -> NDArray[float]:
        """
        Retrieve the ESA sweep values.

        Get the ElectroStatic Analyzer (ESA) sweep values from the data file
        based on the values of ``plan_id`` and ``plan_step``

        CoDICE-Lo measures ions between ~0.5 and 80 keV/q that enter the
        aperture and are selected and focused according to their E/q into the
        Time of Flight (TOF) assembly.  The E/q sweeping steps up to the max
        voltage for the next stepping cycle when solar wind count rate exceed a
        predefined threshold rate.

        The ESA sweep table defines the voltage steps that are used to cover the
        full energy per charge range.

        Returns
        -------
        energy_table : NDArray[float]
            The list of ESA sweep values (i.e. voltage steps).
        """
        # Read in the ESA sweep data table
        esa_sweep_data_file = Path(
            f"{imap_module_directory}/codice/data/esa_sweep_values.csv"
        )
        sweep_data = pd.read_csv(esa_sweep_data_file)

        # Determine which ESA sweep table is needed
        sweep_table_id = constants.ESA_SWEEP_TABLE_ID_LOOKUP[
            (self.plan_id, self.plan_step)
        ]

        # Get the appropriate values
        sweep_table = sweep_data[sweep_data["table_idx"] == sweep_table_id]
        energy_table: NDArray[float] = sweep_table["esa_v"].values

        return energy_table

    def get_hi_energy_table_data(
        self, species: str
    ) -> tuple[NDArray[float], NDArray[float]]:
        """
        Retrieve energy table data for CoDICE-Hi products.

        This includes the centers and deltas of the energy bins for a given
        species. These data eventually get included in the CoDICE-Hi CDF data
        products.

        Parameters
        ----------
        species : str
            The species of interest, which determines which lookup table to
            use (e.g. ``h``).

        Returns
        -------
        centers : NDArray[float]
            An array whose values represent the centers of the energy bins.
        deltas : NDArray[float]
            An array whose values represent the deltas of the energy bins.
        """
        data_product = self.config["dataset_name"].split("-")[-1].upper()
        energy_table = getattr(constants, f"{data_product}_ENERGY_TABLE")[species]

        # Find the centers and deltas of the energy bins
        centers = np.array(
            [
                (energy_table[i] + energy_table[i + 1]) / 2
                for i in range(len(energy_table) - 1)
            ]
        )
        deltas = energy_table[1:] - centers

        return centers, deltas

    def reshape_data(self) -> None:
        """
        Reshape the data arrays based on the data product being made.

        These data need to be divided up by species or priorities (or
        what I am calling "counters" as a general term), and re-arranged into
        4D arrays representing dimensions such as time, spin sectors, positions,
        and energies (depending on the data product).

        However, the existence and order of these dimensions can vary depending
        on the specific data product, so we define this in the "input_dims"
        and "output_dims" values configuration dictionary; the "input_dims"
        defines how the dimensions are written into the packet data, while
        "output_dims" defines how the dimensions should be written to the final
        CDF product.
        """
        # This will contain the reshaped data for all counters
        self.data = []

        # First reshape the data based on how it is written to the data array of
        # the packet data. The number of counters is the first dimension / axis.
        reshape_dims = (
            self.config["num_counters"],
            *self.config["input_dims"].values(),
        )

        # Then, transpose the data based on how the dimensions should be written
        # to the CDF file. Since this is specific to each data product, we need
        # to determine this dynamically based on the "output_dims" config.
        input_keys = ["num_counters", *self.config["input_dims"].keys()]
        output_keys = ["num_counters", *self.config["output_dims"].keys()]
        transpose_axes = [input_keys.index(dim) for dim in output_keys]

        for packet_data in self.raw_data:
            reshaped_packet_data = np.array(packet_data, dtype=np.uint32).reshape(
                reshape_dims
            )
            reshaped_cdf_data = np.transpose(reshaped_packet_data, axes=transpose_axes)

            self.data.append(reshaped_cdf_data)

        # No longer need to keep the raw data around
        del self.raw_data

    def set_data_product_config(
        self, apid: int, dataset: xr.Dataset, data_version: str
    ) -> None:
        """
        Set the various settings for defining the data products.

        Parameters
        ----------
        apid : int
            The APID of interest.
        dataset : xarray.Dataset
            The dataset for the APID of interest.
        data_version : str
            Version of the data product being created.
        """
        # Set the packet dataset so that it can be easily called from various
        # methods
        self.dataset = dataset

        # Set various configurations of the data product
        self.config: dict[str, Any] = constants.DATA_PRODUCT_CONFIGURATIONS.get(apid)  # type: ignore

        # Gather and set the CDF attributes
        self.cdf_attrs = ImapCdfAttributes()
        self.cdf_attrs.add_instrument_global_attrs("codice")
        self.cdf_attrs.add_instrument_variable_attrs("codice", "l1a")
        self.cdf_attrs.add_global_attribute("Data_version", data_version)


def create_event_dataset(
    apid: int, packet: xr.Dataset, data_version: str
) -> xr.Dataset:
    """
    Create dataset for event data.

    Parameters
    ----------
    apid : int
        The APID of the packet.
    packet : xarray.Dataset
        The packet to process.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    dataset : xarray.Dataset
        Xarray dataset containing the event data.
    """
    if apid == CODICEAPID.COD_LO_PHA:
        dataset_name = "imap_codice_l1a_lo-pha"
    elif apid == CODICEAPID.COD_HI_PHA:
        dataset_name = "imap_codice_l1a_hi-pha"

    # Extract the data
    # event_data = packet.event_data.data (Currently turned off, see TODO)

    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_global_attrs("codice")
    cdf_attrs.add_instrument_variable_attrs("codice", "l1a")
    cdf_attrs.add_global_attribute("Data_version", data_version)

    # Define coordinates
    epoch = xr.DataArray(
        packet.epoch,
        name="epoch",
        dims=["epoch"],
        attrs=cdf_attrs.get_variable_attributes("epoch"),
    )

    # Create the dataset to hold the data variables
    dataset = xr.Dataset(
        coords={
            "epoch": epoch,
        },
        attrs=cdf_attrs.get_global_attributes(dataset_name),
    )

    return dataset


def create_hskp_dataset(
    packet: xr.Dataset,
    data_version: str,
) -> xr.Dataset:
    """
    Create dataset for each metadata field for housekeeping data.

    Parameters
    ----------
    packet : xarray.Dataset
        The packet to process.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    dataset : xarray.Dataset
        Xarray dataset containing the metadata.
    """
    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_global_attrs("codice")
    cdf_attrs.add_instrument_variable_attrs("codice", "l1a")
    cdf_attrs.add_global_attribute("Data_version", data_version)

    epoch = xr.DataArray(
        packet.epoch,
        name="epoch",
        dims=["epoch"],
        attrs=cdf_attrs.get_variable_attributes("epoch"),
    )

    dataset = xr.Dataset(
        coords={"epoch": epoch},
        attrs=cdf_attrs.get_global_attributes("imap_codice_l1a_hskp"),
    )

    # These variables don't need to carry over from L0 to L1a
    exclude_variables = [
        "spare_1",
        "spare_2",
        "spare_3",
        "spare_4",
        "spare_5",
        "spare_6",
        "spare_62",
        "spare_68",
        "chksum",
    ]

    for variable in packet:
        if variable in exclude_variables:
            continue

        attrs = cdf_attrs.get_variable_attributes(variable)

        dataset[variable] = xr.DataArray(
            packet[variable].data, dims=["epoch"], attrs=attrs
        )

    return dataset


def get_params(dataset: xr.Dataset) -> tuple[int, int, int, int]:
    """
    Return the four 'main' parameters used for l1a processing.

    The combination of these parameters largely determines what steps/values
    are used to create CoDICE L1a data products and what steps are needed in
    the pipeline algorithm.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset for the APID of interest. We expect each packet in the
        dataset to have the same values for the four main parameters, so the
        first index of the dataset can be used to determine them.

    Returns
    -------
    table_id : int
        A unique ID assigned to a specific table configuration. This field is
        used to link the overall acquisition and processing settings to a
        specific table configuration.
    plan_id : int
        The plan table that was in use.  In conjunction with ``plan_step``,
        describes which counters are included in the data packet.
    plan_step : int
        Plan step that was active when the data was acquired and processed. In
        conjunction with ``plan_id``, describes which counters are included
        in the data packet.
    view_id : int
        Provides information about how data was collapsed and/or compressed.
    """
    table_id = int(dataset.table_id.data[0])
    plan_id = int(dataset.plan_id.data[0])
    plan_step = int(dataset.plan_step.data[0])
    view_id = int(dataset.view_id.data[0])

    return table_id, plan_id, plan_step, view_id


def log_dataset_info(datasets: dict[int, xr.Dataset]) -> None:
    """
    Log info about the input data to help with tracking and/or debugging.

    Parameters
    ----------
    datasets : dict[int, xarray.Dataset]
        Mapping from apid to ``xarray`` dataset, one dataset per apid.
    """
    launch_time = np.datetime64("2010-01-01T00:01:06.184", "ns")
    logger.info("\nThis input file contains the following APIDs:\n")
    for apid in datasets:
        num_packets = len(datasets[apid].epoch.data)
        time_deltas = [np.timedelta64(item, "ns") for item in datasets[apid].epoch.data]
        times = [launch_time + delta for delta in time_deltas]
        start = np.datetime_as_string(times[0])
        end = np.datetime_as_string(times[-1])
        logger.info(
            f"{CODICEAPID(apid).name}: {num_packets} packets spanning {start} to {end}"
        )


def process_codice_l1a(file_path: Path, data_version: str) -> list[xr.Dataset]:
    """
    Will process CoDICE l0 data to create l1a data products.

    Parameters
    ----------
    file_path : pathlib.Path | str
        Path to the CoDICE L0 file to process.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    processed_datasets : list[xarray.Dataset]
        A list of the ``xarray`` datasets containing the science data and
        supporting metadata.
    """
    # Decom the packets, group data by APID, and sort by time
    datasets = decom_packets(file_path)

    # Log some information about the contents of the data
    log_dataset_info(datasets)

    # Placeholder to hold the final, processed datasets
    processed_datasets = []

    # Process each APID separately
    for apid in datasets:
        dataset = datasets[apid]
        logger.info(f"\nProcessing {CODICEAPID(apid).name} packet")

        # Housekeeping data
        if apid == CODICEAPID.COD_NHK:
            processed_dataset = create_hskp_dataset(dataset, data_version)
            logger.info(f"\nFinal data product:\n{processed_dataset}\n")

        # Event data
        elif apid in [CODICEAPID.COD_LO_PHA, CODICEAPID.COD_HI_PHA]:
            processed_dataset = create_event_dataset(apid, dataset, data_version)
            logger.info(f"\nFinal data product:\n{processed_dataset}\n")

        # Everything else
        elif apid in constants.APIDS_FOR_SCIENCE_PROCESSING:
            # Extract the data
            science_values = [packet.data for packet in dataset.data]

            # Get the four "main" parameters for processing
            table_id, plan_id, plan_step, view_id = get_params(dataset)

            # Run the pipeline to create a dataset for the product
            pipeline = CoDICEL1aPipeline(table_id, plan_id, plan_step, view_id)
            pipeline.set_data_product_config(apid, dataset, data_version)
            pipeline.decompress_data(science_values)
            pipeline.reshape_data()
            pipeline.define_coordinates()
            processed_dataset = pipeline.define_data_variables()

            logger.info(f"\nFinal data product:\n{processed_dataset}\n")

        # TODO: Still need to implement I-ALiRT data products
        elif apid in [
            CODICEAPID.COD_HI_IAL,
            CODICEAPID.COD_LO_IAL,
        ]:
            logger.info("\tStill need to properly implement")
            processed_dataset = None

        # For APIDs that don't require processing
        else:
            logger.info(f"\t{apid} does not require processing")
            continue

        processed_datasets.append(processed_dataset)

    return processed_datasets
