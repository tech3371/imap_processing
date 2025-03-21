#!/usr/bin/env python3
# ruff: noqa: PLR0913
"""
Run the processing for a specific instrument & data level.

This module serves as a command line utility to invoke the processing for
a user-supplied instrument and data level.

Examples
--------
    imap_cli --instrument <instrument> --level <data_level>
"""

import argparse
import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import final
from urllib.error import HTTPError

import imap_data_access
import numpy as np
import xarray as xr
from imap_data_access.processing_input import (
    ProcessingInputCollection,
    ProcessingInputType,
)

import imap_processing
from imap_processing._version import __version__, __version_tuple__  # noqa: F401
from imap_processing.cdf.utils import load_cdf, write_cdf

# TODO: change how we import things and also folder
# structure may?
# From this:
#   from imap_processing.cdf.utils import write_cdf
# To this:
#   from imap_processing import cdf
# In code:
#   call cdf.utils.write_cdf
from imap_processing.codice import codice_l1a, codice_l1b
from imap_processing.glows.l1a.glows_l1a import glows_l1a
from imap_processing.glows.l1b.glows_l1b import glows_l1b
from imap_processing.glows.l2.glows_l2 import glows_l2
from imap_processing.hi.l1a import hi_l1a
from imap_processing.hi.l1b import hi_l1b
from imap_processing.hi.l1c import hi_l1c
from imap_processing.hit.l1a.hit_l1a import hit_l1a
from imap_processing.hit.l1b.hit_l1b import hit_l1b
from imap_processing.hit.l2.hit_l2 import hit_l2
from imap_processing.idex.idex_l1a import PacketParser
from imap_processing.idex.idex_l1b import idex_l1b
from imap_processing.lo.l1a import lo_l1a
from imap_processing.lo.l1b import lo_l1b
from imap_processing.lo.l1c import lo_l1c
from imap_processing.mag.l1a.mag_l1a import mag_l1a
from imap_processing.mag.l1b.mag_l1b import mag_l1b
from imap_processing.mag.l1c.mag_l1c import mag_l1c
from imap_processing.swapi.l1.swapi_l1 import swapi_l1
from imap_processing.swapi.l2.swapi_l2 import swapi_l2
from imap_processing.swe.l1a.swe_l1a import swe_l1a
from imap_processing.swe.l1b.swe_l1b import swe_l1b
from imap_processing.ultra.l1a import ultra_l1a
from imap_processing.ultra.l1b import ultra_l1b
from imap_processing.ultra.l1c import ultra_l1c

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """
    Parse the command line arguments.

    The expected input format is:
    --instrument "mag"
    --data-level "l1a"
    --descriptor "all"
    --start-date "20231212"
    --version "v001"
    --dependency '[
            {
                "type": "ancillary",
                "files": [
                    "imap_mag_l1b-cal_20250101_v001.cdf",
                    "imap_mag_l1b-cal_20250103_20250104_v002.cdf"
                ]
            },
            {
                "type": "science",
                "files": [
                    "imap_idex_l2_sci_20240312_v000.cdf",
                    "imap_idex_l2_sci_20240312_v001.cdf"
                ]
            }
        ]'
    --upload-to-sdc

    Returns
    -------
    args : argparse.Namespace
        An object containing the parsed arguments and their values.
    """
    description = (
        "This command line program invokes the processing pipeline "
        "for a specific instrument and data level. Example usage: "
        '"imap_cli --instrument "mag" '
        '--data-level "l1a" '
        '--descriptor "all" '
        ' --start-date "20231212" '
        '--version "v001" '
        '--dependency "['
        "    {"
        '        "type": "ancillary",'
        '        "files": ['
        '            "imap_mag_l1b-cal_20250101_v001.cdf",'
        '            "imap_mag_l1b-cal_20250103_20250104_v002.cdf"'
        "        ]"
        "    },"
        "    {"
        '        "type": "science",'
        '        "files": ['
        '            "imap_idex_l2_sci_20240312_v000.cdf",'
        '            "imap_idex_l2_sci_20240312_v001.cdf"'
        "        ]"
        "    }"
        "]"
        ' --upload-to-sdc"'
    )
    instrument_help = (
        "The instrument to process. Acceptable values are: "
        f"{imap_data_access.VALID_INSTRUMENTS}"
    )
    level_help = (
        "The data level to process. Acceptable values are: "
        f"{imap_processing.PROCESSING_LEVELS}"
    )
    descriptor_help = (
        "The descriptor of the product to process. This could be 'all' or a specific "
        "descriptor like 'sci-1min'. Default is 'all'."
    )
    dependency_help = (
        "Dependency information in str format."
        "Example:"
        "'["
        "    {"
        '        "type": "ancillary",'
        '        "files": ['
        '            "imap_mag_l1b-cal_20250101_v001.cdf",'
        '            "imap_mag_l1b-cal_20250103_20250104_v002.cdf"'
        "        ]"
        "    },"
        "    {"
        '        "type": "science",'
        '        "files": ['
        '            "imap_idex_l2_sci_20240312_v000.cdf",'
        '            "imap_idex_l2_sci_20240312_v001.cdf"'
        "        ]"
        "    }"
        "]'"
    )

    parser = argparse.ArgumentParser(prog="imap_cli", description=description)
    # TODO: Add version here and change our current "version" to "data-version"?
    # parser.add_argument(
    #     "--version",
    #     action="version",
    #     version=f"%(prog)s {imap_processing.__version__}",
    # )
    # Logging level
    parser.add_argument(
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Add verbose output",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )
    parser.add_argument("--instrument", type=str, required=True, help=instrument_help)
    parser.add_argument("--data-level", type=str, required=True, help=level_help)
    # TODO: unused for now, but needed for batch job handling
    # pass through of status in AWS
    parser.add_argument(
        "--descriptor", type=str, required=False, help=descriptor_help, default="all"
    )

    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start time for the output data. Format: YYYYMMDD",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        required=False,
        help="End time for the output data. If not provided, start_time will be used "
        "for end_time. Format: YYYYMMDD",
    )
    # TODO: Will need to add some way of including pointing numbers

    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Version of the data. Format: vXXX",
    )
    parser.add_argument(
        "--dependency",
        type=str,
        required=True,
        help=dependency_help,
    )

    parser.add_argument(
        "--upload-to-sdc",
        action="store_true",
        required=False,
        help="Upload completed output files to the IMAP SDC.",
    )
    args = parser.parse_args()

    return args


def _validate_args(args: argparse.Namespace) -> None:
    """
    Ensure that the  arguments are valid before kicking off the processing.

    Parameters
    ----------
    args : argparse.Namespace
        An object containing the parsed arguments and their values.
    """
    if args.instrument not in imap_data_access.VALID_INSTRUMENTS:
        raise ValueError(
            f"{args.instrument} is not in the supported instrument list: "
            f"{imap_data_access.VALID_INSTRUMENTS}"
        )
    if args.data_level not in imap_processing.PROCESSING_LEVELS[args.instrument]:
        raise ValueError(
            f"{args.data_level} is not a supported data level for the {args.instrument}"
            " instrument, valid levels are: "
            f"{imap_processing.PROCESSING_LEVELS[args.instrument]}"
        )


class ProcessInstrument(ABC):
    """
    An abstract base class containing a method to process an instrument.

    Parameters
    ----------
    data_level : str
        The data level to process (e.g. ``l1a``).
    data_descriptor : str
        The descriptor of the data to process (e.g. ``sci``).
    dependency_str : str
        A string representation of the dependencies for the instrument in the
        format:
        '[
            {
                "type": "ancillary",
                "files": [
                    "imap_mag_l1b-cal_20250101_v001.cdf",
                    "imap_mag_l1b-cal_20250103_20250104_v002.cdf"
                ]
            },
            {
                "type": "ancillary",
                "files": [
                    "imap_mag_l1b-lut_20250101_v001.cdf",
                ]
            },
            {
                "type": "science",
                "files": [
                    "imap_mag_l1a_norm-magi_20240312_v000.cdf",
                    "imap_mag_l1a_norm-magi_20240312_v001.cdf"
                ]
            },
            {
                "type": "science",
                "files": [
                    "imap_idex_l2_sci_20240312_v000.cdf",
                    "imap_idex_l2_sci_20240312_v001.cdf"
                ]
            }
        ]'
        This is what ProcessingInputCollection.serialize() outputs.
    start_date : str
        The start date for the output data in YYYYMMDD format.
    end_date : str
        The end date for the output data in YYYYMMDD format.
    version : str
        The version of the data in vXXX format.
    upload_to_sdc : bool
        A flag indicating whether to upload the output file to the SDC.
    """

    def __init__(
        self,
        data_level: str,
        data_descriptor: str,
        dependency_str: str,
        start_date: str,
        end_date: str,
        version: str,
        upload_to_sdc: bool,
    ) -> None:
        self.data_level = data_level
        self.descriptor = data_descriptor

        # Convert string into a dictionary
        self.dependencies = dependency_str
        self._dependency_list: ProcessingInputCollection = None

        self.start_date = start_date
        self.end_date = end_date
        if not end_date:
            self.end_date = start_date

        self.version = version
        self.upload_to_sdc = upload_to_sdc

    def download_dependencies(self) -> ProcessingInputCollection:
        """
        Download the dependencies for the instrument.

        Returns
        -------
        file_list : ProcessingInputCollection
            Object containing dependencies for the instrument.
        """
        input_collection = ProcessingInputCollection()
        input_collection.deserialize(self.dependencies)
        # Go through science, ancillary or SPICE dependencies list and
        # download all files
        for dependency in input_collection.processing_input:
            if dependency.input_type == ProcessingInputType.SPICE_FILE:
                logger.warning("SPICE is not implemented yet")
                continue

            # walk through and download all dependencies data
            for filepath in dependency.imap_file_paths:
                try:
                    download_path = filepath.construct_path()
                    logger.info(f"Downloading {download_path}")
                    imap_data_access.download(download_path)
                except HTTPError as e:
                    raise ValueError(f"Unable to download {filepath} file") from e

        return input_collection

    def upload_products(self, products: list[Path]) -> None:
        """
        Upload data products to the IMAP SDC.

        Parameters
        ----------
        products : list[Path]
            A list of file paths to upload to the SDC.
        """
        if self.upload_to_sdc:
            if len(products) == 0:
                logger.info("No files to upload.")
            for filename in products:
                logger.info(f"Uploading file: {filename}")
                imap_data_access.upload(filename)

    @final
    def process(self) -> None:
        """
        Run the processing workflow and cannot be overridden by subclasses.

        Each IMAP processing step consists of three steps:
        1. Pre-processing actions such as downloading dependencies for processing.
        2. Do the data processing. The result of this step will usually be a list
        of new products (files).
        3. Post-processing actions such as uploading files to the IMAP SDC.
        """
        logger.info(f"IMAP Processing Version: {imap_processing._version.__version__}")
        logger.info(f"Processing {self.__class__.__name__} level {self.data_level}")
        logger.info("Beginning preprocessing (download dependencies)")
        dependencies = self.pre_processing()
        logger.info("Beginning actual processing")
        products = self.do_processing(dependencies)
        logger.info("Beginning postprocessing (uploading data products)")
        self.post_processing(products)
        logger.info("Processing complete")

    def pre_processing(self) -> ProcessingInputCollection:
        """
        Complete pre-processing.

        For this baseclass, pre-processing consists of downloading dependencies
        for processing. Child classes can override this method to customize the
        pre-processing actions.

        Returns
        -------
        list[Path]
            List of dependencies downloaded from the IMAP SDC.
        """
        self._dependency_list = self.download_dependencies()
        return self._dependency_list

    @abstractmethod
    def do_processing(
        self, dependencies: ProcessingInputCollection
    ) -> list[xr.Dataset]:
        """
        Abstract method that processes the IMAP processing steps.

        All child classes must implement this method. Input is
        object containing dependencies and output is
        list of xr.Dataset containing processed data(s).

        Parameters
        ----------
        dependencies : ProcessingInputCollection
            Object containing dependencies to process.

        Returns
        -------
        list[xr.Dataset]
            List of products produced.
        """
        raise NotImplementedError

    def post_processing(self, datasets: list[xr.Dataset]) -> None:
        """
        Complete post-processing.

        Default post-processing consists of writing the datasets to local storage
        and then uploading those newly generated products to the IMAP SDC.
        Child classes can override this method to customize the
        post-processing actions.

        Parameters
        ----------
        datasets : list[xarray.Dataset]
            A list of datasets (products) produced by do_processing method.
        """
        if len(datasets) == 0:
            logger.info("No products to write to CDF file.")
            return

        logger.info("Writing products to local storage")

        list_of_files = [
            dep_ojb.filename_list for dep_ojb in self._dependency_list.processing_input
        ]
        list_of_files = np.array(list_of_files).flatten()
        logger.info("Parent files: %s", list_of_files)

        products = [
            write_cdf(dataset, parent_files=list_of_files) for dataset in datasets
        ]
        self.upload_products(products)


class Codice(ProcessInstrument):
    """Process CoDICE."""

    def do_processing(
        self, dependencies: ProcessingInputCollection
    ) -> list[xr.Dataset]:
        """
        Perform CoDICE specific processing.

        Parameters
        ----------
        dependencies : ProcessingInputCollection
            Object containing dependencies to process.

        Returns
        -------
         dataset : xr.Dataset
            Xr.Dataset of cdf file paths.
        """
        print(f"Processing CoDICE {self.data_level}")
        datasets: list[xr.Dataset] = []

        dependency_list = dependencies.processing_input
        if self.data_level == "l1a":
            if len(dependency_list) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for CoDICE L1a:"
                    f"{dependency_list}. Expected only one dependency."
                )
            # process data
            l0_file_obj = dependency_list[0].imap_file_paths[0]
            datasets = [
                codice_l1a.process_codice_l1a(
                    l0_file_obj.construct_path(), self.version
                )
            ]

        if self.data_level == "l1b":
            if len(dependency_list) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for CoDICE L1b:"
                    f"{dependency_list}. Expected only one dependency."
                )
            # process data
            l1b_science_file_obj = dependency_list[0].imap_file_paths[0]
            dependency = load_cdf(l1b_science_file_obj)
            datasets = [codice_l1b.process_codice_l1b(dependency, self.version)]

        return datasets


class Glows(ProcessInstrument):
    """Process GLOWS."""

    def do_processing(
        self, dependencies: ProcessingInputCollection
    ) -> list[xr.Dataset]:
        """
        Perform GLOWS specific processing.

        Parameters
        ----------
        dependencies : ProcessingInputCollection
            Object containing dependencies to process.

        Returns
        -------
        datasets : xr.Dataset
            Xr.dataset of products.
        """
        print(f"Processing GLOWS {self.data_level}")
        datasets: list[xr.Dataset] = []

        dependency_list = dependencies.processing_input
        if self.data_level == "l1a":
            if len(dependency_list) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for GLOWS L1A:"
                    f"{dependency_list}. Expected only one input dependency."
                )
            l0_file_obj = dependency_list[0].imap_file_paths[0]
            datasets = glows_l1a(l0_file_obj.construct_path(), self.version)

        if self.data_level == "l1b":
            if len(dependency_list) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for GLOWS L1B:"
                    f"{dependency_list}. Expected at least one input dependency."
                )
            l1a_file_obj = dependency_list[0].imap_file_paths[0]
            input_dataset = load_cdf(l1a_file_obj)
            datasets = [glows_l1b(input_dataset, self.version)]

        if self.data_level == "l2":
            if len(dependency_list) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for GLOWS L2:"
                    f"{dependency_list}. Expected only one input dependency."
                )
            l2_file_obj = dependency_list[0].imap_file_paths[0]
            input_dataset = load_cdf(l2_file_obj)
            datasets = glows_l2(input_dataset, self.version)

        return datasets


class Hi(ProcessInstrument):
    """Process IMAP-Hi."""

    def do_processing(
        self, dependencies: ProcessingInputCollection
    ) -> list[xr.Dataset]:
        """
        Perform IMAP-Hi specific processing.

        Parameters
        ----------
        dependencies : ProcessingInputCollection
            Object containing dependencies to process.

        Returns
        -------
        datasets : xr.Dataset
            Xr.Dataset of products.
        """
        print(f"Processing IMAP-Hi {self.data_level}")
        datasets: list[xr.Dataset] = []

        dependency_list = dependencies.processing_input
        if self.data_level == "l1a":
            # File path is expected output file path
            if len(dependency_list) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for Hi L1A:"
                    f"{dependency_list}. Expected only one dependency."
                )
            l0_file_obj = dependency_list[0].imap_file_paths[0]
            datasets = hi_l1a.hi_l1a(l0_file_obj.construct_path(), self.version)
        elif self.data_level == "l1b":
            l1a_file_obj = dependency_list[0].imap_file_paths
            hi_dependencies = [load_cdf(l1a_obj) for l1a_obj in l1a_file_obj]
            datasets = [hi_l1b.hi_l1b(hi_dependencies[0], self.version)]
        elif self.data_level == "l1c":
            # TODO: Add PSET calibration product config file dependency and remove
            #    below injected dependency
            # TODO: fix this once anc file of csv file format upload is available
            hi_dependencies = []
            hi_dependencies.append(
                Path(__file__).parent
                / "tests/hi/test_data/l1"
                / "imap_his_pset-calibration-prod-config_20240101_v001.csv"
            )
            l1b_file_obj = dependency_list[0].imap_file_paths[0]
            hi_dependencies[0] = load_cdf(l1b_file_obj)
            datasets = [hi_l1c.hi_l1c(hi_dependencies, self.version)]
        else:
            raise NotImplementedError(
                f"Hi processing not implemented for level {self.data_level}"
            )
        return datasets


class Hit(ProcessInstrument):
    """Process HIT."""

    def do_processing(
        self, dependencies: ProcessingInputCollection
    ) -> list[xr.Dataset]:
        """
        Perform HIT specific processing.

        Parameters
        ----------
        dependencies : ProcessingInputCollection
            Object containing dependencies to process.

        Returns
        -------
        datasets : xr.Dataset
            Xr.Dataset of datasets.
        """
        print(f"Processing HIT {self.data_level}")
        datasets: list[xr.Dataset] = []

        dependency_list = dependencies.processing_input
        if self.data_level == "l1a":
            if len(dependency_list) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for HIT L1A:"
                    f"{dependency_list}. Expected only one dependency."
                )
            l0_file_obj = dependency_list[0].imap_file_paths[0]
            # process data to L1A products
            datasets = hit_l1a(l0_file_obj.construct_path(), self.version)

        elif self.data_level == "l1b":
            if len(dependency_list) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for HIT L1B:"
                    f"{dependency_list}. Expected only one dependency."
                )
            data_dict = {}
            if dependency_list[0]["descriptor"] == "hk":
                # Add path to CCSDS file to process housekeeping
                l0_file_obj = dependency_list[0].imap_file_paths[0]
                data_dict["imap_hit_l0_raw"] = l0_file_obj.construct_path()
            else:
                # Add L1A dataset to process science data
                l1a_file_obj = dependency_list[0].imap_file_paths[0]
                l1a_dataset = load_cdf(l1a_file_obj)
                data_dict[l1a_dataset.attrs["Logical_source"]] = l1a_dataset
            # process data to L1B products
            datasets = hit_l1b(data_dict, self.version)
        elif self.data_level == "l2":
            if len(dependency_list) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for HIT L2:"
                    f"{dependency_list}. Expected only one dependency."
                )
            # Add L1B dataset to process science data
            l1b_file_obj = dependency_list[0].imap_file_paths[0]
            l1b_dataset = load_cdf(l1b_file_obj)
            # process data to L2 products
            datasets = hit_l2(l1b_dataset, self.version)

        return datasets


class Idex(ProcessInstrument):
    """Process IDEX."""

    def do_processing(
        self, dependencies: ProcessingInputCollection
    ) -> list[xr.Dataset]:
        """
        Perform IDEX specific processing.

        Parameters
        ----------
        dependencies : ProcessingInputCollection
            Object containing dependencies to process.

        Returns
        -------
        datasets : xr.Dataset
            Xr.Dataset of cdf file paths.
        """
        print(f"Processing IDEX {self.data_level}")
        datasets: list[xr.Dataset] = []

        dependency_list = dependencies.processing_input
        if self.data_level == "l1a":
            if len(dependency_list) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for IDEX L1A:"
                    f"{dependency_list}. Expected only one science dependency."
                )
            # get l0 file
            l0_file_obj = dependency_list[0].imap_file_paths[0]
            datasets = [PacketParser(l0_file_obj.construct_path(), self.version).data]
        elif self.data_level == "l1b":
            if len(dependency_list) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for IDEX L1B:"
                    f"{dependency_list}. Expected only one science dependency."
                )
            # get CDF file
            science_file_obj = dependency_list[0].imap_file_paths[0]
            # process data
            dependency = load_cdf(science_file_obj)
            datasets = [idex_l1b(dependency, self.version)]
        return datasets


class Lo(ProcessInstrument):
    """Process IMAP-Lo."""

    def do_processing(
        self, dependencies: ProcessingInputCollection
    ) -> list[xr.Dataset]:
        """
        Perform IMAP-Lo specific processing.

        Parameters
        ----------
        dependencies : ProcessingInputCollection
            Object containing dependencies to process.

        Returns
        -------
        dataset : xr.Dataset
            Xr.Dataset of output files.
        """
        print(f"Processing IMAP-Lo {self.data_level}")
        datasets: list[xr.Dataset] = []
        dependency_list = dependencies.processing_input
        if self.data_level == "l1a":
            # L1A packet / products are 1 to 1. Should only have
            # one dependency file
            if len(dependency_list) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for IMAP-Lo L1A:"
                    f"{dependency_list}. Expected only one dependency."
                )
            l0_file_obj = dependency_list[0].imap_file_paths[0]
            datasets = lo_l1a.lo_l1a(l0_file_obj.construct_path(), self.version)

        elif self.data_level == "l1b":
            data_dict = {}
            # TODO: ask what we expect here with Sean
            for file_obj in dependency_list.imap_file_paths:
                dataset = load_cdf(file_obj)
                data_dict[dataset.attrs["Logical_source"]] = dataset
            datasets = lo_l1b.lo_l1b(data_dict, self.version)

        elif self.data_level == "l1c":
            data_dict = {}
            for file_obj in dependency_list.imap_file_paths:
                dataset = load_cdf(file_obj)
                data_dict[dataset.attrs["Logical_source"]] = dataset
            # TODO: This is returning the wrong type
            datasets = lo_l1c.lo_l1c(data_dict, self.version)

        return datasets


class Mag(ProcessInstrument):
    """Process MAG."""

    def do_processing(
        self, dependencies: ProcessingInputCollection
    ) -> list[xr.Dataset]:
        """
        Perform MAG specific processing.

        Parameters
        ----------
        dependencies : ProcessingInputCollection
            Object containing dependencies to process.

        Returns
        -------
        dataset : xr.Dataset
            Xr.Dataset of output files.
        """
        print(f"Processing MAG {self.data_level}")
        datasets: list[xr.Dataset] = []

        dependency_list = dependencies.processing_input
        if self.data_level == "l1a":
            # File path is expected output file path
            if len(dependency_list) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for MAG L1A:"
                    f"{dependency_list}. Expected only one dependency."
                )
            # TODO: Update this type
            l0_file_obj = dependency_list[0].imap_file_paths[0]
            datasets = mag_l1a(l0_file_obj.construct_path(), data_version=self.version)

        if self.data_level == "l1b":
            if len(dependency_list) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for MAG L1B:"
                    f"{dependency_list}. Expected only one dependency."
                )
            l1a_file_obj = dependency_list[0].imap_file_paths[0]
            input_data = load_cdf(l1a_file_obj)
            datasets = [mag_l1b(input_data, self.version)]

        if self.data_level == "l1c":
            # L1C depends on matching norm/burst files: eg burst-magi and norm-magi or
            # burst-mago and norm-mago
            if len(dependency_list) != 2:
                raise ValueError(
                    f"Invalid dependencies found for MAG L1C:"
                    f"{dependency_list}. Expected two dependencies."
                )

            input_data = [load_cdf(dep.imap_file_paths[0]) for dep in dependency_list]
            # Input datasets can be in any order
            datasets = [mag_l1c(input_data[0], input_data[1], self.version)]

        return datasets


class Swapi(ProcessInstrument):
    """Process SWAPI."""

    def do_processing(
        self, dependencies: ProcessingInputCollection
    ) -> list[xr.Dataset]:
        """
        Perform SWAPI specific processing.

        Parameters
        ----------
        dependencies : ProcessingInputCollection
            Object containing dependencies to process.

        Returns
        -------
        dataset : xr.Dataset
            Xr.Dataset of products.
        """
        print(f"Processing SWAPI {self.data_level}")
        datasets: list[xr.Dataset] = []

        dependency_list = dependencies.processing_input
        if self.data_level == "l1":
            # For science, we expect l0 raw file and L1 housekeeping file
            if self.descriptor == "sci" and len(dependency_list) != 2:
                raise ValueError(
                    f"Unexpected dependencies found for SWAPI L1 science:"
                    f"{dependency_list}. Expected only two dependencies."
                )
            # For housekeeping, we expect only L0 raw file
            if self.descriptor == "hk" and len(dependency_list) != 1:
                raise ValueError(
                    f"Unexpected dependencies found for SWAPI L1 housekeeping:"
                    f"{dependency_list}. Expected only one dependency."
                )
            # process science or housekeeping data
            datasets = swapi_l1(dependencies, self.version)
        elif self.data_level == "l2":
            if len(dependency_list) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for SWAPI L2:"
                    f"{dependency_list}. Expected only one dependency."
                )
            # process data
            l1_file_obj = dependency_list[0].imap_file_paths[0]
            l1_dataset = load_cdf(l1_file_obj)
            datasets = [swapi_l2(l1_dataset, self.version)]

        return datasets


class Swe(ProcessInstrument):
    """Process SWE."""

    def do_processing(
        self, dependencies: ProcessingInputCollection
    ) -> list[xr.Dataset]:
        """
        Perform SWE specific processing.

        Parameters
        ----------
        dependencies : ProcessingInputCollection
            Object containing dependencies to process.

        Returns
        -------
        dataset : xr.Dataset
            Path to cdf file.
        """
        print(f"Processing SWE {self.data_level}")
        datasets: list[xr.Dataset] = []

        dependency_list = dependencies.processing_input
        if self.data_level == "l1a":
            if len(dependency_list) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for SWE L1A:"
                    f"{dependency_list}. Expected only one dependency."
                )
            l0_file_obj = dependency_list[0].imap_file_paths[0]
            datasets = swe_l1a(
                str(l0_file_obj.construct_path()), data_version=self.version
            )
            # Right now, we only process science data. Therefore,
            # we expect only one dataset to be returned.

        elif self.data_level == "l1b":
            if len(dependency_list) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for SWE L1B:"
                    f"{dependency_list}. Expected only one dependency."
                )
            l1a_file_obj = dependency_list[0].imap_file_paths[0]
            # read CDF file
            l1a_dataset = load_cdf(l1a_file_obj)
            # TODO: read lookup table and in-flight calibration data here.
            datasets = swe_l1b(l1a_dataset, data_version=self.version)
        else:
            print("Did not recognize data level. No processing done.")

        return datasets


class Ultra(ProcessInstrument):
    """Process IMAP-Ultra."""

    def do_processing(
        self, dependencies: ProcessingInputCollection
    ) -> list[xr.Dataset]:
        """
        Perform IMAP-Ultra specific processing.

        Parameters
        ----------
        dependencies : ProcessingInputCollection
            Object containing dependencies to process.

        Returns
        -------
        datasets : xr.Dataset
            Xr.Dataset of products.
        """
        print(f"Processing IMAP-Ultra {self.data_level}")
        datasets: list[xr.Dataset] = []

        dependency_list = dependencies.processing_input
        if self.data_level == "l1a":
            # File path is expected output file path
            if len(dependency_list) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for ULTRA L1A:"
                    f"{dependency_list}. Expected only one dependency."
                )
            l0_file_obj = dependency_list[0].imap_file_paths[0]
            datasets = ultra_l1a.ultra_l1a(l0_file_obj.construct_path(), self.version)

        elif self.data_level == "l1b":
            data_dict = {}
            for dep in dependency_list:
                dataset = load_cdf(dep.imap_file_paths[0])
                data_dict[dataset.attrs["Logical_source"]] = dataset
            datasets = ultra_l1b.ultra_l1b(data_dict, self.version)

        elif self.data_level == "l1c":
            data_dict = {}
            for dep in dependency_list:
                dataset = load_cdf(dep.imap_file_paths[0])
                data_dict[dataset.attrs["Logical_source"]] = dataset
            datasets = ultra_l1c.ultra_l1c(data_dict, self.version)

        return datasets


def main() -> None:
    """
    Run the processing for a specific instrument & data level.

    Set up the command line arguments, parse them, and then invoke the
    appropriate instrument processing function.
    """
    # NOTE: This is to allow the cli script to be installed and reference
    #       this function for an entrypoint.
    args = _parse_args()

    _validate_args(args)
    cls = getattr(sys.modules[__name__], args.instrument.capitalize())
    instrument = cls(
        args.data_level,
        args.descriptor,
        args.dependency,
        args.start_date,
        args.end_date,
        args.version,
        args.upload_to_sdc,
    )
    instrument.process()


if __name__ == "__main__":
    main()
