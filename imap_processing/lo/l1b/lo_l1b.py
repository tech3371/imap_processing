"""IMAP-Lo L1B Data Processing."""

from dataclasses import Field
from pathlib import Path
from typing import Any, Union

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.spice.time import met_to_ttj2000ns


def lo_l1b(dependencies: dict) -> list[Path]:
    """
    Will process IMAP-Lo L1A data into L1B CDF data products.

    Parameters
    ----------
    dependencies : dict
        Dictionary of datasets needed for L1B data product creation in xarray Datasets.

    Returns
    -------
    created_file_paths : list[pathlib.Path]
        Location of created CDF files.
    """
    # create the attribute manager for this data level
    attr_mgr_l1b = ImapCdfAttributes()
    attr_mgr_l1b.add_instrument_global_attrs(instrument="lo")
    attr_mgr_l1b.add_instrument_variable_attrs(instrument="lo", level="l1b")
    # create the attribute manager to access L1A fillval attributes
    attr_mgr_l1a = ImapCdfAttributes()
    attr_mgr_l1a.add_instrument_variable_attrs(instrument="lo", level="l1a")

    # if the dependencies are used to create Annotated Direct Events
    if "imap_lo_l1a_de" in dependencies and "imap_lo_l1a_spin" in dependencies:
        logical_source = "imap_lo_l1b_de"
        # get the dependency dataset for l1b direct events
        l1a_de = dependencies["imap_lo_l1a_de"]
        spin_data = dependencies["imap_lo_l1a_spin"]

        # Initialize the L1B DE dataset
        l1b_de = initialize_l1b_de(l1a_de, attr_mgr_l1b, logical_source)
        # Get the start and end times for each spin epoch
        acq_start, acq_end = convert_start_end_acq_times(spin_data)
        # Get the average spin durations for each epoch
        avg_spin_durations = get_avg_spin_durations(acq_start, acq_end)
        # get spin angle (0 - 360 degrees) for each DE
        spin_angle = get_spin_angle(l1a_de)
        # calculate and set the spin bin based on the spin angle
        # spin bins are 0 - 60 bins
        l1b_de = set_spin_bin(l1b_de, spin_angle)
        # set the spin cycle for each direct event
        l1b_de = set_spin_cycle(l1a_de, l1b_de)
        # get spin start times for each event
        spin_start_time = get_spin_start_times(l1a_de, l1b_de, spin_data, acq_end)
        # get the absolute met for each event
        l1b_de = set_event_met(l1a_de, l1b_de, spin_start_time, avg_spin_durations)
        # set the epoch for each event
        l1b_de = set_each_event_epoch(l1b_de)

    return [l1b_de]


def initialize_l1b_de(
    l1a_de: xr.Dataset, attr_mgr_l1b: ImapCdfAttributes, logical_source: str
) -> xr.Dataset:
    """
    Initialize the L1B DE dataset.

    Create an empty L1B DE dataset and copy over fields from the L1A DE that will
    not change during L1B processing.

    Parameters
    ----------
    l1a_de : xarray.Dataset
        The L1A DE dataset.
    attr_mgr_l1b : ImapCdfAttributes
        Attribute manager used to get the global attributes for the L1B DE dataset.
    logical_source : str
        The logical source of the direct event product.

    Returns
    -------
    l1b_de : xarray.Dataset
        The initialized L1B DE dataset.
    """
    l1b_de = xr.Dataset(
        attrs=attr_mgr_l1b.get_global_attributes(logical_source),
    )

    # Copy over fields from L1A DE that will not change in L1B processing
    l1b_de["pos"] = xr.DataArray(
        l1a_de["pos"].values,
        dims=["epoch"],
        # TODO: Add pos to YAML file
        # attrs=attr_mgr.get_variable_attributes("pos"),
    )
    l1b_de["mode"] = xr.DataArray(
        l1a_de["mode"].values,
        dims=["epoch"],
        # TODO: Add mode to YAML file
        # attrs=attr_mgr.get_variable_attributes("mode"),
    )
    l1b_de["absent"] = xr.DataArray(
        l1a_de["coincidence_type"].values,
        dims=["epoch"],
        # TODO: Add absent to YAML file
        # attrs=attr_mgr.get_variable_attributes("absent"),
    )
    l1b_de["esa_step"] = xr.DataArray(
        l1a_de["esa_step"].values,
        dims=["epoch"],
        # TODO: Add esa_step to YAML file
        # attrs=attr_mgr.get_variable_attributes("esa_step"),
    )

    return l1b_de


def convert_start_end_acq_times(
    spin_data: xr.Dataset,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Convert the start and end times from the spin data.

    The L1A spin data start and end acquisition times are stored in seconds and
    subseconds (microseconds). This function converts them to a single time in seconds.

    Parameters
    ----------
    spin_data : xarray.Dataset
        The L1A Spin dataset containing the start and end acquisition times.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        A tuple containing the start and end acquisition times as xarray DataArrays.
    """
    # Convert subseconds from microseconds to seconds
    acq_start = spin_data["acq_start_sec"] + spin_data["acq_start_subsec"] * 1e-6
    acq_end = spin_data["acq_end_sec"] + spin_data["acq_end_subsec"] * 1e-6
    return (acq_start, acq_end)


def get_avg_spin_durations(
    acq_start: xr.DataArray, acq_end: xr.DataArray
) -> xr.DataArray:
    """
    Get the average spin duration for each spin epoch.

    Parameters
    ----------
    acq_start : xarray.DataArray
        The start acquisition times for each spin epoch.
    acq_end : xarray.DataArray
        The end acquisition times for each spin epoch.

    Returns
    -------
    avg_spin_durations : xarray.DataArray
        The average spin duration for each spin epoch.
    """
    # Get the avg spin duration for each spin epoch
    # There are 28 spins per epoch (1 aggregated science cycle)
    avg_spin_durations = (acq_end - acq_start) / 28
    return avg_spin_durations


def get_spin_angle(l1a_de: xr.Dataset) -> Union[np.ndarray[np.float64], Any]:
    """
    Get the spin angle (0 - 360 degrees) for each DE.

    Parameters
    ----------
    l1a_de : xarray.Dataset
        The L1A DE dataset.

    Returns
    -------
    spin_angle : np.ndarray
        The spin angle for each DE.
    """
    de_times = l1a_de["de_time"].values
    # DE Time is 12 bit DN. The max possible value is 4096
    spin_angle = np.array(de_times / 4096 * 360, dtype=np.float64)
    return spin_angle


def set_spin_bin(l1b_de: xr.Dataset, spin_angle: np.ndarray) -> xr.Dataset:
    """
    Set the spin bin (0 - 60 bins) for each Direct Event where each bin is 6 degrees.

    Parameters
    ----------
    l1b_de : xarray.Dataset
        The L1B Direct Event dataset.
    spin_angle : np.ndarray
        The spin angle (0-360 degrees) for each Direct Event.

    Returns
    -------
    l1b_de : xarray.Dataset
        The L1B DE dataset with the spin bin added.
    """
    # Get the spin bin for each DE
    # Spin bins are 0 - 60 where each bin is 6 degrees
    spin_bin = (spin_angle // 6).astype(int)
    l1b_de["spin_bin"] = xr.DataArray(
        spin_bin,
        dims=["epoch"],
        # TODO: Add spin angle to YAML file
        # attrs=attr_mgr.get_variable_attributes("spin_bin"),
    )
    return l1b_de


def set_spin_cycle(l1a_de: xr.Dataset, l1b_de: xr.Dataset) -> xr.Dataset:
    """
    Set the spin cycle for each direct event.

    spin_cycle = spin_start + 7 + (esa_step - 1) * 2

    where spin_start is the spin number for the first spin
    in an Aggregated Science Cycle (ASC) and esa_step is the esa_step for a direct event

    The 28 spins in a spin epoch spans one ASC.

    Parameters
    ----------
    l1a_de : xarray.Dataset
        The L1A DE dataset.
    l1b_de : xarray.Dataset
        The L1B DE dataset.

    Returns
    -------
    l1b_de : xarray.Dataset
        The L1B DE dataset with the spin cycle added for each direct event.
    """
    counts = l1a_de["de_count"].values
    # split the esa_steps into ASC groups
    de_asc_groups = np.split(l1a_de["esa_step"].values, np.cumsum(counts)[:-1])
    spin_cycle = []
    for i, esa_asc_group in enumerate(de_asc_groups):
        # TODO: Spin Number does not reset for each pointing. Need to figure out
        #  how to retain this information across days
        # increment the spin_start by 28 after each aggregated science cycle
        spin_start = i * 28
        # calculate the spin cycle for each DE in the ASC group
        # TODO: Add equation number in algorithm document when new version is
        # available. Add to docstring as well
        spin_cycle.extend(spin_start + 7 + (esa_asc_group - 1) * 2)

    l1b_de["spin_cycle"] = xr.DataArray(
        spin_cycle,
        dims=["epoch"],
        # TODO: Add spin cycle to YAML file
        # attrs=attr_mgr.get_variable_attributes("spin_cycle"),
    )

    return l1b_de


def get_spin_start_times(
    l1a_de: xr.Dataset, l1b_de: xr.Dataset, spin_data: xr.Dataset, acq_end: xr.DataArray
) -> xr.DataArray:
    """
    Get the start time for the spin that each direct event is in.

    The resulting array of spin start times will be equal to the length of the direct
    events. If two direct events occurred in the same spin, then there will be repeating
    spin start times.

    Parameters
    ----------
    l1a_de : xr.Dataset
        The L1A DE dataset.
    l1b_de : xr.Dataset
        The L1B DE dataset.
    spin_data : xr.Dataset
        The L1A Spin dataset.
    acq_end : xr.DataArray
        The end acquisition times for each spin ASC.

    Returns
    -------
    spin_start_time : xr.DataArray
        The start time for the spin that each direct event is in.
    """
    met = l1a_de["met"].values
    # Find the closest stop_acq for each shcoarse
    closest_stop_acq_indices = np.abs(met[:, None] - acq_end.values).argmin(axis=1)
    # There are 28 spins per epoch (1 aggregated science cycle)
    # Set the spin_cycle_num to the spin number relative to the
    # start of the ASC
    spin_cycle_num = l1b_de["spin_cycle"] % 28
    # Get the seconds portion of the start time for each spin
    start_sec_spins = spin_data["start_sec_spin"].values[
        closest_stop_acq_indices, spin_cycle_num
    ]
    # Get the subseconds portion of the spin start time and convert from
    # microseconds to seconds
    start_subsec_spins = (
        spin_data["start_subsec_spin"].values[closest_stop_acq_indices, spin_cycle_num]
        * 1e-6
    )

    # Combine the seconds and subseconds to get the start time for each spin
    spin_start_time = start_sec_spins + start_subsec_spins
    return xr.DataArray(spin_start_time)


def set_event_met(
    l1a_de: xr.Dataset,
    l1b_de: xr.Dataset,
    spin_start_time: xr.DataArray,
    avg_spin_durations: xr.DataArray,
) -> xr.Dataset:
    """
    Get the event MET for each direct event.

    Each direct event is converted from a data number to engineering unit in seconds.
    de_eu_time = de_dn_time / 4096 * avg_spin_duration
    where de_dn_time is the direct event time Data Number (DN) and avg_spin_duration
    is the average spin duration for the Aggregated Science Cycle (ASC) that the
    event was measured in.

    The direct event time is the time of direct event relative to the start of the spin.
    The event MET is the sum of the start time of the spin and the
    direct event EU time.

    Parameters
    ----------
    l1a_de : xr.Dataset
        The L1A DE dataset.
    l1b_de : xr.Dataset
        The L1B DE dataset.
    spin_start_time : np.ndarray
        The start time for the spin that each direct event is in.
    avg_spin_durations : xr.DataArray
        The average spin duration for each epoch.

    Returns
    -------
    l1b_de : xr.Dataset
        The L1B DE dataset with the event MET.
    """
    counts = l1a_de["de_count"].values
    de_time_asc_groups = np.split(l1a_de["de_time"].values, np.cumsum(counts)[:-1])
    de_times_eu = []
    for i, de_time_asc in enumerate(de_time_asc_groups):
        # DE Time is 12 bit DN. The max possible value is 4095
        # divide by 4096 to get fraction of a spin duration
        de_times_eu.extend(de_time_asc / 4096 * avg_spin_durations[i].values)

    l1b_de["event_met"] = xr.DataArray(
        spin_start_time + de_times_eu,
        dims=["epoch"],
        # attrs=attr_mgr.get_variable_attributes("epoch")
    )
    return l1b_de


def set_each_event_epoch(l1b_de: xr.Dataset) -> xr.Dataset:
    """
    Set the epoch for each direct event.

    Parameters
    ----------
    l1b_de : xr.Dataset
        The L1B DE dataset.

    Returns
    -------
    l1b_de : xr.Dataset
        The L1B DE dataset with the epoch set for each event.
    """
    l1b_de["epoch"] = xr.DataArray(
        met_to_ttj2000ns(l1b_de["event_met"].values),
        dims=["epoch"],
        # attrs=attr_mgr.get_variable_attributes("epoch")
    )
    return l1b_de


# TODO: This is going to work differently when I sample data.
#  The data_fields input is temporary.
def create_datasets(
    attr_mgr: ImapCdfAttributes,
    logical_source: str,
    data_fields: list[Field],
) -> xr.Dataset:
    """
    Create a dataset using the populated data classes.

    Parameters
    ----------
    attr_mgr : ImapCdfAttributes
        Attribute manager used to get the data product field's attributes.
    logical_source : str
        The logical source of the data product that's being created.
    data_fields : list[dataclasses.Field]
        List of Fields for data classes.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset with all data product fields in xr.DataArray.
    """
    # TODO: Once L1B DE processing is implemented using the spin packet
    #  and relative L1A DE time to calculate the absolute DE time,
    #  this epoch conversion will go away and the time in the DE dataclass
    #  can be used direction
    epoch_converted_time = met_to_ttj2000ns([0, 1, 2])

    # Create a data array for the epoch time
    # TODO: might need to update the attrs to use new YAML file
    epoch_time = xr.DataArray(
        data=epoch_converted_time,
        name="epoch",
        dims=["epoch"],
        attrs=attr_mgr.get_variable_attributes("epoch"),
    )

    if logical_source == "imap_lo_l1b_de":
        direction_vec = xr.DataArray(
            data=[0, 1, 2],
            name="direction_vec",
            dims=["direction_vec"],
            attrs=attr_mgr.get_variable_attributes("direction_vec"),
        )

        direction_vec_label = xr.DataArray(
            data=direction_vec.values.astype(str),
            name="direction_vec_label",
            dims=["direction_vec_label"],
            attrs=attr_mgr.get_variable_attributes("direction_vec_label"),
        )

        dataset = xr.Dataset(
            coords={
                "epoch": epoch_time,
                "direction_vec": direction_vec,
                "direction_vec_label": direction_vec_label,
            },
            attrs=attr_mgr.get_global_attributes(logical_source),
        )

    # Loop through the data fields that were pulled from the
    # data class. These should match the field names given
    # to each field in the YAML attribute file
    for data_field in data_fields:
        field = data_field.name.lower()
        # Create a list of all the dimensions using the DEPEND_I keys in the
        # YAML attributes
        dims = [
            value
            for key, value in attr_mgr.get_variable_attributes(field).items()
            if "DEPEND" in key
        ]

        # Create a data array for the current field and add it to the dataset
        # TODO: TEMPORARY. need to update to use l1a data once that's available.
        #  Won't need to check for the direction field when I have sample data either.
        if field == "direction":
            dataset[field] = xr.DataArray(
                [[0, 0, 1], [0, 1, 0], [0, 0, 1]],
                dims=dims,
                attrs=attr_mgr.get_variable_attributes(field),
            )
        # TODO: This is temporary.
        #  The data type will be set in the data class when that's created
        elif field in ["tof0", "tof1", "tof2", "tof3"]:
            dataset[field] = xr.DataArray(
                [np.float16(1), np.float16(1), np.float16(1)],
                dims=dims,
                attrs=attr_mgr.get_variable_attributes(field),
            )
        else:
            dataset[field] = xr.DataArray(
                [1, 1, 1], dims=dims, attrs=attr_mgr.get_variable_attributes(field)
            )

    return dataset
