"""MAG L1B Processing."""

from pathlib import Path

import numpy as np
import xarray as xr
from xarray import Dataset

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import load_cdf


def mag_l1b(input_dataset: xr.Dataset, version: str) -> Dataset:
    """
    Will process MAG L1B data from L1A data.

    Parameters
    ----------
    input_dataset : xr.Dataset
        The input dataset to process.
    version : str
        The version of the output data.

    Returns
    -------
    output_dataset : xr.Dataset
        The processed dataset.
    """
    # TODO:
    # Read in calibration file
    # multiply all vectors by calibration file
    if "raw" in input_dataset.attrs["Logical_source"]:
        # Raw files should not be processed in L1B.
        raise ValueError("Raw L1A file passed into L1B. Unable to process.")

    output_dataset = mag_l1b_processing(input_dataset)
    attribute_manager = ImapCdfAttributes()
    attribute_manager.add_instrument_global_attrs("mag")
    attribute_manager.add_global_attribute("Data_version", version)

    # Variable attributes can remain the same as L1A
    input_logical_source = input_dataset.attrs["Logical_source"]
    if isinstance(input_dataset.attrs["Logical_source"], list):
        input_logical_source = input_dataset.attrs["Logical_source"][0]

    logical_source = input_logical_source.replace("l1a", "l1b")
    output_dataset.attrs = attribute_manager.get_global_attributes(logical_source)

    return output_dataset


def mag_l1b_processing(input_dataset: xr.Dataset) -> xr.Dataset:
    """
    Will process MAG L1B data from L1A data.

    MAG L1B is almost identical to L1A, with only the vectors and attributes getting
    updated. All non-vector variables are the same.

    Parameters
    ----------
    input_dataset : xr.Dataset
        The input dataset to process.

    Returns
    -------
    output_dataset : xr.Dataset
        L1b dataset.
    """
    # TODO: There is a time alignment step that will add a lot of complexity.
    # This needs to be done once we have some SPICE time data.

    mag_attributes = ImapCdfAttributes()
    mag_attributes.add_instrument_variable_attrs("mag", "l1")

    dims = [["direction"], ["compression"]]
    new_dims = [["direction"], ["compression"]]
    # TODO: This should definitely be loaded from AWS
    calibration_dataset = load_cdf(
        Path(__file__).parent / "imap_calibration_mag_20240229_v01.cdf"
    )
    # TODO: add time shift
    # TODO: Check validity of time range for calibration
    source = input_dataset.attrs["Logical_source"]
    if isinstance(source, list):
        source = source[0]
    if "mago" in source:
        calibration_matrix = calibration_dataset["MFOTOURFO"]
    elif "magi" in source:
        calibration_matrix = calibration_dataset["MFITOURFI"]
    else:
        raise ValueError(
            f"Calibration matrix not found, invalid logical source "
            f"{input_dataset.attrs['Logical_source']}"
        )

    l1b_fields = xr.apply_ufunc(
        update_vector,
        input_dataset["vectors"],
        input_dataset["compression_flags"],
        input_core_dims=dims,
        output_core_dims=new_dims,
        vectorize=True,
        keep_attrs=True,
        kwargs={"calibration_matrix": calibration_matrix},
    )

    output_dataset = input_dataset.copy()
    output_dataset["vectors"].data = l1b_fields[0].data

    output_dataset["epoch"].attrs = mag_attributes.get_variable_attributes("epoch")
    output_dataset["direction"].attrs = mag_attributes.get_variable_attributes(
        "direction_attrs"
    )
    output_dataset["compression"].attrs = mag_attributes.get_variable_attributes(
        "compression_attrs"
    )
    output_dataset["direction_label"].attrs = mag_attributes.get_variable_attributes(
        "direction_label", check_schema=False
    )
    output_dataset["compression_label"].attrs = mag_attributes.get_variable_attributes(
        "compression_label", check_schema=False
    )

    return output_dataset


def update_vector(
    input_vector: np.ndarray,
    input_compression: np.ndarray,
    calibration_matrix: xr.DataArray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply calibration and compression scaling to vector.

    This calls, in sequence, calibrate_vector and rescale_vector to apply L1B processing
    to the input vector.

    Parameters
    ----------
    input_vector : numpy.ndarray
        One input vector to update, looking like (x, y, z, range).
    input_compression : numpy.ndarray
        Compression flags corresponding to the vector, looking like (is_compressed,
        compression_width).
    calibration_matrix : xr.DataArray
        DataArray containing the full set of calibration matrices, for each range.
        Size is ((3, 3, 4)).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Updated vector and the same compression flags.
    """
    vector = rescale_vector(input_vector, input_compression)
    cal_vector = calibrate_vector(vector, calibration_matrix)
    return cal_vector, input_compression


def rescale_vector(
    input_vector: np.ndarray, compression_flags: np.ndarray
) -> np.ndarray:
    """
    Rescale vector based on compression flags.

    If the first value of compression_flags is zero, this just returns the input_vector
    unchanged. Otherwise, the vector is scaled using the compression width, which is
    the second part of compression_flags.

    The vector is scaled using the following equation:
    M = 2 ^ (16-width)
    output_vector = input_vector * M

    Therefore, for a 16 bit width, the same vector is returned.

    Parameters
    ----------
    input_vector : numpy.ndarray
        One input vector to update, looking like (x, y, z, range).
    compression_flags : numpy.ndarray
        Compression flags corresponding to the vector, looking like (is_compressed,
        compression_width).

    Returns
    -------
    output_vector : numpy.ndarray
        Updated vector.
    """
    output_vector: np.ndarray = input_vector.astype(np.float64)

    if compression_flags[0]:
        factor = np.float_power(2, (16 - compression_flags[1]))
        output_vector[:3] = input_vector.astype(np.float64)[:3] * factor

    return output_vector


def calibrate_vector(
    input_vector: np.ndarray, calibration_matrix: xr.DataArray = None
) -> np.ndarray:
    """
    Apply calibration matrix to input vector.

    The calibration matrix contains 3x3 matrices for each range. To calibrate the input
    vector, we take the range (which is the fourth value of the vector) to determine
    the correct calibration matrix. We then multiply the input vector by that matrix.

    Parameters
    ----------
    input_vector : numpy.ndarray
        The input vector to calibrate [x, y, z, range].
    calibration_matrix : xr.DataArray
        The full set of calibration matrices, for each range. Size is ((3, 3, 4)).

    Returns
    -------
    updated_vector : numpy.ndarray
        Calibrated vector.
    """
    updated_vector: np.ndarray = input_vector.copy()
    if input_vector[3] % 1 != 0:
        raise ValueError("Range must be an integer.")

    range = int(input_vector[3])
    x_y_z = input_vector[:3]
    updated_vector[:3] = np.dot(calibration_matrix.values[:, :, range], x_y_z)
    return updated_vector
