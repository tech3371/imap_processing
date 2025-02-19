"""Various utility classes and functions to support SWE processing."""

from enum import IntEnum

import numpy as np
import numpy.typing as npt
import pandas as pd

from imap_processing import imap_module_directory

N_ESA_STEPS = 24
N_MEASUREMENTS = 30
N_CEMS = 7
N_QUARTER_CYCLES = 4
N_ANGLE_BINS = 30

MICROSECONDS_IN_SECOND = 1e6

# TODO: add these to instrument status summary
ENERGY_CONVERSION_FACTOR = 4.75
# 7 CEMs geometric factors in cm^2 sr eV/eV units.
GEOMETRIC_FACTORS = np.array(
    [
        435e-6,
        599e-6,
        808e-6,
        781e-6,
        876e-6,
        548e-6,
        432e-6,
    ]
)

ELECTRON_MASS = 9.10938356e-31  # kg

# See doc string of calculate_phase_space_density() for more details.
VELOCITY_CONVERSION_FACTOR = 1.237e31
# See doc string of calculate_flux() for more details.
FLUX_CONVERSION_FACTOR = 6.187e30

CEM_DETECTORS_ANGLE = np.array([-63, -42, -21, 0, 21, 42, 63])

# ESA voltage and index in the final data table
ESA_VOLTAGE_ROW_INDEX_DICT = {
    0.56: 0,
    0.78: 1,
    1.08: 2,
    1.51: 3,
    2.10: 4,
    2.92: 5,
    4.06: 6,
    5.64: 7,
    7.85: 8,
    10.92: 9,
    15.19: 10,
    21.13: 11,
    29.39: 12,
    40.88: 13,
    56.87: 14,
    79.10: 15,
    110.03: 16,
    153.05: 17,
    212.89: 18,
    296.14: 19,
    411.93: 20,
    572.99: 21,
    797.03: 22,
    1108.66: 23,
}


class SWEAPID(IntEnum):
    """Create ENUM for apid."""

    SWE_SCIENCE = 1344


def read_lookup_table() -> pd.DataFrame:
    """
    Read lookup table.

    Returns
    -------
    esa_table : pandas.DataFrame
        ESA table.
    """
    # Read lookup table
    lookup_table_path = imap_module_directory / "swe/l1b/swe_esa_lookup_table.csv"
    esa_table = pd.read_csv(lookup_table_path)
    return esa_table


def combine_acquisition_time(
    acq_start_coarse: np.ndarray, acq_start_fine: np.ndarray
) -> npt.NDArray:
    """
    Combine acquisition time of each quarter cycle measurement.

    Each quarter cycle data should have same acquisition start time coarse
    and fine value. We will use that as base time to calculate each
    acquisition time for each count data.
    base_quarter_cycle_acq_time = acq_start_coarse +
    |                            acq_start_fine / 1000000

    Parameters
    ----------
    acq_start_coarse : np.ndarray
        Acq start coarse. It is in seconds.
    acq_start_fine : np.ndarray
        Acq start fine. It is in microseconds.

    Returns
    -------
    np.ndarray
        Acquisition time in seconds.
    """
    epoch_center_time = acq_start_coarse + (acq_start_fine / MICROSECONDS_IN_SECOND)
    return epoch_center_time


def calculate_data_acquisition_time(
    acq_start_time: np.ndarray,
    esa_step_number: int,
    acq_duration: int,
    settle_duration: int,
) -> npt.NDArray:
    """
    Calculate center acquisition time of each science data point.

    Center acquisition time (in seconds) of each count data
    point at each energy and at angle step will be
    calculated using this formula:
    |    each_count_acq_time = acq_start_time +
    |           (step * ( acq_duration + settle_duration) / 1000000 )
    where 'step' goes from 0 to 179, acq_start_time is in seconds and
    settle_duration and acq_duration are in microseconds.

    To calculate center time of data acquisition time, we will add
    |    each_count_acq_time + (acq_duration / 1000000) / 2

    Parameters
    ----------
    acq_start_time : np.ndarray
        Start acquisition time in seconds.
    esa_step_number : int
        Energy step.
    acq_duration : int
        Acquisition duration in microseconds.
    settle_duration : int
        Settle duration in microseconds.

    Returns
    -------
    esa_step_number_acq_time : np.ndarray
        ESA step number acquisition center time in seconds.
    """
    # Calculate time for each ESA step
    esa_step_number_acq_time = (
        acq_start_time
        + (esa_step_number * (acq_duration + settle_duration) / MICROSECONDS_IN_SECOND)
        + (acq_duration / MICROSECONDS_IN_SECOND) / 2
    )

    return esa_step_number_acq_time
