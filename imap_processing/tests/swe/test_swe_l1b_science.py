from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from imap_processing import imap_module_directory
from imap_processing.swe.l1a.swe_science import swe_science
from imap_processing.swe.l1b.swe_l1b_science import (
    apply_in_flight_calibration,
    get_checker_board_pattern,
    get_indices_of_full_cycles,
    swe_l1b_science,
)
from imap_processing.swe.l1a.swe_l1a import swe_l1a
from imap_processing.swe.utils import swe_constants


@pytest.fixture(scope="session")
def l1a_test_data(decom_test_data):
    """Read test data from file and process to l1a"""
    processed_data = swe_science(decom_test_data)
    return processed_data


def test_get_full_cycle_data_indices():
    q = np.array([0, 1, 2, 0, 1, 2, 3, 2, 3, 0, 2, 3, 0, 1, 2, 3, 2, 3, 1, 0])
    filtered_q = get_indices_of_full_cycles(q)
    np.testing.assert_array_equal(filtered_q, np.array([3, 4, 5, 6, 12, 13, 14, 15]))

    q = np.array([0, 1, 0, 1, 2, 3, 0, 2])
    filtered_q = get_indices_of_full_cycles(q)
    np.testing.assert_array_equal(filtered_q, np.array([2, 3, 4, 5]))

    q = np.array([0, 1, 2, 3])
    filtered_q = get_indices_of_full_cycles(q)
    np.testing.assert_array_equal(filtered_q, np.array([0, 1, 2, 3]))

    q = np.array([1, 2])
    filtered_q = get_indices_of_full_cycles(q)
    np.testing.assert_array_equal(filtered_q, np.array([]))


@patch(
    "imap_processing.swe.l1b.swe_l1b_science.read_in_flight_cal_data",
    return_value=pd.DataFrame(
        {
            "met_time": [453051300, 453051900],
            "cem1": [1, 2],
            "cem2": [1, 2],
            "cem3": [1, 2],
            "cem4": [1, 2],
            "cem5": [1, 2],
            "cem6": [1, 2],
            "cem7": [1, 2],
        }
    ),
)
def test_in_flight_calibration_factor(mock_read_in_flight_cal_data, l1a_test_data):
    """Test that the L1B processing is working as expected."""
    # create sample data

    input_time = 453051355.0
    input_count = 19967
    one_full_cycle_data = np.full(
        (
            swe_constants.N_ESA_STEPS,
            swe_constants.N_ANGLE_SECTORS,
            swe_constants.N_CEMS,
        ),
        input_count,
    )
    acquisition_time = np.full(
        (swe_constants.N_ESA_STEPS, swe_constants.N_ANGLE_SECTORS), input_time
    )

    # Test that calibration factor is within correct range given test data
    expected_cal_factor = 1 + ((2 - 1) / (453051900 - 453051300)) * (
        input_time - 453051300
    )

    calibrated_count = apply_in_flight_calibration(
        one_full_cycle_data,
        acquisition_time,
    )

    np.testing.assert_allclose(
        calibrated_count,
        np.full(
            (
                swe_constants.N_ESA_STEPS,
                swe_constants.N_ANGLE_SECTORS,
                swe_constants.N_CEMS,
            ),
            input_count * expected_cal_factor,
        ),
        rtol=1e-9,
    )

    # Check for value outside of calibration time range
    input_time = 1.0
    acquisition_time = np.full(
        (swe_constants.N_ESA_STEPS, swe_constants.N_ANGLE_SECTORS), input_time
    )

    with pytest.raises(ValueError, match="Acquisition min/max times: "):
        apply_in_flight_calibration(one_full_cycle_data, acquisition_time)

@patch(
    "imap_processing.swe.l1b.swe_l1b_science.read_in_flight_cal_data",
    return_value=pd.DataFrame(
        {
            "met_time": [452051300, 454051900],
            "cem1": [1, 1],
            "cem2": [1, 1],
            "cem3": [1, 1],
            "cem4": [1, 1],
            "cem5": [1, 1],
            "cem6": [1, 1],
            "cem7": [1, 1],
        }
    ),
)
def test_get_checkerboard_patter(mock_read_in_flight_cal_data):
    """Test that the checkerboard pattern is correct."""
    esa_lut_path = (
        imap_module_directory / "tests/swe/lut/imap_swe_esa-lut_20250301_v000.csv"
    )
    esa_lut_df = pd.read_csv(esa_lut_path)
    checkerboard_indices = get_checker_board_pattern(esa_lut_df)
    # print(checkerboard_indices)
    test_data_path = "tests/swe/l0_data/2024051010_SWE_SCIENCE_packet.bin"
    l1a_datasets = swe_l1a(imap_module_directory / test_data_path, "002")
    swe_l1b_science(l1a_datasets[0], "002", esa_lut_df)
