from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import json

from imap_processing.swe.l1a.swe_science import swe_science
from imap_processing.swe.l1b.swe_l1b_science import (
    apply_in_flight_calibration,
    get_indices_of_full_cycles,
    swe_l1b_science,
)
from imap_processing.swe.utils import swe_constants
from imap_processing import imap_module_directory

@pytest.fixture(scope="session")
def l1a_test_data(decom_test_data):
    """Read test data from file and process to l1a"""
    processed_data = swe_science(decom_test_data, "001")
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


def test_in_flight_calibration_factor(l1a_test_data):
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

    in_flight_cal_files = [
        imap_module_directory / "tests/swe/lut/imap_swe_l1b-in-flight-cal_20240510_20260716_v000.csv"
    ]
    calibrated_count = apply_in_flight_calibration(
        one_full_cycle_data,
        acquisition_time,
        in_flight_cal_files,
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
        apply_in_flight_calibration(one_full_cycle_data, acquisition_time, in_flight_cal_files)


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_swe_science(mock_get_file_paths):
    """Test that the checkerboard pattern is correct."""
    def get_file_paths_side_effect(descriptor):
        if descriptor == "sci":
            return [imap_module_directory / "tests/swe/data/imap_swe_l1a_sci_20260924_v001.cdf"]
        elif descriptor == "l1b-in-flight-cal":
            return [imap_module_directory / "tests/swe/lut/imap_swe_l1b-in-flight-cal_20240510_20260716_v000.csv"]
        elif descriptor == "eu-conversion":
            return [imap_module_directory / "tests/swe/lut/imap_swe_eu-conversion_20240510_v000.csv"]
        elif descriptor == "esa-lut":
            return [
                imap_module_directory / "tests/swe/lut/imap_swe_esa-lut_20250301_v000.csv"
            ]
        else:
            raise ValueError(f"Unknown descriptor: {descriptor}")

    mock_get_file_paths.side_effect = get_file_paths_side_effect

    dependencies = [
        {"type": "science", "files": ["imap_swe_l1a_sci_20260924_v001.cdf"]},
        {
            "type": "ancillary",
            "files": [
                "imap_swe_l1b-in-flight-cal_20240510_20260716_v000.csv",
            ],
        },
        {
            "type": "ancillary",
            "files": [
                "imap_swe_eu-conversion_20240510_v000.csv",
            ],
        },
    ]
    swe_l1b_science(json.dumps(dependencies))
