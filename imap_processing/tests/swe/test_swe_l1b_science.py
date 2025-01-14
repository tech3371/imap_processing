from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from imap_processing.swe.l1a.swe_science import swe_science
from imap_processing.swe.l1b.swe_l1b_science import (
    apply_in_flight_calibration,
    calculate_calibration_factor,
    get_indices_of_full_cycles,
)


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
    input_time = l1a_test_data["shcoarse"].data[0]  # 453051308

    cal_factor = calculate_calibration_factor(input_time)
    # Test that calibration factor is within correct range given test data
    expected_cal_factor = 1 + (2 - 1) / (453051900 - 453051300) * (
        input_time - 453051300
    )
    np.testing.assert_array_equal(cal_factor, np.repeat(expected_cal_factor, 7))
    assert cal_factor.shape == (7,)

    # Test that applying calibration factor works as expected
    # Picking non-zero data to test
    counts_data = l1a_test_data["science_data"].data[15]  # 19967
    l1b_data = apply_in_flight_calibration(counts_data, input_time)
    assert l1b_data.shape == (180, 7)
    # Since data is zero in some index and 19967 in some index,
    # we can't just multiply by expected_cal_factor to get expected result.
    # Instead, we will just check that the first index is correct.
    assert l1b_data[0][1] == 19967 * expected_cal_factor
