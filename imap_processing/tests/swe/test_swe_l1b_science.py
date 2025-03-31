import numpy as np
import pandas as pd
import pytest

from imap_processing.swe.l1a.swe_science import swe_science
from imap_processing.swe.l1b.swe_l1b_science import (
    apply_in_flight_calibration,
    get_indices_of_full_cycles,
)
from imap_processing.swe.utils import swe_constants


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


def test_in_flight_calibration_factor(l1a_test_data, tmp_path):
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
    # TODO: change this in future PR to pass the dataframe instead.
    cal_df = pd.DataFrame(
        [[453051300, 1, 1, 1, 1, 1, 1, 1], [453051900, 2, 2, 2, 2, 2, 2, 2]]
    )
    cal_file = tmp_path / "calibration_data.csv"
    cal_df.to_csv(cal_file, index=False, header=False)

    calibrated_count = apply_in_flight_calibration(
        one_full_cycle_data,
        acquisition_time,
        cal_file,
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
        apply_in_flight_calibration(one_full_cycle_data, acquisition_time, cal_file)
