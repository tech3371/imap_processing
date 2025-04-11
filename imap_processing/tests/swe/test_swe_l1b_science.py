import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.swe.l1a.swe_l1a import swe_l1a
from imap_processing.swe.l1a.swe_science import swe_science
from imap_processing.swe.l1b.swe_l1b_science import (
    apply_in_flight_calibration,
    convert_counts_to_rate,
    deadtime_correction,
    get_indices_of_full_cycles,
)
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


@pytest.mark.skip(reason="Need to fix it")
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
        imap_module_directory
        / "tests/swe/lut/imap_swe_l1b-in-flight-cal_20240510_20260716_v000.csv"
    ]
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
        apply_in_flight_calibration(
            one_full_cycle_data, acquisition_time, in_flight_cal_files
        )


def test_swe_l1b(decom_test_data_derived):
    """Test that calculate engineering unit(EU) matches validation data.

    Parameters
    ----------
    decom_test_data_derived : xarray.dataset
        Dataset with derived values
    """
    science_l1a_ds = swe_science(decom_test_data_derived)

    # read science validation data
    test_data_path = imap_module_directory / "tests/swe/l0_validation_data"
    eu_validation_data = pd.read_csv(
        test_data_path / "idle_export_eu.SWE_SCIENCE_20240510_092742.csv",
        index_col="SHCOARSE",
    )

    second_data = science_l1a_ds.isel(epoch=1)
    validation_data = eu_validation_data.loc[second_data["shcoarse"].values]

    science_eu_field_list = [
        "SPIN_PHASE",
        "SPIN_PERIOD",
        "THRESHOLD_DAC",
    ]

    # Test EU values for science data
    for field in science_eu_field_list:
        np.testing.assert_almost_equal(
            second_data[field.lower()].values, validation_data[field], decimal=5
        )


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_swe_l1b_science(mock_get_file_paths, l1b_validation_df):
    """Test that CDF file is created and has the correct name."""
    test_data_path = "tests/swe/l0_data/2024051010_SWE_SCIENCE_packet.bin"
    l1a_datasets = swe_l1a(imap_module_directory / test_data_path)

    l1b_input = l1a_datasets[0]

    # Set these two as cli.py -> post_processing would have.
    l1b_input.attrs["Data_version"] = "v002"
    # write data to CDF
    l1a_cdf_filepath = write_cdf(l1b_input)
    assert l1a_cdf_filepath.name == "imap_swe_l1a_sci_20240510_v002.cdf"

    def get_file_paths_side_effect(descriptor):
        if descriptor == "sci":
            return [l1a_cdf_filepath]
        elif descriptor == "l1b-in-flight-cal":
            return [
                imap_module_directory
                / "tests/swe/lut/imap_swe_l1b-in-flight-cal_20240510_20260716_v000.csv"
            ]
        elif descriptor == "eu-conversion":
            return [
                imap_module_directory
                / "tests/swe/lut/imap_swe_eu-conversion_20240510_v000.csv"
            ]
        elif descriptor == "esa-lut":
            return [
                imap_module_directory
                / "tests/swe/lut/imap_swe_esa-lut_20250301_v000.csv"
            ]
        else:
            raise ValueError(f"Unknown descriptor: {descriptor}")

    mock_get_file_paths.side_effect = get_file_paths_side_effect
    dependencies = [
        {"type": "science", "files": [l1a_cdf_filepath.name]},
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
    l1b_dataset = swe_l1b_science(json.dumps(dependencies))
    # Injected during cli processing calls, not instrument functions,
    # so set it manually here before writing
    l1b_dataset.attrs["Data_version"] = "v002"

    sci_l1b_filepath = write_cdf(l1b_dataset)

    assert sci_l1b_filepath.name == "imap_swe_l1b_sci_20240510_v002.cdf"
    # load the CDF file and compare the values
    l1b_cdf_dataset = load_cdf(sci_l1b_filepath)
    processed_science = l1b_cdf_dataset["science_data"].data
    validation_science = l1b_validation_df.values[:, 1:].reshape(6, 24, 30, 7)
    np.testing.assert_allclose(processed_science, validation_science, rtol=1e-7)


def test_count_rate():
    x = np.array([1, 10, 100, 1000, 10000, 38911, 65535])
    acq_duration = np.array([80000])
    deatime_corrected = deadtime_correction(x, acq_duration)
    count_rate = convert_counts_to_rate(deatime_corrected, acq_duration)
    count_rate = count_rate.flatten()  # Ensure the shape matches expected_output
    # Ruth provided the expected output for this test
    expected_output = [
        12.50005653,
        125.00562805,
        1250.56278121,
        12556.50455087,
        130890.05519127,
        589631.73670132,
        1161815.68783304,
    ]
    np.testing.assert_allclose(count_rate, expected_output, rtol=1e-7)
