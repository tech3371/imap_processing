"""Test coverage for imap_processing.hi.l1c.hi_l1c.py"""

from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.hi.l1a.science_direct_event import DE_CLOCK_TICK_S
from imap_processing.hi.l1c import hi_l1c
from imap_processing.hi.l1c.hi_l1c import CalibrationProductConfig
from imap_processing.hi.utils import HIAPID, CoincidenceBitmap


@pytest.fixture(scope="module")
def hi_test_cal_prod_config_path(hi_l1_test_data_path):
    return (
        hi_l1_test_data_path / "imap_his_pset-calibration-prod-config_20240101_v001.csv"
    )


@mock.patch("imap_processing.hi.l1c.hi_l1c.generate_pset_dataset")
def test_hi_l1c(mock_generate_pset_dataset, hi_test_cal_prod_config_path):
    """Test coverage for hi_l1c function"""
    mock_generate_pset_dataset.return_value = xr.Dataset(attrs={"Data_version": None})
    pset = hi_l1c.hi_l1c(
        [xr.Dataset(), hi_test_cal_prod_config_path], data_version="99"
    )
    assert pset.attrs["Data_version"] == "99"


def test_hi_l1c_not_implemented():
    """Test coverage for hi_l1c function with unrecognized dependencies"""
    with pytest.raises(NotImplementedError):
        hi_l1c.hi_l1c([None, None], "0")


@pytest.mark.external_kernel()
@pytest.mark.use_test_metakernel("imap_ena_sim_metakernel.template")
def test_generate_pset_dataset(hi_l1_test_data_path, hi_test_cal_prod_config_path):
    """Test coverage for generate_pset_dataset function"""
    l1b_de_path = hi_l1_test_data_path / "imap_hi_l1b_45sensor-de_20250415_v999.cdf"
    l1b_dataset = load_cdf(l1b_de_path)
    l1c_dataset = hi_l1c.generate_pset_dataset(
        l1b_dataset, hi_test_cal_prod_config_path
    )

    assert l1c_dataset.epoch.data[0] == np.mean(l1b_dataset.epoch.data[[0, -1]]).astype(
        np.int64
    )

    np.testing.assert_array_equal(l1c_dataset.despun_z.data.shape, (1, 3))
    np.testing.assert_array_equal(l1c_dataset.hae_latitude.data.shape, (1, 3600))
    np.testing.assert_array_equal(l1c_dataset.hae_longitude.data.shape, (1, 3600))
    for var in [
        "counts",
        "exposure_times",
        "background_rates",
        "background_rates_uncertainty",
    ]:
        np.testing.assert_array_equal(l1c_dataset[var].data.shape, (1, 9, 2, 3600))

    # Test ISTP compliance by writing CDF
    l1c_dataset.attrs["Data_version"] = 1
    write_cdf(l1c_dataset)


def test_empty_pset_dataset():
    """Test coverage for empty_pset_dataset function"""
    n_energy_steps = 8
    l1b_esa_energy_steps = np.arange(n_energy_steps + 1).repeat(2)
    n_calibration_prods = 5
    sensor_str = HIAPID.H90_SCI_DE.sensor
    dataset = hi_l1c.empty_pset_dataset(
        l1b_esa_energy_steps, n_calibration_prods, sensor_str
    )

    assert dataset.epoch.size == 1
    assert dataset.spin_angle_bin.size == 3600
    assert dataset.esa_energy_step.size == n_energy_steps
    np.testing.assert_array_equal(
        dataset.esa_energy_step.data, np.arange(n_energy_steps) + 1
    )
    assert dataset.calibration_prod.size == n_calibration_prods

    # verify that attrs defined in hi_pset_epoch have overwritten default
    # epoch attributes
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs("hi")
    attr_mgr.add_instrument_variable_attrs(instrument="hi", level=None)
    pset_epoch_attrs = attr_mgr.get_variable_attributes(
        "hi_pset_epoch", check_schema=False
    )
    for k, v in pset_epoch_attrs.items():
        assert k in dataset.epoch.attrs
        assert dataset.epoch.attrs[k] == v


@pytest.mark.parametrize("sensor_str", ["90sensor", "45sensor"])
@mock.patch("imap_processing.spice.geometry.frame_transform")
@mock.patch("imap_processing.hi.l1c.hi_l1c.frame_transform")
def test_pset_geometry(mock_frame_transform, mock_geom_frame_transform, sensor_str):
    """Test coverage for pset_geometry function"""
    # pset_geometry uses both frame_transform and frame_transform_az_el. By mocking
    # the frame_transform imported into hi_l1c as well as the geometry.frame_transform
    # the underlying need for SPICE kernels is remove. Mock them both to just return
    # the input position vectors.
    mock_frame_transform.side_effect = lambda et, pos, from_frame, to_frame: pos
    mock_geom_frame_transform.side_effect = lambda et, pos, from_frame, to_frame: pos

    geometry_vars = hi_l1c.pset_geometry(0, sensor_str)

    assert "despun_z" in geometry_vars
    np.testing.assert_array_equal(geometry_vars["despun_z"].data, [[0, 0, 1]])

    assert "hae_latitude" in geometry_vars
    assert "hae_longitude" in geometry_vars
    # frame_transform is mocked to return the input vectors. For Hi-90, we
    # expect hae_latitude to be 0, and for Hi-45 we expect -45. Both sensors
    # have an expected longitude to be 0.1 degree steps starting at 0.05
    expected_latitude = 0 if sensor_str == "90sensor" else -45
    np.testing.assert_array_equal(
        geometry_vars["hae_latitude"].data, np.full((1, 3600), expected_latitude)
    )
    np.testing.assert_allclose(
        geometry_vars["hae_longitude"].data,
        np.arange(0.05, 360, 0.1, dtype=np.float32).reshape((1, 3600)),
        atol=4e-05,
    )


def test_find_second_de_packet_data():
    """Test coverage for find_second_de_packet_data function"""
    # Create a test l1b_dataset
    # Expect to remove index 0 and 5 due to missing esa_step pair
    # Expect to remove index 11 due to 0 being a calibration step
    # Expect to return indices 2, 4, 7, 9, 13
    esa_steps = np.array([1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 0, 0, 7, 7])
    l1b_dataset = xr.Dataset(
        coords={
            "epoch": xr.DataArray(
                np.arange(esa_steps.size),
                dims=["epoch"],
            )
        },
        data_vars={
            "esa_step": xr.DataArray(
                esa_steps,
                dims=["epoch"],
            )
        },
    )
    subset = hi_l1c.find_second_de_packet_data(l1b_dataset)
    np.testing.assert_array_equal(subset.epoch.data, np.array([2, 4, 7, 9, 13]))


@pytest.fixture(scope="module")
def fake_spin_df():
    """Generate a synthetic spin dataframe"""
    # Generate some spin periods that vary by a random fraction of a second
    spin_period = np.full(10, 15) + np.random.randn(10) / 10
    d = {
        "spin_start_time": np.add.accumulate(spin_period),
        "spin_period_sec": spin_period,
    }
    spin_df = pd.DataFrame.from_dict(d)
    return spin_df


def test_get_de_clock_ticks_for_esa_step(fake_spin_df):
    """Test coverage for get_de_clock_ticks_for_esa_step function."""

    # Test nominal cases where CCSDS met falls after 8th spin start and before
    # the end spin in the table + 1/2 spin period
    for _, spin_row in fake_spin_df.iloc[8:].iterrows():
        for ccsds_met in np.linspace(
            spin_row.spin_start_time,
            spin_row.spin_start_time + np.floor(spin_row.spin_period_sec / 2),
            10,
        ):
            clock_tick_mets, clock_tick_weights = (
                hi_l1c.get_de_clock_ticks_for_esa_step(ccsds_met, fake_spin_df)
            )
            np.testing.assert_array_equal(clock_tick_mets.shape, clock_tick_mets.shape)
            # Verify last weight entry
            exp_final_weight = (
                np.absolute(
                    fake_spin_df.spin_start_time.to_numpy() - clock_tick_mets[-1]
                ).min()
                / DE_CLOCK_TICK_S
            )
            assert clock_tick_weights[-1] == exp_final_weight
            assert np.all(clock_tick_weights[:-1] == 1)


def test_get_de_clock_ticks_for_esa_step_exceptions(fake_spin_df):
    """Test the exception logic in the get_de_clock_ticks_for_esa_step function."""
    # Test the ccsds_met being > 1/2 spin period past the spin start
    bad_ccsds_met = (
        fake_spin_df.iloc[8].spin_start_time
        + fake_spin_df.iloc[8].spin_period_sec / 2
        + 0.1
    )
    with pytest.raises(
        ValueError, match="The difference between ccsds_met and spin_start_met"
    ):
        hi_l1c.get_de_clock_ticks_for_esa_step(bad_ccsds_met, fake_spin_df)

    # Test the ccsds_met being too close to the start of the spin table
    bad_ccsds_met = fake_spin_df.iloc[7].spin_start_time
    with pytest.raises(
        ValueError, match="Error determining start/end time for exposure time"
    ):
        hi_l1c.get_de_clock_ticks_for_esa_step(bad_ccsds_met, fake_spin_df)


class TestCalibrationProductConfig:
    """
    All test coverage for the pd.DataFrame accessor extension "cal_prod_config".
    """

    def test_wrong_columns(self):
        """Test coverage for a dataframe with the wrong columns."""
        required_columns = CalibrationProductConfig.required_columns
        for exclude_column_name in required_columns:
            include_columns = set(required_columns) - {exclude_column_name}
            df = pd.DataFrame({col: [1, 2, 3] for col in include_columns})
            with pytest.raises(AttributeError, match="Required column*"):
                _ = df.cal_prod_config.number_of_products

    def test_from_csv(self, hi_test_cal_prod_config_path):
        """Test coverage for read_csv function."""
        df = CalibrationProductConfig.from_csv(hi_test_cal_prod_config_path)
        assert isinstance(df["coincidence_type_list"][0, 1], list)

    def test_added_coincidence_type_values_column(self, hi_test_cal_prod_config_path):
        df = CalibrationProductConfig.from_csv(hi_test_cal_prod_config_path)
        assert "coincidence_type_values" in df.columns
        for _, row in df.iterrows():
            for detect_string, val in zip(
                row["coincidence_type_list"], row["coincidence_type_values"]
            ):
                assert val == CoincidenceBitmap.detector_hit_str_to_int(detect_string)

    def test_number_of_products(self, hi_test_cal_prod_config_path):
        """Test coverage for number of products accessor."""
        df = CalibrationProductConfig.from_csv(hi_test_cal_prod_config_path)
        assert df.cal_prod_config.number_of_products == 2
