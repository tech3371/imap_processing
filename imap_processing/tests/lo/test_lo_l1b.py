from collections import namedtuple

import numpy as np
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import load_cdf
from imap_processing.lo.l1b.lo_l1b import (
    calculate_tof1_for_golden_triples,
    convert_start_end_acq_times,
    create_datasets,
    get_avg_spin_durations,
    get_spin_angle,
    get_spin_start_times,
    initialize_l1b_de,
    lo_l1b,
    set_coincidence_type,
    set_each_event_epoch,
    set_event_met,
    set_spin_bin,
    set_spin_cycle,
)
from imap_processing.spice.time import met_to_ttj2000ns


@pytest.fixture
def dependencies():
    return {
        "imap_lo_l1a_de": load_cdf(
            imap_module_directory
            / "tests/lo/test_cdfs/imap_lo_l1a_de_20241022_v002.cdf"
        ),
        "imap_lo_l1a_spin": load_cdf(
            imap_module_directory
            / "tests/lo/test_cdfs/imap_lo_l1a_spin_20241022_v002.cdf"
        ),
    }


@pytest.fixture
def attr_mgr_l1b():
    attr_mgr_l1b = ImapCdfAttributes()
    attr_mgr_l1b.add_instrument_global_attrs(instrument="lo")
    attr_mgr_l1b.add_instrument_variable_attrs(instrument="lo", level="l1b")
    return attr_mgr_l1b


@pytest.fixture
def attr_mgr_l1a():
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs(instrument="lo")
    attr_mgr.add_instrument_variable_attrs(instrument="lo", level="l1a")
    return attr_mgr


def test_lo_l1b():
    # Arrange
    de_file = (
        imap_module_directory / "tests/lo/test_cdfs/imap_lo_l1a_de_20241022_v002.cdf"
    )
    spin_file = (
        imap_module_directory / "tests/lo/test_cdfs/imap_lo_l1a_spin_20241022_v002.cdf"
    )
    data = {}
    for file in [de_file, spin_file]:
        dataset = load_cdf(file)
        data[dataset.attrs["Logical_source"]] = dataset

    expected_logical_source = "imap_lo_l1b_de"
    # Act
    output_file = lo_l1b(data)

    # Assert
    assert expected_logical_source == output_file[0].attrs["Logical_source"]


def test_create_datasets():
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs(instrument="lo")
    attr_mgr.add_instrument_variable_attrs(instrument="lo", level="l1b")

    logical_source = "imap_lo_l1b_de"

    data_field_tup = namedtuple("data_field_tup", ["name"])
    data_fields = [
        data_field_tup("ESA_STEP"),
        data_field_tup("MODE"),
        data_field_tup("TOF0"),
        data_field_tup("TOF1"),
        data_field_tup("TOF2"),
        data_field_tup("TOF3"),
        data_field_tup("COINCIDENCE_TYPE"),
        data_field_tup("POS"),
        data_field_tup("COINCIDENCE"),
        data_field_tup("BADTIME"),
        data_field_tup("DIRECTION"),
    ]

    dataset = create_datasets(attr_mgr, logical_source, data_fields)

    assert len(dataset.tof0.shape) == 1
    assert dataset.tof0.shape[0] == 3
    assert len(dataset.tof1.shape) == 1
    assert dataset.tof1.shape[0] == 3
    assert len(dataset.tof2.shape) == 1
    assert dataset.tof2.shape[0] == 3
    assert len(dataset.tof3.shape) == 1
    assert dataset.tof3.shape[0] == 3
    assert len(dataset.mode.shape) == 1
    assert dataset.mode.shape[0] == 3
    assert len(dataset.coincidence_type.shape) == 1
    assert dataset.coincidence_type.shape[0] == 3
    assert len(dataset.pos.shape) == 1
    assert dataset.pos.shape[0] == 3
    assert len(dataset.direction.shape) == 2
    assert dataset.direction.shape[0] == 3
    assert dataset.direction.shape[1] == 3
    assert len(dataset.badtime.shape) == 1
    assert dataset.badtime.shape[0] == 3
    assert len(dataset.esa_step.shape) == 1
    assert dataset.esa_step.shape[0] == 3


def test_initialize_dataset(dependencies, attr_mgr_l1b):
    # Arrange
    l1a_de = dependencies["imap_lo_l1a_de"]
    logical_source = "imap_lo_l1b_de"

    # Act
    l1b_de = initialize_l1b_de(l1a_de, attr_mgr_l1b, logical_source)

    # Assert
    assert l1b_de.attrs["Logical_source"] == logical_source
    assert list(l1b_de.coords.keys()) == []
    assert len(l1b_de.data_vars) == 4
    assert len(l1b_de.coords) == 0
    for l1b_name, l1a_name in {
        "pos": "pos",
        "mode": "mode",
        "absent": "coincidence_type",
        "esa_step": "esa_step",
    }.items():
        assert l1b_name in l1b_de.data_vars
        np.testing.assert_array_equal(l1b_de[l1b_name], l1a_de[l1a_name])


def test_convert_start_end_acq_times():
    # Arrange
    spin = xr.Dataset(
        {
            "acq_start_sec": ("epoch", [1, 2, 3]),
            "acq_start_subsec": ("epoch", [4, 5, 6]),
            "acq_end_sec": ("epoch", [7, 8, 9]),
            "acq_end_subsec": ("epoch", [10, 11, 12]),
        },
        coords={"epoch": [0, 1, 2]},
    )

    acq_start_expected = xr.DataArray(
        [
            spin["acq_start_sec"][0] + spin["acq_start_subsec"][0] * 1e-6,
            spin["acq_start_sec"][1] + spin["acq_start_subsec"][1] * 1e-6,
            spin["acq_start_sec"][2] + spin["acq_start_subsec"][2] * 1e-6,
        ],
        dims="epoch",
    )
    acq_end_expected = xr.DataArray(
        [
            spin["acq_end_sec"][0] + spin["acq_end_subsec"][0] * 1e-6,
            spin["acq_end_sec"][1] + spin["acq_end_subsec"][1] * 1e-6,
            spin["acq_end_sec"][2] + spin["acq_end_subsec"][2] * 1e-6,
        ],
        dims="epoch",
    )

    # Act
    acq_start, acq_end = convert_start_end_acq_times(spin)

    # Assert
    np.testing.assert_array_equal(acq_start.values, acq_start_expected.values)
    np.testing.assert_array_equal(acq_end.values, acq_end_expected.values)


def test_get_avg_spin_durations():
    # Arrange
    acq_start = xr.DataArray([0, 423, 846.2], dims="epoch")
    acq_end = xr.DataArray([422.8, 846, 1269.7], dims="epoch")
    expected_avg_spin_durations = np.array([422.8, 423, 423.5]) / 28

    # Act
    avg_spin_durations = get_avg_spin_durations(acq_start, acq_end)

    # Assert
    np.testing.assert_array_equal(avg_spin_durations, expected_avg_spin_durations)


def test_get_spin_angle():
    # Arrange
    de = xr.Dataset(
        {
            "de_count": ("epoch", [2, 3]),
            "de_time": ("direct_event", [0000, 1000, 2000, 3000, 4000]),
        },
        coords={"epoch": [0, 1], "direct_event": [0, 1, 2, 3, 4]},
    )
    spin_angle_expected = np.array([0, 87.89, 175.78, 263.67, 351.56])

    # Act
    spin_angle = get_spin_angle(de)

    # Assert
    np.testing.assert_allclose(
        spin_angle,
        spin_angle_expected,
        atol=1e-2,
    )


def test_spin_bin():
    # Arrange
    l1b_de = xr.Dataset()
    spin_angle = np.array([0, 50, 150, 250, 365])
    expected_spin_bins = np.array([0, 8, 25, 41, 60])

    # Act
    l1b_de = set_spin_bin(l1b_de, spin_angle)

    # Assert
    np.testing.assert_array_equal(l1b_de["spin_bin"], expected_spin_bins)


def test_spin_cycle():
    # Arrange
    de = xr.Dataset(
        {
            "de_count": ("epoch", [2, 3]),
            "esa_step": ("direct_event", [1, 2, 3, 4, 5]),
        },
        coords={"epoch": [0, 1], "direct_event": [1, 2, 3, 4, 5]},
    )

    # spin_cycle = spin_start + 7 + (esa_step - 1) * 2
    # where spin start is the spin number for the first spin
    # in an Aggregated Science Cycle (first spin number of an epoch)
    # and esa_step is the esa_step for a direct event
    spin_cycle_expected = np.array([7, 9, 39, 41, 43])
    spin_cycle_data = xr.Dataset()

    # Act
    spin_cycle_data = set_spin_cycle(de, spin_cycle_data)

    # Assert
    np.testing.assert_array_equal(spin_cycle_data["spin_cycle"], spin_cycle_expected)


def test_get_spin_start_times():
    # Arrange
    l1b_de = xr.Dataset(
        {
            "spin_cycle": ("epoch", [0, 1, 2, 3, 4]),
        },
        coords={
            "epoch": [
                0,
                1,
                2,
                3,
                4,
            ]
        },
    )
    l1a_de = xr.Dataset(
        {
            "de_count": ("epoch", [2, 3]),
            "met": ("direct_event", [0, 1, 2, 3, 4]),
            "de_time": ("direct_event", [0000, 1000, 2000, 3000, 4000]),
        },
        coords={"epoch": [0, 1], "direct_event": [0, 1, 2, 3, 4]},
    )
    spin = xr.Dataset(
        {
            "start_sec_spin": (
                ["epoch", "spin"],
                [[20, 25, 30, 35, 40], [45, 50, 55, 60, 65]],
            ),
            "start_subsec_spin": (
                ["epoch", "spin"],
                [[2000, 3000, 4000, 5000, 6000], [1000, 1500, 2000, 3000, 4000]],
            ),
        }
    )

    end_acq = xr.DataArray([0, 1], dims="epoch")
    spin_start_times_expected = np.array([20.002, 50.0015, 55.002, 60.003, 65.004])
    spin_start_times = get_spin_start_times(l1a_de, l1b_de, spin, end_acq)

    np.testing.assert_allclose(
        spin_start_times,
        spin_start_times_expected,
        atol=1e-4,
    )


def test_set_event_met():
    # Arrange
    l1b_de = xr.Dataset()
    l1a_de = xr.Dataset(
        {
            "de_count": ("epoch", [2, 3]),
            "de_time": ("direct_event", [0000, 1000, 2000, 3000, 4000]),
        },
        coords={
            "epoch": [0, 1],
            "direct_event": [
                0,
                1,
                2,
                3,
                4,
            ],
        },
    )
    avg_spin_durations = xr.DataArray([5, 10])
    spin_start_times = xr.DataArray([10, 20, 30, 40, 50])
    expected_event_met = np.array([10, 21.2207, 34.8828, 47.3242, 59.7656])

    # Act
    l1b_de = set_event_met(l1a_de, l1b_de, spin_start_times, avg_spin_durations)

    # Assert
    np.testing.assert_allclose(
        l1b_de["event_met"].values,
        expected_event_met,
        atol=1e-4,
    )

    def test_set_each_event_epoch():
        l1b_de = xr.Dataset(
            {
                "event_met": ("epoch", [10, 20, 30, 40, 50]),
            },
            coords={
                "epoch": [0, 1, 2, 3, 4],
            },
        )
        epoch_expected = met_to_ttj2000ns(np.array([10, 20, 30, 40, 50]))

        l1b_de = set_each_event_epoch(l1b_de)

        np.testing.assert_allclose(
            l1b_de["epoch"].values,
            epoch_expected,
            atol=1e-4,
        )


def test_calculate_tof1_for_golden_triples():
    # Arrange
    l1a_de = xr.Dataset(
        {
            "coincidence_type": ("epoch", [0, 0, 0]),
            "mode": ("epoch", [0, 0, 1]),
            "tof0": ("epoch", [2, 4, 2]),
            "tof1": ("epoch", [0, 0, 0]),
            "tof2": ("epoch", [2, 6, 2]),
            "tof3": ("epoch", [2, 8, 2]),
            "cksm": ("epoch", [2, 12, 2]),
        }
    )

    l1a_de_expected = xr.Dataset(
        {
            "coincidence_type": ("epoch", [0, 0, 0]),
            "mode": ("epoch", [0, 0, 1]),
            "tof0": ("epoch", [2, 4, 2]),
            "tof1": ("epoch", [42, 36, 0]),
            "tof2": ("epoch", [2, 6, 2]),
            "tof3": ("epoch", [2, 8, 2]),
            "cksm": ("epoch", [2, 12, 2]),
        }
    )

    # Act
    l1a_de = calculate_tof1_for_golden_triples(l1a_de)

    # Assert
    assert l1a_de_expected.equals(l1a_de)


def test_set_coincidence_type(attr_mgr_l1a):
    # Arrange
    l1b_de = xr.Dataset()
    tof_fill = attr_mgr_l1a.get_variable_attributes("tof0")["FILLVAL"]
    ckm_fill = attr_mgr_l1a.get_variable_attributes("cksm")["FILLVAL"]
    l1a_de = xr.Dataset(
        {
            "de_count": ("epoch", [3]),
            "coincidence_type": ("direct_event", [0, 0, 4]),
            "mode": ("direct_event", [1, 0, 1]),
            "tof0": ("direct_event", [5, 2, 10]),
            "tof1": ("direct_event", [10, 4, tof_fill]),
            "tof2": ("direct_event", [15, 6, 20]),
            "tof3": ("direct_event", [20, 8, 30]),
            "cksm": ("direct_event", [25, ckm_fill, ckm_fill]),
        }
    )

    coincidence_type_expected = np.array(["111111", "111100", "101101"])

    # Act
    l1b_de = set_coincidence_type(l1a_de, l1b_de, attr_mgr_l1a)

    # Assert
    np.testing.assert_array_equal(
        l1b_de["coincidence_type"].values,
        coincidence_type_expected,
    )
