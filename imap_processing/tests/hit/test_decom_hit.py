from pathlib import Path

import numpy as np
import pytest

from imap_processing import imap_module_directory
from imap_processing.hit.hit_utils import (
    HitAPID,
)
from imap_processing.hit.l0.constants import AZIMUTH_ANGLES, ZENITH_ANGLES
from imap_processing.hit.l0.decom_hit import (
    assemble_science_frames,
    decom_hit,
    decompress_rates_16_to_32,
    get_valid_starting_indices,
    is_sequential,
    parse_count_rates,
    parse_data,
    update_ccsds_header_dims,
)
from imap_processing.utils import packet_file_to_datasets


@pytest.fixture
def sci_dataset():
    """Create a xarray dataset for testing from sample data."""
    packet_definition = (
        imap_module_directory / "hit/packet_definitions/hit_packet_definitions.xml"
    )

    # L0 file path
    packet_file = Path(imap_module_directory / "tests/hit/test_data/sci_sample.ccsds")

    datasets_by_apid = packet_file_to_datasets(
        packet_file=packet_file,
        xtce_packet_definition=packet_definition,
        use_derived_value=False,
    )

    science_dataset = datasets_by_apid[HitAPID.HIT_SCIENCE]
    return science_dataset


def test_parse_data():
    """Test the parse_data function."""
    # Test parsing a single integer
    bin_str = "110"
    bits_per_index = 2
    start = 0
    end = 2
    result = parse_data(bin_str, bits_per_index, start, end)
    assert result == [3]  # 11 in binary is 3

    # Test parsing multiple integers
    bin_str = "110010101011"
    bits_per_index = 2
    start = 0
    end = 12
    result = parse_data(bin_str, bits_per_index, start, end)
    assert result == [3, 0, 2, 2, 2, 3]  # 11, 00, 10, 10, 10, 11 in binary


def test_parse_count_rates(sci_dataset):
    """Test the parse_count_rates function."""

    # Update ccsds header fields to use sc_tick as dimension
    sci_dataset = update_ccsds_header_dims(sci_dataset)

    # Group science packets into groups of 20
    sci_dataset = assemble_science_frames(sci_dataset)
    # Parse count rates and add to dataset
    parse_count_rates(sci_dataset)
    # Added count rate variables to dataset
    count_rate_vars = [
        "hdr_unit_num",
        "hdr_frame_version",
        "hdr_dynamic_threshold_state",
        "hdr_leak_conv",
        "hdr_heater_duty_cycle",
        "hdr_code_ok",
        "hdr_minute_cnt",
        "spare",
        "livetime_counter",
        "num_trig",
        "num_reject",
        "num_acc_w_pha",
        "num_acc_no_pha",
        "num_haz_trig",
        "num_haz_reject",
        "num_haz_acc_w_pha",
        "num_haz_acc_no_pha",
        "sngrates",
        "nread",
        "nhazard",
        "nadcstim",
        "nodd",
        "noddfix",
        "nmulti",
        "nmultifix",
        "nbadtraj",
        "nl2",
        "nl3",
        "nl4",
        "npen",
        "nformat",
        "naside",
        "nbside",
        "nerror",
        "nbadtags",
        "coinrates",
        "pbufrates",
        "l2fgrates",
        "l2bgrates",
        "l3fgrates",
        "l3bgrates",
        "penfgrates",
        "penbgrates",
        "ialirtrates",
        "sectorates",
        "l4fgrates",
        "l4bgrates",
    ]
    if count_rate_vars in list(sci_dataset.keys()):
        assert True

    assert np.allclose(sci_dataset["zenith"].values, ZENITH_ANGLES)
    assert np.allclose(sci_dataset["azimuth"].values, AZIMUTH_ANGLES)


def test_is_sequential():
    """Test the is_sequential function."""
    assert is_sequential(np.array([0, 1, 2, 3, 4]))
    assert not is_sequential(np.array([0, 2, 3, 4, 5]))
    # Wrap-around case
    assert is_sequential(np.array([16382, 16383, 0, 1, 2]))


def test_get_valid_starting_indices():
    """Test the find_valid_starting_indices function."""
    flags = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
        ]
    )
    counters = np.arange(35)
    result = get_valid_starting_indices(flags, counters)
    # The only valid starting index for a science frame
    # in the flags array is 15.
    assert len(result) == 1
    assert result[0] == 15


def test_update_ccsds_header_dims(sci_dataset):
    """Test the update_ccsds_header_data function.

    Replaces epoch dimension with sc_tick dimension.
    """
    updated_dataset = update_ccsds_header_dims(sci_dataset)
    assert "sc_tick" in updated_dataset.dims
    assert "epoch" not in updated_dataset.dims


def test_assemble_science_frames(sci_dataset):
    """Test the assemble_science_frames function."""
    updated_dataset = update_ccsds_header_dims(sci_dataset)
    updated_dataset = assemble_science_frames(updated_dataset)
    assert "count_rates_raw" in updated_dataset
    assert "pha_raw" in updated_dataset


@pytest.mark.parametrize(
    "packed, expected",
    [
        (0, 0),  # Test with zero
        (15, 15),  # Test with packed integer with no scaling
        (4096, 4096),  # Test with packed integer with power = 1
        (64188, 112132096),  # Test with packed integer requiring scaling
        (65535, 134201344),  # Test with maximum 16-bit value
        (62218, 79855616),  # Test with arbitrary packed integer
    ],
)
def test_decompress_rates_16_to_32(packed, expected):
    """Test the decompress_rates_16_to_32 function.

    This function decompresses a 16-bit packed integer
    to a 32-bit integer. Used to decompress rates data.
    """
    assert decompress_rates_16_to_32(packed) == expected


def test_decom_hit(sci_dataset):
    """Test the decom_hit function.

    This function orchestrates the unpacking and decompression
    of the HIT science data.
    """
    updated_dataset = decom_hit(sci_dataset)
    # Check if the dataset has the expected data variables
    sci_fields = [
        "version",
        "type",
        "sec_hdr_flg",
        "pkt_apid",
        "seq_flgs",
        "src_seq_ctr",
        "pkt_len",
        "pha_raw",
        "hdr_unit_num",
        "hdr_frame_version",
        "hdr_dynamic_threshold_state",
        "hdr_leak_conv",
        "hdr_heater_duty_cycle",
        "hdr_code_ok",
        "hdr_minute_cnt",
        "livetime_counter",
        "num_trig",
        "num_reject",
        "num_acc_w_pha",
        "num_acc_no_pha",
        "num_haz_trig",
        "num_haz_reject",
        "num_haz_acc_w_pha",
        "num_haz_acc_no_pha",
        "sngrates",
        "nread",
        "nhazard",
        "nadcstim",
        "nodd",
        "noddfix",
        "nmulti",
        "nmultifix",
        "nbadtraj",
        "nl2",
        "nl3",
        "nl4",
        "npen",
        "nformat",
        "naside",
        "nbside",
        "nerror",
        "nbadtags",
        "coinrates",
        "pbufrates",
        "l2fgrates",
        "l2bgrates",
        "l3fgrates",
        "l3bgrates",
        "penfgrates",
        "penbgrates",
        "ialirtrates",
        "sectorates",
        "l4fgrates",
        "l4bgrates",
    ]

    for field in sci_fields:
        assert field in updated_dataset
