import numpy as np
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.hit.hit_utils import (
    HitAPID,
    add_energy_variables,
    add_summed_particle_data_to_dataset,
    concatenate_leak_variables,
    get_attribute_manager,
    get_datasets_by_apid,
    initialize_particle_data_arrays,
    process_housekeeping_data,
    sum_particle_data,
)

np.random.seed(42)


@pytest.fixture(scope="module")
def packet_filepath():
    """Set path to test data file"""
    return (
        imap_module_directory / "tests/hit/test_data/imap_hit_l0_raw_20100105_v001.pkts"
    )


@pytest.fixture(scope="module")
def attribute_manager():
    """Create the attribute manager"""
    level = "l1a"
    attr_mgr = get_attribute_manager(level)
    return attr_mgr


@pytest.fixture(scope="module")
def housekeeping_dataset(packet_filepath):
    """Get the housekeeping dataset"""
    # Unpack ccsds file to xarray datasets
    datasets_by_apid = get_datasets_by_apid(packet_filepath)
    return datasets_by_apid[HitAPID.HIT_HSKP]


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for summing particle data"""
    data = {
        "l2fgrates": (("epoch", "energy"), np.random.rand(10, 5)),
        "l3fgrates": (("epoch", "energy"), np.random.rand(10, 5)),
        "penfgrates": (("epoch", "energy"), np.random.rand(10, 5)),
        "l2fgrates_delta_minus": (("epoch", "energy"), np.random.rand(10, 5)),
        "l3fgrates_delta_minus": (("epoch", "energy"), np.random.rand(10, 5)),
        "penfgrates_delta_minus": (("epoch", "energy"), np.random.rand(10, 5)),
        "l2fgrates_delta_plus": (("epoch", "energy"), np.random.rand(10, 5)),
        "l3fgrates_delta_plus": (("epoch", "energy"), np.random.rand(10, 5)),
        "penfgrates_delta_plus": (("epoch", "energy"), np.random.rand(10, 5)),
    }
    return xr.Dataset(data)


def test_get_datasets_by_apid(packet_filepath):
    result = get_datasets_by_apid(packet_filepath)

    assert isinstance(result, dict)
    assert HitAPID.HIT_HSKP in result
    # assert HitAPID.HIT_SCIENCE in result


def test_get_attribute_manager():
    level = "l1a"
    attr_mgr = get_attribute_manager(level)

    assert isinstance(attr_mgr, ImapCdfAttributes)


def test_concatenate_leak_variables(housekeeping_dataset):
    """Test concatenation of leak_i variables"""

    # Create data array for leak_i dependency
    adc_channels = xr.DataArray(
        np.arange(64, dtype=np.uint8),
        name="adc_channels",
        dims=["adc_channels"],
    )

    updated_dataset = concatenate_leak_variables(housekeeping_dataset, adc_channels)

    # Assertions
    # ----------------
    assert "leak_i" in updated_dataset
    assert updated_dataset["leak_i"].shape == (88, 64)
    for i in range(64):
        # Check if the values in the `leak_i` variable match the values in
        # the original `leak_i_XX` variable.
        #  - First access the `leak_i` variable in the `updated_dataset`.
        #    The [:, i] selects all rows (`:`) and the `i`-th column of the `leak_i`
        #    variable.
        #  - Then access the `leak_i_XX` variable in the `housekeeping_dataset`.
        #    The `f"leak_i_{i:02d}"` selects the variable with the name `leak_i_XX`
        #    where `XX` is the `i`-th value.
        #  - Compare values
        np.testing.assert_array_equal(
            updated_dataset["leak_i"][:, i], housekeeping_dataset[f"leak_i_{i:02d}"]
        )


def test_process_housekeeping(housekeeping_dataset, attribute_manager):
    """Test processing of housekeeping dataset"""

    # Call the function
    processed_hskp_dataset = process_housekeeping_data(
        housekeeping_dataset, attribute_manager, "imap_hit_l1a_hk"
    )

    # Define the keys that should have dropped from the dataset
    dropped_keys = {
        "pkt_apid",
        "version",
        "type",
        "sec_hdr_flg",
        "seq_flgs",
        "src_seq_ctr",
        "pkt_len",
        "hskp_spare1",
        "hskp_spare2",
        "hskp_spare3",
        "hskp_spare4",
        "hskp_spare5",
    }
    # Define the keys that should be present
    valid_keys = {
        "sc_tick",
        "heater_on",
        "fsw_version_b",
        "ebox_m12va",
        "phasic_stat",
        "ebox_3d4vd",
        "ebox_p2d0vd",
        "temp1",
        "last_bad_seq_num",
        "ebox_m5d7va",
        "ebox_p12va",
        "table_status",
        "enable_50khz",
        "mram_disabled",
        "temp3",
        "preamp_l1a",
        "l2ab_bias",
        "l34b_bias",
        "fsw_version_c",
        "num_evnt_last_hk",
        "dac1_enable",
        "preamp_l234b",
        "analog_temp",
        "fee_running",
        "fsw_version_a",
        "num_errors",
        "test_pulser_on",
        "dac0_enable",
        "preamp_l1b",
        "l1ab_bias",
        "l34a_bias",
        "leak_i",
        "last_good_cmd",
        "lvps_temp",
        "idpu_temp",
        "temp2",
        "preamp_l234a",
        "last_good_seq_num",
        "num_good_cmds",
        "heater_control",
        "hvps_temp",
        "ebox_p5d7va",
        "spin_period_long",
        "enable_hvps",
        "temp0",
        "spin_period_short",
        "dyn_thresh_lvl",
        "num_bad_cmds",
        "adc_mode",
        "ebox_5d1vd",
        "active_heater",
        "last_error_num",
        "last_bad_cmd",
        "ref_p5v",
        "code_checksum",
        "mode",
    }

    # Define the dataset attributes
    dataset_attrs = {
        "Acknowledgement": "Please acknowledge the IMAP Mission Principal "
        "Investigator, Prof. David J. McComas of Princeton "
        "University.\n",
        "Data_type": "L1A_HK>Level-1A Housekeeping",
        "Data_version": None,
        "Descriptor": "HIT>IMAP High-energy Ion Telescope",
        "Discipline": "Solar Physics>Heliospheric Physics",
        "File_naming_convention": "source_descriptor_datatype_yyyyMMdd_vNNN",
        "HTTP_LINK": "https://imap.princeton.edu/",
        "Instrument_type": "Particles (space)",
        "LINK_TITLE": "IMAP The Interstellar Mapping and Acceleration Probe",
        "Logical_file_id": None,
        "Logical_source": "imap_hit_l1a_hk",
        "Logical_source_description": "IMAP Mission HIT Instrument Level-1A "
        "Housekeeping Data.",
        "Mission_group": "IMAP",
        "PI_affiliation": "Princeton University",
        "PI_name": "Prof. David J. McComas",
        "Project": "STP>Solar Terrestrial Probes",
        "Rules_of_use": "All IMAP data products are publicly released and citable for "
        "use in publications. Please consult the IMAP team "
        "publications and personnel for further details on "
        "production, processing, and usage of these data.\n",
        "Source_name": "IMAP>Interstellar Mapping and Acceleration Probe",
        "TEXT": "The High-energy Ion Telescope (HIT) measures the elemental "
        "composition, energy spectra, angle distributions, and arrival "
        "times of high-energy ions. HIT delivers full-sky coverage from "
        "a wide instrument field-of-view (FOV) to enable a high resolution "
        "of ion measurements, such as observing shock-accelerated ions, "
        "determining the origin of the solar energetic particles (SEPs) "
        "spectra, and resolving particle transport in the heliosphere. "
        "See https://imap.princeton.edu/instruments/hit for more details.\n",
    }

    # Define the coordinates and dimensions. Both have equivalent values
    dataset_coords_dims = {"epoch", "adc_channels", "adc_channels_label"}

    # Assertions
    # ----------------
    # Check that the dataset has the correct variables
    assert valid_keys == set(processed_hskp_dataset.data_vars.keys())
    assert set(dropped_keys).isdisjoint(set(processed_hskp_dataset.data_vars.keys()))
    # Check that the dataset has the correct attributes, coordinates, and dimensions
    assert processed_hskp_dataset.attrs == dataset_attrs
    assert processed_hskp_dataset.coords.keys() == dataset_coords_dims


def test_add_energy_variables():
    """Test adding energy variables to a dataset"""
    # Create an empty dataset
    dataset = xr.Dataset()

    # Create sample data
    particle = "test_particle"
    energy_min = np.array([1.8, 4.0, 6.0], dtype=np.float32)
    energy_max = np.array([2.2, 6.0, 10.0], dtype=np.float32)
    energy_mean = np.mean([energy_min, energy_max], axis=0)

    # Call the function
    dataset = add_energy_variables(dataset, particle, energy_min, energy_max)

    # Assertions
    assert f"{particle}_energy_delta_minus" in dataset.data_vars
    assert f"{particle}_energy_delta_plus" in dataset.data_vars
    assert f"{particle}_energy_mean" in dataset.coords
    assert np.all(
        dataset[f"{particle}_energy_delta_minus"].values
        == np.array(energy_mean - np.array(energy_min), dtype=np.float32)
    )
    assert np.all(
        dataset[f"{particle}_energy_delta_plus"].values
        == np.array(energy_max - energy_mean, dtype=np.float32)
    )
    assert np.all(dataset[f"{particle}_energy_mean"].values == energy_mean)


def test_sum_particle_data(sample_dataset):
    # Create a sample dataset
    dataset = xr.Dataset(sample_dataset)

    # Define indices for summing
    indices = {
        "R2": [0, 1],
        "R3": [2, 3],
        "R4": [4],
    }

    # Call the function
    summed_data, summed_uncertainty_delta_minus, summed_uncertainty_delta_plus = (
        sum_particle_data(dataset, indices)
    )

    # Assertions
    assert summed_data.shape == (10,)
    assert summed_uncertainty_delta_minus.shape == (10,)
    assert summed_uncertainty_delta_plus.shape == (10,)
    assert np.all(
        summed_data
        == dataset["l2fgrates"][:, indices["R2"]].sum(axis=1)
        + dataset["l3fgrates"][:, indices["R3"]].sum(axis=1)
        + dataset["penfgrates"][:, indices["R4"]].sum(axis=1)
    )
    assert np.all(
        summed_uncertainty_delta_minus
        == dataset["l2fgrates_delta_minus"][:, indices["R2"]].sum(axis=1)
        + dataset["l3fgrates_delta_minus"][:, indices["R3"]].sum(axis=1)
        + dataset["penfgrates_delta_minus"][:, indices["R4"]].sum(axis=1)
    )
    assert np.all(
        summed_uncertainty_delta_plus
        == dataset["l2fgrates_delta_plus"][:, indices["R2"]].sum(axis=1)
        + dataset["l3fgrates_delta_plus"][:, indices["R3"]].sum(axis=1)
        + dataset["penfgrates_delta_plus"][:, indices["R4"]].sum(axis=1)
    )


def test_add_summed_particle_data_to_dataset(sample_dataset):
    """Test adding summed particle data to a dataset"""
    # Create a sample source dataset
    source_dataset = xr.Dataset(sample_dataset)

    # Create an empty dataset to update
    dataset_to_update = xr.Dataset()

    # Define particle and energy ranges
    particle = "test_particle"
    energy_ranges = [
        {"energy_min": 1.8, "energy_max": 2.2, "R2": [0], "R3": [1], "R4": [2]},
        {"energy_min": 4.0, "energy_max": 6.0, "R2": [3], "R3": [4], "R4": []},
    ]

    # Call the function
    dataset_to_update = add_summed_particle_data_to_dataset(
        dataset_to_update, source_dataset, particle, energy_ranges
    )

    # Assertions
    assert f"{particle}" in dataset_to_update.data_vars
    assert f"{particle}_delta_minus" in dataset_to_update.data_vars
    assert f"{particle}_delta_plus" in dataset_to_update.data_vars
    assert f"{particle}_energy_delta_minus" in dataset_to_update.data_vars
    assert f"{particle}_energy_delta_plus" in dataset_to_update.data_vars
    assert f"{particle}_energy_mean" in dataset_to_update.coords

    assert dataset_to_update[f"{particle}"].shape == (10, len(energy_ranges))
    assert dataset_to_update[f"{particle}_delta_minus"].shape == (
        10,
        len(energy_ranges),
    )
    assert dataset_to_update[f"{particle}_delta_plus"].shape == (10, len(energy_ranges))
    assert dataset_to_update[f"{particle}_energy_mean"].shape == (len(energy_ranges),)
    assert dataset_to_update[f"{particle}_energy_delta_minus"].shape == (
        len(energy_ranges),
    )
    assert dataset_to_update[f"{particle}_energy_delta_plus"].shape == (
        len(energy_ranges),
    )

    assert np.all(
        dataset_to_update[f"{particle}_energy_mean"].values
        == np.mean([[1.8, 4.0], [2.2, 6.0]], axis=0)
    )


def test_initialize_particle_data_arrays():
    # Create an empty dataset
    dataset = xr.Dataset()

    # Define parameters
    particle = "test_particle"
    num_energy_ranges = 5
    epoch_size = 10

    # Call the function
    dataset = initialize_particle_data_arrays(
        dataset, particle, num_energy_ranges, epoch_size
    )

    # Assertions
    assert f"{particle}" in dataset.data_vars
    assert f"{particle}_delta_minus" in dataset.data_vars
    assert f"{particle}_delta_plus" in dataset.data_vars
    assert f"{particle}_energy_mean" in dataset.coords

    assert dataset[f"{particle}"].shape == (epoch_size, num_energy_ranges)
    assert dataset[f"{particle}_delta_minus"].shape == (epoch_size, num_energy_ranges)
    assert dataset[f"{particle}_delta_plus"].shape == (epoch_size, num_energy_ranges)
    assert dataset[f"{particle}_energy_mean"].shape == (num_energy_ranges,)

    assert np.all(dataset[f"{particle}"].values == 0)
    assert np.all(dataset[f"{particle}_delta_minus"].values == 0)
    assert np.all(dataset[f"{particle}_delta_plus"].values == 0)
    assert np.all(dataset[f"{particle}_energy_mean"].values == 0)
