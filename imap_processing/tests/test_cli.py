"""Tests coverage for imap_processing.cli."""

import json
import shutil
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import spiceypy
import xarray as xr
from imap_data_access.processing_input import (
    ProcessingInputCollection,
    ScienceInput,
    SPICEInput,
)

from imap_processing.cli import (
    Codice,
    Glows,
    Hi,
    Hit,
    ProcessInstrument,
    Spacecraft,
    Swe,
    Ultra,
    _validate_args,
    main,
)


@pytest.fixture
def mock_instrument_dependencies():
    with (
        mock.patch("imap_processing.cli.imap_data_access.query") as mock_query,
        mock.patch("imap_processing.cli.imap_data_access.download") as mock_download,
        mock.patch("imap_processing.cli.imap_data_access.upload") as mock_upload,
        mock.patch("imap_processing.cli.write_cdf") as mock_write_cdf,
        mock.patch("imap_processing.cli.load_cdf") as mock_load_cdf,
        mock.patch(
            "imap_processing.cli.ProcessInstrument.pre_processing"
        ) as mock_pre_processing,
    ):
        mocks = {
            "mock_query": mock_query,
            "mock_download": mock_download,
            "mock_upload": mock_upload,
            "mock_write_cdf": mock_write_cdf,
            "mock_load_cdf": mock_load_cdf,
            "mock_pre_processing": mock_pre_processing,
        }
        yield mocks


@mock.patch("imap_processing.cli.Mag")
def test_main(mock_instrument):
    """Test imap_processing.cli.main()"""
    test_args = [
        "imap_cli",
        "--instrument",
        "mag",
        "--dependency",
        (
            "["
            "{"
            '"type": "ancillary",'
            '"files": ['
            '"imap_mag_l1b-cal_20250101_v001.cdf",'
            '"imap_mag_l1b-cal_20250103-20250104_v002.cdf"'
            "]"
            "},"
            "{"
            '"type": "science",'
            '"files": ['
            '"imap_mag_l0_raw_20240430_v001.cdf",'
            "]"
            "}"
            "]"
        ),
        "--data-level",
        "l1a",
        "--start-date",
        "20240430",
        "--repointing",
        "repoint12345",
        "--version",
        "v001",
        "--upload-to-sdc",
    ]
    with mock.patch.object(sys, "argv", test_args):
        # Running without raising an exception is a pass.
        # No asserts needed.
        main()


@pytest.mark.parametrize(
    "instrument, data_level, start_date, repointing, raises_value_error",
    [
        ("mag", "l1a", "20250101", None, ""),
        ("foo", "l1a", None, None, "foo is not in the supported .*"),
        ("codice", "l1z", None, None, "l1z is not a supported .*"),
        ("glows", "l1a", None, "repoint12345", ""),
        ("glows", "l1a", None, "12345", ".* not a valid repointing.*"),
        ("glows", "l1a", "2000001", None, ".* not a valid date.*"),
    ],
)
def test_validate_args(
    instrument, data_level, start_date, repointing, raises_value_error
):
    """Test coverage for imap_processing.cli._validate_args()"""
    args = mock.Mock
    args.instrument = instrument
    args.data_level = data_level
    args.start_date = start_date
    args.repointing = repointing

    if raises_value_error:
        with pytest.raises(ValueError, match=raises_value_error):
            _validate_args(args)
    else:
        _validate_args(args)


@mock.patch("imap_processing.cli.codice_l1a.process_codice_l1a")
def test_codice(mock_codice_l1a, mock_instrument_dependencies):
    """Test coverage for cli.CoDICE class"""

    test_dataset = xr.Dataset({}, attrs={"cdf_filename": "file0"})
    input_collection = ProcessingInputCollection(
        ScienceInput("imap_codice_l0_raw_20230822_v001.pkts")
    )
    mocks = mock_instrument_dependencies
    mocks["mock_query"].return_value = [{"file_path": "/path/to/file0"}]
    mocks["mock_download"].return_value = "file0"
    mock_codice_l1a.return_value = [test_dataset]
    mocks["mock_write_cdf"].side_effect = ["/path/to/file0"]
    mocks["mock_pre_processing"].return_value = input_collection

    dependency_str = (
        '[{"type": "science","files": ["imap_codice_l0_raw_20230822_v001.pkts"]}]'
    )

    instrument = Codice("l1a", "hskp", dependency_str, "20230822", None, "v001", False)

    instrument.process()
    assert mock_codice_l1a.call_count == 1
    # Assert that write_cdf was called with the expected arguments
    assert mock_instrument_dependencies["mock_write_cdf"].call_count == 1


def test_repointing_file_creation(mock_instrument_dependencies):
    test_datasets = [xr.Dataset({}, attrs={"cdf_filename": "file0"})]
    input_collection = ProcessingInputCollection(
        ScienceInput("imap_glows_l0_raw_20230822-repoint00001_v001.pkts")
    )
    dependency_str = (
        '[{"type": "science","files": '
        '["imap_glows_l0_raw_20230822-repoint00001_v001.pkts"]}]'
    )
    instrument = Glows(
        "l1a", "hist", dependency_str, None, "repoint00002", "v001", False
    )

    mock_instrument_dependencies["mock_write_cdf"].side_effect = ["/path/to/file0"]

    # Call the method that uses write_cdf
    instrument.post_processing(test_datasets, input_collection)

    # Assert that write_cdf was called with the expected arguments
    assert mock_instrument_dependencies["mock_write_cdf"].call_count == 1
    assert (
        mock_instrument_dependencies["mock_write_cdf"]
        .call_args[0][0]
        .attrs.get("Repointing", None)
        == "repoint00002"
    )


@pytest.mark.parametrize(
    "data_level, science_input, n_prods",
    [
        ("l1a", ["imap_hi_l0_raw_20231212_v001.pkts"], 2),
        ("l1b", ["imap_hi_l1a_90sensor-de_20241105_v001.cdf"], 1),
        ("l1b", ["imap_hi_l0_raw_20231212_v001.pkts"], 2),
        ("l1c", ["imap_hi_l1b_45sensor-de_20250415_v001.cdf"], 1),
    ],
)
def test_hi_l1(mock_instrument_dependencies, data_level, science_input, n_prods):
    """Test coverage for cli.Hi class"""
    mocks = mock_instrument_dependencies
    mocks["mock_write_cdf"].side_effect = ["/path/to/file0"] * n_prods
    mocks["mock_load_cdf"].return_value = xr.Dataset()
    input_collection = ProcessingInputCollection(
        *[ScienceInput(file) for file in science_input]
    )
    mocks["mock_pre_processing"].return_value = input_collection

    # patch autospec=True makes this test confirm that the function call in cli.py
    # matches the mocked function signature.
    with mock.patch(
        f"imap_processing.cli.hi_{data_level}.hi_{data_level}", autospec=True
    ) as mock_hi:
        mock_hi.return_value = [xr.Dataset()] * n_prods
        dependency_str = (
            '[{"type": "science","files": ["imap_hi_l0_raw_20231212_v001.pkts"]}]'
        )
        instrument = Hi(
            data_level, "sci", dependency_str, "20231212", "20231213", "v005", False
        )

        instrument.process()
        assert mock_hi.call_count == 1
        assert mock_instrument_dependencies["mock_write_cdf"].call_count == n_prods


@mock.patch("imap_processing.cli.quaternions.process_quaternions", autospec=True)
def test_spacecraft(mock_spacecraft_l1a, mock_instrument_dependencies):
    """Test coverage for cli.Spacecraft class"""

    test_dataset = xr.Dataset({}, attrs={"cdf_filename": "file0"})
    input_collection = ProcessingInputCollection(
        ScienceInput("imap_spacecraft_l0_raw_20230822_v001.pkts")
    )
    mocks = mock_instrument_dependencies
    mocks["mock_query"].return_value = [{"file_path": "/path/to/file0"}]
    mocks["mock_download"].return_value = "file0"
    mock_spacecraft_l1a.return_value = [test_dataset]
    mocks["mock_write_cdf"].side_effect = ["/path/to/file0"]
    mocks["mock_pre_processing"].return_value = input_collection

    dependency_str = (
        '[{"type": "science","files": ["imap_spacecraft_l0_raw_20230822_v001.pkts"]}]'
    )

    instrument = Spacecraft(
        "l1a", "quaternions", dependency_str, "20230822", "20230822", "v001", False
    )

    instrument.process()
    assert mock_spacecraft_l1a.call_count == 1
    assert mock_instrument_dependencies["mock_write_cdf"].call_count == 1


@mock.patch("imap_processing.cli.ultra_l1a.ultra_l1a")
def test_ultra_l1a(mock_ultra_l1a, mock_instrument_dependencies):
    """Test coverage for cli.Ultra class with l1a data level"""
    mocks = mock_instrument_dependencies
    mock_ultra_l1a.return_value = [xr.Dataset(), xr.Dataset()]
    mocks["mock_write_cdf"].side_effect = ["/path/to/product0", "/path/to/product1"]
    input_collection = ProcessingInputCollection(
        ScienceInput("imap_ultra_l0_raw_20240207_v001.pkts")
    )
    mocks["mock_pre_processing"].return_value = input_collection

    dependency_str = (
        '[{"type": "science","files": ["imap_ultra_l0_raw_20240207_v001.pkts"]}]'
    )
    instrument = Ultra(
        "l1a", "raw", dependency_str, "20240207", "20240208", "v001", False
    )

    instrument.process()
    assert mock_ultra_l1a.call_count == 1
    assert mock_instrument_dependencies["mock_write_cdf"].call_count == 2


@mock.patch("imap_processing.cli.ultra_l1b.ultra_l1b")
def test_ultra_l1b(mock_ultra_l1b, mock_instrument_dependencies):
    """Test coverage for cli.Ultra class with l1b data level"""
    mocks = mock_instrument_dependencies
    mocks["mock_download"].return_value = "dependency0"
    mock_ultra_l1b.return_value = [xr.Dataset(), xr.Dataset()]
    mocks["mock_write_cdf"].side_effect = ["/path/to/product0", "/path/to/product1"]
    input_collection = ProcessingInputCollection(
        ScienceInput("imap_ultra_l1a_de_20240207_v001.cdf")
    )
    mocks["mock_pre_processing"].return_value = input_collection

    instrument = Ultra("l1b", "de", "[]", "20240207", "20240208", "v001", False)

    instrument.process()
    assert mocks["mock_download"].call_count == 0
    assert mock_ultra_l1b.call_count == 1
    assert mock_instrument_dependencies["mock_write_cdf"].call_count == 2


@mock.patch("imap_processing.cli.ultra_l1c.ultra_l1c")
def test_ultra_l1c(mock_ultra_l1c, mock_instrument_dependencies):
    """Test coverage for cli.Ultra class with l1c data level"""
    mocks = mock_instrument_dependencies
    mock_ultra_l1c.return_value = [xr.Dataset(), xr.Dataset()]
    mocks["mock_write_cdf"].side_effect = ["/path/to/product0", "/path/to/product1"]
    input_collection = ProcessingInputCollection(
        ScienceInput("imap_ultra_l1b_de_20240207_v001.cdf")
    )
    mocks["mock_pre_processing"].return_value = input_collection

    instrument = Ultra("l1c", "pset", "[]", "20240207", "20240208", "v001", False)

    instrument.process()
    assert mock_ultra_l1c.call_count == 1
    assert mock_instrument_dependencies["mock_write_cdf"].call_count == 2


@mock.patch("imap_processing.cli.ultra_l2.ultra_l2")
def test_ultra_l2(mock_ultra_l2, mock_instrument_dependencies):
    """Test coverage for cli.Ultra class with l2 data level"""
    mocks = mock_instrument_dependencies
    mock_ultra_l2.return_value = [xr.Dataset()]
    mocks["mock_write_cdf"].side_effect = ["/path/to/product0", "/path/to/product1"]
    input_collection = ProcessingInputCollection(
        ScienceInput("imap_ultra_l1c_90sensor-pset_20240207_v001.cdf"),
        ScienceInput("imap_ultra_l1c_90sensor-pset_20240208_v001.cdf"),
        ScienceInput("imap_ultra_l1c_90sensor-pset_20240209_v001.cdf"),
    )
    mocks["mock_pre_processing"].return_value = input_collection

    instrument = Ultra(
        "l2",
        "u90-ena-h-sf-nsp-full-hae-2deg-3mo",
        "[]",
        "20240207",
        "20240207",
        "v001",
        False,
    )

    instrument.process()
    assert mock_ultra_l2.call_count == 1
    assert mock_instrument_dependencies["mock_write_cdf"].call_count == 1


@mock.patch("imap_processing.cli.hit_l1a")
def test_hit_l1a(mock_hit_l1a, mock_instrument_dependencies):
    """Test coverage for cli.Hit class with l1a data level"""
    mocks = mock_instrument_dependencies
    mock_hit_l1a.return_value = [xr.Dataset(), xr.Dataset()]
    mocks["mock_write_cdf"].side_effect = ["/path/to/product0", "/path/to/product1"]
    input_collection = ProcessingInputCollection(
        ScienceInput("imap_hit_l0_raw_20100105_v001.pkts")
    )
    mocks["mock_pre_processing"].return_value = input_collection

    dependency_str = (
        '[{"type": "science","files": ["imap_hit_l0_raw_20100105_v001.pkts"]}]'
    )
    instrument = Hit(
        "l1a", "raw", dependency_str, "20100105", "20100101", "v001", False
    )

    instrument.process()
    assert mock_hit_l1a.call_count == 1
    assert mock_instrument_dependencies["mock_write_cdf"].call_count == 2


@pytest.mark.usefixtures("_unset_metakernel_path")
def test_spice_kernel_handling(spice_test_data_path):
    """Test coverage for ProcessInstrument.pre_processing method()."""
    kernels_to_furnish = ["naif0012.tls", "imap_sclk_0000.tsc"]
    dependency_obj = [
        {"type": "science", "files": ["imap_hi_l2a_sensor45-de_20100105_v001.cdf"]},
        {"type": "spice", "files": kernels_to_furnish},
        {"type": "repoint", "files": ["imap_2010_104_01.repoint.csv"]},
    ]
    dependency_str = json.dumps(dependency_obj)

    def download_side_effect(path: Path):
        """Copy kernels from spice_test_data_path to expected DATA_DIR location."""
        if (spice_test_data_path / path.name).exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(spice_test_data_path / path.name, path)

    # Test that the expected kernels are furnished within the function that
    # gets substituted in for the processing function.
    def do_processing_side_effect(*args, **kwargs):
        """Check that the expected kernels are furnished"""
        kernel_count = spiceypy.ktotal("ALL")
        furnished_kernels = [
            Path(spiceypy.kdata(i, "ALL")[0]).name for i in range(kernel_count)
        ]
        np.testing.assert_array_equal(
            sorted(furnished_kernels), sorted(kernels_to_furnish)
        )
        return [xr.Dataset()]

    with (
        mock.patch("imap_data_access.processing_input.download") as mock_download,
        mock.patch("imap_processing.cli.load_cdf"),
        mock.patch("imap_processing.cli.Hi.do_processing") as mock_do_processing,
        # mock.patch("imap_processing.cli.ProcessInstrument.post_processing"),
        mock.patch("imap_processing.cli.write_cdf"),
        mock.patch("imap_processing.cli.ProcessInstrument.upload_products"),
    ):
        mock_download.side_effect = download_side_effect
        mock_do_processing.side_effect = do_processing_side_effect

        instrument = Hi(
            "l1b", "sensor45-de", dependency_str, "20100105", None, "v001", True
        )
        # Verify no kernels are furnished prior to calling process
        assert spiceypy.ktotal("ALL") == 0
        # Verification that the expected kernels get furnished is done in the
        # mocked do_processing function
        instrument.process()
        # Verify that the furnished kernels get cleared in the post_processing method
        assert spiceypy.ktotal("ALL") == 0


@mock.patch("imap_processing.cli.swe_l1a")
@pytest.mark.parametrize(
    "query_return, expected_error",
    [
        ([], None),
        (
            [
                '{"file_path": '
                '"/path/to/imap_swe_l1a_test_20100105_v001.cdf", "instrument": "swe"}'
            ],
            ProcessInstrument.ImapFileExistsError,
        ),
    ],
)
def test_post_processing(
    mock_swe_l1a, mock_instrument_dependencies, query_return, expected_error
):
    """Test coverage for post processing"""
    mocks = mock_instrument_dependencies
    mocks["mock_download"].return_value = "dependency0"
    mocks["mock_write_cdf"].side_effect = [
        "/path/to/imap_swe_l1a_test_20100105_v001.cdf"
    ]
    mocks[
        "mock_write_cdf"
    ].return_value = "/path/to/imap_swe_l1a_test_20100105_v001.cdf"
    mocks["mock_query"].return_value = query_return

    test_ds = xr.Dataset()
    mock_swe_l1a.return_value = [test_ds]
    input_collection = ProcessingInputCollection(
        ScienceInput("imap_swe_l0_raw_20100105_v001.pkts"),
        SPICEInput("naif0012.tls", "imap_sclk_0001.tsc"),
    )
    mocks["mock_pre_processing"].return_value = input_collection

    dependency_str = (
        '[{"type": "science","files": ["imap_swe_l0_raw_20100105_v001.pkts"]}, '
        '{"type": "spice", "files": ["naif0012.tls", "imap_sclk_0001.tsc"]}]'
    )
    instrument = Swe("l1a", "raw", dependency_str, "20100105", None, "v001", True)

    if expected_error:
        with pytest.raises(expected_error):
            instrument.process()
    else:
        # This function calls both the instrument.do_processing() and
        # instrument.post_processing()
        instrument.process()
        assert mock_swe_l1a.call_count == 1
        # This test is testing that one file was uploaded
        assert mocks["mock_upload"].call_count == 1

    # Test parent injection
    assert test_ds.attrs["Parents"] == [
        "imap_swe_l0_raw_20100105_v001.pkts",
        "naif0012.tls",
        "imap_sclk_0001.tsc",
    ]
