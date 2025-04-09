"""Global pytest configuration for the package."""

import logging
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union

import cdflib
import imap_data_access
import numpy as np
import pandas as pd
import pytest
import requests
import spiceypy

from imap_processing import imap_module_directory
from imap_processing.spice.time import met_to_ttj2000ns


@pytest.fixture(autouse=True)
def _set_global_config(monkeypatch, tmp_path):
    """Set the global data directory to a temporary directory."""
    monkeypatch.setitem(imap_data_access.config, "DATA_DIR", tmp_path)
    monkeypatch.setitem(
        imap_data_access.config, "DATA_ACCESS_URL", "https://api.test.com"
    )


@pytest.fixture(scope="session")
def imap_tests_path():
    return imap_module_directory / "tests"


# Furnishing fixtures for testing kernels
# ---------------------------------------
@pytest.fixture(autouse=True)
def _autoclear_spice():
    """Automatically clears out all SPICE remnants after every single test to
    prevent the kernel pool from interfering with future tests. Option autouse
    ensures this is run after every test."""
    yield
    spiceypy.kclear()


@pytest.fixture(scope="session")
def _download_external_kernels(spice_test_data_path):
    """This fixture downloads externally-located kernels into the tests/spice/test_data
    directory if they do not already exist there. The fixture is not intended to be
    used directly. It is automatically added to tests marked with "external_kernel"
    in the hook below."""
    logger = logging.getLogger(__name__)
    kernel_urls = [
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp",
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc",
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/"
        "earth_1962_240827_2124_combined.bpc",
    ]

    for kernel_url in kernel_urls:
        kernel_name = kernel_url.split("/")[-1]
        local_filepath = spice_test_data_path / kernel_name

        if local_filepath.exists():
            continue
        allowed_attempts = 3
        for attempt_number in range(allowed_attempts):
            try:
                with requests.get(kernel_url, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    with open(local_filepath, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                logger.info("Cached kernel file to %s", local_filepath)
                continue
            except requests.exceptions.RequestException as error:
                logger.info(f"Request failed. {error}")
                if attempt_number < allowed_attempts:
                    logger.info(
                        f"Trying again, retries left "
                        f"{allowed_attempts - attempt_number}, "
                        f"Exception: {error}"
                    )
                    time.sleep(1)
                else:
                    logger.error(
                        f"Failed to download file {kernel_name} after "
                        f"{allowed_attempts} attempts, Final Error: {error}"
                    )
                    raise


@pytest.fixture(scope="session")
def _download_test_data():
    _download_external_data(test_data_paths())


def _download_external_data(test_data_path_list):
    """This fixture downloads externally-located test data files into a specific
    location. The list of files and their storage locations are specified in
    the `test_data_paths` parameter, which is a list of tuples; the zeroth
    element being the source of the test file in the AWS S3 bucket, and the
    first element being the location in which to store the downloaded file."""

    logger = logging.getLogger(__name__)

    api_path = "https://api.dev.imap-mission.com/download/test_data/"
    for test_data_path in test_data_path_list:
        source = api_path + test_data_path[0]
        destination = test_data_path[1]

        # Download the test data if necessary and write it to the appropriate
        # directory
        if not destination.exists():
            response = requests.get(source, timeout=60)
            if response.status_code == 200:
                with open(destination, "wb") as file:
                    file.write(response.content)
                logger.info(f"Downloaded file: {source}")
            else:
                logger.error(f"Failed to download file: {response.status_code}")
        else:
            logger.info(f"File already exists: {destination}")


def test_data_paths():
    """Defines a list of test data files to download from the AWS S3 bucket
    and the corresponding location in which to store the downloaded file"""
    test_data_path_list = [
        (
            "imap_codice_l0_raw_20241110_v001.pkts",
            imap_module_directory
            / "tests"
            / "codice"
            / "data"
            / "imap_codice_l0_raw_20241110_v001.pkts",
        ),
        (
            "imap_hi_l1a_45sensor-de_20250415_v999.cdf",
            imap_module_directory
            / "tests"
            / "hi"
            / "data"
            / "l1"
            / "imap_hi_l1a_45sensor-de_20250415_v999.cdf",
        ),
        (
            "imap_hi_l1b_45sensor-de_20250415_v999.cdf",
            imap_module_directory
            / "tests"
            / "hi"
            / "data"
            / "l1"
            / "imap_hi_l1b_45sensor-de_20250415_v999.cdf",
        ),
        (
            "idex_l1a_validation_file.h5",
            imap_module_directory
            / "tests"
            / "idex"
            / "test_data"
            / "idex_l1a_validation_file.h5",
        ),
        (
            "idex_l1b_validation_file.h5",
            imap_module_directory
            / "tests"
            / "idex"
            / "test_data"
            / "idex_l1b_validation_file.h5",
        ),
        (
            "ultra-90_raw_event_data_shortened.csv",
            imap_module_directory
            / "tests"
            / "ultra"
            / "data"
            / "l1"
            / "ultra-90_raw_event_data_shortened.csv",
        ),
        (
            "Ultra_90_DPS_efficiencies_all.csv",
            imap_module_directory
            / "tests"
            / "ultra"
            / "data"
            / "l1"
            / "Ultra_90_DPS_efficiencies_all.csv",
        ),
        (
            "ultra_90_dps_gf.csv",
            imap_module_directory
            / "tests"
            / "ultra"
            / "data"
            / "l1"
            / "ultra_90_dps_gf.csv",
        ),
        (
            "ultra_90_dps_exposure.csv",
            imap_module_directory
            / "tests"
            / "ultra"
            / "data"
            / "l1"
            / "ultra_90_dps_exposure.csv",
        ),
        (
            "Ultra_efficiencies_45_combined_logistic_interpolation.csv",
            imap_module_directory
            / "tests"
            / "ultra"
            / "data"
            / "l1"
            / "Ultra_efficiencies_45_combined_logistic_interpolation.csv",
        ),
    ]

    return test_data_path_list


def pytest_collection_modifyitems(items):
    """
    The use of this hook allows modification of test `Items` after tests have
    been collected. In this case, it automatically adds fixtures based on the
    following table:

    +---------------------+----------------------------+
    | pytest mark         | fixture added              |
    +=====================+============================+
    | external_kernel     | _download_external_kernels |
    | external_test_data  | _download_test_data        |
    | use_test_metakernel | use_test_metakernel        |
    +---------------------+----------------------------+

    Notes
    -----
    See the following link for details about this function, also known as a
    pytest hook:
    https://docs.pytest.org/en/stable/reference/reference.html#
    pytest.hookspec.pytest_collection_modifyitems
    """
    markers_to_fixtures = {
        "external_kernel": "_download_external_kernels",
        "external_test_data": "_download_test_data",
        "use_test_metakernel": "use_test_metakernel",
    }

    for item in items:
        for marker, fixture in markers_to_fixtures.items():
            if item.get_closest_marker(marker) is not None:
                item.fixturenames.append(fixture)


@pytest.fixture(scope="session")
def spice_test_data_path(imap_tests_path):
    return imap_tests_path / "spice/test_data"


@pytest.fixture
def furnish_time_kernels(spice_test_data_path):
    """Furnishes (temporarily) the testing LSK and SCLK"""
    spiceypy.kclear()
    test_lsk = spice_test_data_path / "naif0012.tls"
    test_sclk = spice_test_data_path / "imap_sclk_0000.tsc"
    spiceypy.furnsh(str(test_lsk))
    spiceypy.furnsh(str(test_sclk))
    yield test_lsk, test_sclk
    spiceypy.kclear()


@pytest.fixture
def furnish_sclk(spice_test_data_path):
    """Furnishes (temporarily) the SCLK for JPSS stored in the package data directory"""
    test_sclk = spice_test_data_path / "imap_sclk_0000.tsc"
    spiceypy.furnsh(str(test_sclk))
    yield test_sclk
    spiceypy.kclear()


@pytest.fixture
def furnish_kernels(spice_test_data_path):
    """Return a function that will furnish an arbitrary list of kernels."""

    @contextmanager
    def furnish_kernels(kernels: list[Path]):
        with spiceypy.KernelPool(
            [str(spice_test_data_path / k) for k in kernels]
        ) as pool:
            yield pool

    return furnish_kernels


@pytest.fixture(scope="session")
def monkeypatch_session():
    from _pytest.monkeypatch import MonkeyPatch

    m = MonkeyPatch()
    yield m
    m.undo()


def make_metakernel_from_kernels(metakernel, kernels):
    """Helper function that writes a test metakernel from a list of filenames"""
    with open(metakernel, "w") as mk:
        mk.writelines(
            [
                "\n",
                "\\begintext\n",
                "\n",
                "This is a temporary metakernel for imap_processing"
                " unit and integration testing.\n",
                "\n",
                "\\begindata\n",
                "\n",
                "KERNELS_TO_LOAD = (\n",
            ]
        )
        # Put single quotes around every kernel name
        kernels_with_quotes = ["    '" + kern + "'" for kern in kernels]
        # Add a comma and EOL to the end of each kernel path except the last.
        formatted_kernels = [kern + ",\n" for kern in kernels_with_quotes[0:-1]]
        # Add ')' to the last kernel
        formatted_kernels.append(kernels_with_quotes[-1] + "\n)\n\n")
        mk.writelines(formatted_kernels)


def get_test_kernels_to_load(template_path, kernel_dir_path):
    """
    Helper function for grabbing a list of kernel filenames from the test
    metakernel template. This is necessary in order to get absolute paths on
    any system. Formats the absolute paths using the test data path fixture
    value.
    """
    kernels_to_load = []
    max_line_length = 80
    with open(template_path) as mk:
        for k in mk:
            kernel = k.rstrip("\n").format(
                **{"SPICE_TEST_DATA_PATH": str(kernel_dir_path.absolute())}
            )
            while len(kernel) > 0:
                if len(kernel) <= max_line_length:
                    kernels_to_load.append(kernel)
                    break
                else:
                    slash_positions = np.array(
                        [m.start() for m in re.finditer("/", kernel)]
                    )
                    stop_idx = (
                        slash_positions[slash_positions < max_line_length - 1].max() + 1
                    )
                    kernels_to_load.append(kernel[0:stop_idx] + "+")
                    kernel = kernel[stop_idx:]
    return kernels_to_load


@pytest.fixture(scope="session", autouse=True)
def session_test_metakernel(monkeypatch_session, tmpdir_factory, spice_test_data_path):
    """Generate a metakernel from the template metakernel by injecting the local
    path into the metakernel and set the SPICE_METAKERNEL environment variable.

    Notes
    -----
    - This fixture needs to `scope=session` so that the SPICE_METAKERNEL
    environment variable is available for other fixtures that require time
    conversions using spiceypy.
    - No furnishing of kernels occur as part of this fixture. This allows other
    fixtures with lesser scope or individual tests to override the environment
    variable as needed. Use the `metakernel_path_not_set` fixture in tests that
    need to override the environment variable.
    """
    template_path = spice_test_data_path / "imap_simple_metakernel.template"
    kernels_to_load = get_test_kernels_to_load(template_path, spice_test_data_path)
    metakernel_path = tmpdir_factory.mktemp("spice") / "imap_2024_v001.tm"
    make_metakernel_from_kernels(metakernel_path, kernels_to_load)
    monkeypatch_session.setenv("SPICE_METAKERNEL", str(metakernel_path))
    yield str(metakernel_path)
    spiceypy.kclear()


@pytest.fixture
def use_test_metakernel(
    request, monkeypatch, spice_test_data_path, session_test_metakernel
):
    """
    Generate a metakernel and set SPICE_METAKERNEL environment variable.

    This fixture generates a metakernel in the directory pointed to by
    `imap_data_access.config["DATA_DIR"]` and sets the SPICE_METAKERNEL
    environment variable to point to it for use by the `@ensure_spice` decorator.
    The default metekernel is named "imap_simple_metakernel.template". Other
    metakerels can be specified by marking the test with metakernel. See
    examples below.

    Parameters
    ----------
    request : fixture
    monkeypatch : fixture
    spice_test_data_path : fixture
    session_test_metakernel : fixture

    Yields
    ------
    metakernel_path : Path

    Examples
    --------
    1. Use the default metakernel template
        >>> def test_my_spicey_func(use_test_metakernel):
        ...     pass

    2. Specify a different metakernel template
        >>> @pytest.mark.use_test_metakernel("other_template_mk.template")
        ... def test_my_spicey_func():
        ...     pass
    """
    marker = request.node.get_closest_marker("use_test_metakernel")
    if marker is None:
        yield session_test_metakernel
    else:
        template_name = marker.args[0]
        template_path = spice_test_data_path / template_name
        metakernel_path = imap_data_access.config["DATA_DIR"] / "imap_2024_v001.tm"
        kernels_to_load = get_test_kernels_to_load(template_path, spice_test_data_path)
        make_metakernel_from_kernels(metakernel_path, kernels_to_load)
        monkeypatch.setenv("SPICE_METAKERNEL", str(metakernel_path))
        yield str(metakernel_path)
    spiceypy.kclear()


@pytest.fixture
def _unset_metakernel_path(monkeypatch):
    """Temporarily unsets the SPICE_METAKERNEL environment variable"""
    if os.getenv("SPICE_METAKERNEL", None) is not None:
        monkeypatch.delenv("SPICE_METAKERNEL")


@pytest.fixture
def use_test_spin_data_csv(monkeypatch):
    """Sets the SPIN_DATA_FILEPATH environment variable to input path."""

    def wrapped_set_spin_data_filepath(path: Path):
        monkeypatch.setenv("SPIN_DATA_FILEPATH", str(path))

    return wrapped_set_spin_data_filepath


@pytest.fixture
def use_fake_spin_data_for_time(
    request, use_test_spin_data_csv, tmpdir, generate_spin_data
):
    """
    Generate and use fake spin data for testing.

    Returns
    -------
    callable
        Returns a callable function that takes start_met and optionally end_met
        as inputs, generates fake spin data, writes the data to a csv file,
        and sets the SPIN_DATA_FILEPATH environment variable to point to the
        fake spin data file.
    """

    def wrapped_set_spin_data_filepath(
        start_met: float, end_met: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate and use fake spin data for testing.
        Parameters
        ----------
        start_met : int
            Provides the start time in Mission Elapsed Time (MET).
        end_met : int
            Provides the end time in MET. If not provided, default to one day
            from start time.
        """
        spin_df = generate_spin_data(start_met, end_met=end_met)
        spin_csv_file_path = tmpdir / "spin_data.spin.csv"
        spin_df.to_csv(spin_csv_file_path, index=False)
        use_test_spin_data_csv(spin_csv_file_path)

    return wrapped_set_spin_data_filepath


@pytest.fixture
def generate_spin_data():
    def make_data(start_met: float, end_met: Optional[float] = None) -> pd.DataFrame:
        """
        Generate a spin table CSV covering one or more days.
        Spin table contains the following fields:
            (
            spin_number,
            spin_start_sec,
            spin_start_subsec,
            spin_period_sec,
            spin_period_valid,
            spin_phase_valid,
            spin_period_source,
            thruster_firing
            )
        This function creates spin data using start MET and end MET time.
        Each spin start data uses the nominal 15-second spin period. The spins that
        occur from 00:00(Mid-night) to 00:10 UTC are marked with flags for
        thruster firing, invalid spin period, and invalid spin phase.
        Parameters
        ----------
        start_met : float
            Provides the start time in Mission Elapsed Time (MET).
        end_met : float
            Provides the end time in MET. If not provided, default to one day
            from start time.
        Returns
        -------
        spin_df : pd.DataFrame
            Spin data.
        """
        if end_met is None:
            # end_time is one day after start_time
            end_met = start_met + 86400

        # Create spin start second data of 15 seconds increment
        spin_start_sec = np.arange(np.floor(start_met), end_met + 1, 15)
        spin_start_subsec = int((start_met - spin_start_sec[0]) * 1000)

        nspins = len(spin_start_sec)

        spin_df = pd.DataFrame.from_dict(
            {
                "spin_number": np.arange(nspins, dtype=np.uint32),
                "spin_start_sec": spin_start_sec,
                "spin_start_subsec": np.full(
                    nspins, spin_start_subsec, dtype=np.uint32
                ),
                "spin_period_sec": np.full(nspins, 15.0, dtype=np.float32),
                "spin_period_valid": np.ones(nspins, dtype=np.uint8),
                "spin_phase_valid": np.ones(nspins, dtype=np.uint8),
                "spin_period_source": np.zeros(nspins, dtype=np.uint8),
                "thruster_firing": np.zeros(nspins, dtype=np.uint8),
            }
        )

        # Convert spin_start_sec to datetime to set repointing times flags
        spin_start_dates = met_to_ttj2000ns(spin_start_sec + spin_start_subsec / 1000)
        spin_start_dates = cdflib.cdfepoch.to_datetime(spin_start_dates)

        # Convert DatetimeIndex to Series for using .dt accessor
        spin_start_dates_series = pd.Series(spin_start_dates)

        # Find index of all timestamps that fall within 10 minutes after midnight
        repointing_times = spin_start_dates_series[
            (spin_start_dates_series.dt.time >= pd.Timestamp("00:00:00").time())
            & (spin_start_dates_series.dt.time < pd.Timestamp("00:10:00").time())
        ]

        repointing_times_index = repointing_times.index

        # Use the repointing times to set thruster firing flag and spin period valid
        spin_df.loc[repointing_times_index.values, "thruster_firing"] = 1
        spin_df.loc[repointing_times_index.values, "spin_period_valid"] = 0
        spin_df.loc[repointing_times_index.values, "spin_phase_valid"] = 0

        return spin_df

    return make_data


@pytest.fixture
def use_test_repoint_data_csv(monkeypatch):
    """Sets the REPOINT_DATA_FILEPATH environment variable to input path."""

    def wrapped_set_repoint_data_filepath(path: Path):
        monkeypatch.setenv("REPOINT_DATA_FILEPATH", str(path))

    return wrapped_set_repoint_data_filepath


def generate_repoint_data(
    repoint_start_met: Union[float, np.ndarray],
    repoint_end_met: Optional[Union[float, np.ndarray]] = None,
    repoint_id_start: Optional[int] = 0,
) -> pd.DataFrame:
    """
    Generate a repoint dataframe for the star/end times provided.

    Parameters
    ----------
    repoint_start_met : float, np.ndarray
            Provides the repoint start time(s) in Mission Elapsed Time (MET).
    repoint_end_met : float, np.ndarray, optional
        Provides the repoint end time(s) in MET. If not provided, end times
        will be 15 minutes after start times.
    repoint_id_start : int, optional
        Provides the starting repoint id number of the first repoint in the
        generated data.

    Returns
    -------
    repoint_df : pd.DataFrame
        Repoint dataframe with start and end repoint times provided and incrementing
        repoint_ids starting at 1.
    """
    repoint_start_times = np.array(repoint_start_met)
    if repoint_end_met is None:
        repoint_end_met = repoint_start_times + 15 * 60
    repoint_df = pd.DataFrame.from_dict(
        {
            "repoint_start_sec": repoint_start_times.astype(int),
            "repoint_start_subsec": ((repoint_start_times % 1.0) * 1e3).astype(int),
            "repoint_end_sec": repoint_end_met.astype(int),
            "repoint_end_subsec": ((repoint_end_met % 1.0) * 1e3).astype(int),
            "repoint_id": np.arange(repoint_start_times.size, dtype=int)
            + repoint_id_start,
        }
    )
    return repoint_df


@pytest.fixture
def use_fake_repoint_data_for_time(use_test_repoint_data_csv, tmpdir):
    """
    Generate and use fake spin data for testing.

    Returns
    -------
    callable
        Returns a callable function that takes start_met and optionally n_repoints
        as inputs, generates fake repoint data, writes the data to a csv file,
        and sets the REPOINT_DATA_FILEPATH environment variable to point to the
        fake repoint data file.
    """

    def wrapped_repoint_data_filepath(
        repoint_start_met: Union[float, np.ndarray],
        repoint_end_met: Optional[Union[float, np.ndarray]] = None,
        repoint_id_start: Optional[int] = 0,
    ) -> pd.DataFrame:
        """
        Generate and use fake repoint data for testing.
        Parameters
        ----------
        repoint_start_met : float, np.ndarray
            Provides the repoint start time(s) in Mission Elapsed Time (MET).
        repoint_end_met : float, np.ndarray
            Provides the repoint end time(s) in MET. If not provided, end times
            will be 15 minutes after start times.
        repoint_id_start : int, optional
            Provides the starting repoint id number of the first repoint in the
            generated data.
        """
        repoint_df = generate_repoint_data(
            repoint_start_met,
            repoint_end_met=repoint_end_met,
            repoint_id_start=repoint_id_start,
        )
        repoint_csv_file_path = tmpdir / "repoint_data.repointing.csv"
        repoint_df.to_csv(repoint_csv_file_path, index=False)
        use_test_repoint_data_csv(repoint_csv_file_path)

    return wrapped_repoint_data_filepath
