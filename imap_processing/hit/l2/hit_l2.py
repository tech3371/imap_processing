"""IMAP-HIT L2 data processing."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from imap_processing.hit.hit_utils import (
    add_summed_particle_data_to_dataset,
    get_attribute_manager,
)
from imap_processing.hit.l2.constants import (
    FILLVAL_FLOAT32,
    L2_SECTORED_ANCILLARY_PATH_PREFIX,
    L2_STANDARD_ANCILLARY_PATH_PREFIX,
    L2_SUMMED_ANCILLARY_PATH_PREFIX,
    N_AZIMUTH,
    SECONDS_PER_10_MIN,
    SECONDS_PER_MIN,
    STANDARD_PARTICLE_ENERGY_RANGE_MAPPING,
    VALID_SECTORED_SPECIES,
    VALID_SPECIES,
)

logger = logging.getLogger(__name__)

# TODO:
#  - review logging levels to use (debug vs. info)
#  - determine where to pull ancillary data. Storing it locally for now


def hit_l2(dependency: xr.Dataset) -> list[xr.Dataset]:
    """
    Will process HIT data to L2.

    Processes dependencies needed to create L2 data products.

    Parameters
    ----------
    dependency : xr.Dataset
        L1B xarray science dataset that is either summed rates
        standard rates or sector rates.

    Returns
    -------
    processed_data : list[xarray.Dataset]
        List of one L2 dataset.
    """
    logger.info("Creating HIT L2 science datasets")

    # Create the attribute manager for this data level
    attr_mgr = get_attribute_manager("l2")

    l2_datasets: dict = {}

    # Process science data to L2 datasets
    if "imap_hit_l1b_summed-rates" in dependency.attrs["Logical_source"]:
        l2_datasets["imap_hit_l2_summed-intensity"] = process_summed_intensity_data(
            dependency
        )
        logger.info("HIT L2 summed intensity dataset created")

    if "imap_hit_l1b_standard-rates" in dependency.attrs["Logical_source"]:
        l2_datasets["imap_hit_l2_standard-intensity"] = process_standard_intensity_data(
            dependency
        )
        logger.info("HIT L2 standard intensity dataset created")

    if "imap_hit_l1b_sectored-rates" in dependency.attrs["Logical_source"]:
        l2_datasets["imap_hit_l2_macropixel-intensity"] = (
            process_sectored_intensity_data(dependency)
        )
        logger.info("HIT L2 macropixel intensity dataset created")

    # Update attributes and dimensions
    for logical_source, dataset in l2_datasets.items():
        dataset.attrs = attr_mgr.get_global_attributes(logical_source)

        # TODO: Add CDF attributes to yaml once they're defined for L2 science data
        #  consider moving attribute handling to hit_utils.py
        # Assign attributes and dimensions to each data array in the Dataset
        for field in dataset.data_vars.keys():
            try:
                # Create a dict of dimensions using the DEPEND_I keys in the
                # attributes
                dims = {
                    key: value
                    for key, value in attr_mgr.get_variable_attributes(field).items()
                    if "DEPEND" in key
                }
                dataset[field].attrs = attr_mgr.get_variable_attributes(field)
                dataset[field].assign_coords(dims)
            except KeyError:
                # TODO: consider raising an error after L2 attributes are defined.
                #  Until then, continue with processing and log warning
                logger.warning(f"Field {field} not found in attribute manager.")

        # Skip schema check for epoch to prevent attr_mgr from adding the
        # DEPEND_0 attribute which isn't required for epoch
        dataset.epoch.attrs = attr_mgr.get_variable_attributes(
            "epoch", check_schema=False
        )

        logger.info(f"HIT L2 dataset created for {logical_source}")

    return list(l2_datasets.values())


def calculate_intensities(
    rates: xr.DataArray,
    factors: xr.Dataset,
) -> xr.DataArray:
    """
    Calculate the intensities for given rates and equation factors.

    Uses vectorization to calculate the intensities for an array of rates
    for all epochs.

        This function uses equation 9 and 12 from the HIT algorithm document:
        ((Summed L1B Rates) / (Delta Time * Delta E * Geometry Factor * Efficiency)) - b

    Parameters
    ----------
    rates : xr.DataArray
        The L1B rates to be converted to intensities.
    factors : xr.Dataset
        The ancillary data factors needed to calculate the intensity.
        This includes delta_e, geometry_factor, efficiency, and b.

    Returns
    -------
    xr.DataArray
        The calculated intensities for all epochs.
    """
    # Calculate the intensity using vectorized operations
    intensity = (
        rates
        / (
            factors.delta_time
            * factors.delta_e
            * factors.geometry_factor
            * factors.efficiency
        )
    ) - factors.b

    # Apply intensity where rates are not equal to the fill value
    intensity = xr.where(rates == FILLVAL_FLOAT32, FILLVAL_FLOAT32, intensity)

    return intensity


def reshape_for_sectored(arr: np.ndarray) -> np.ndarray:
    """
    Reshape the ancillary data for sectored rates.

    Reshape the 3D arrays (epoch, energy, declination) to 4D arrays
    (epoch, energy, azimuth, declination) by repeating the data
    along the azimuth dimension. This is done to match the dimensions
    of the sectored rates data to allow for proper calculation of
    intensities.

    Parameters
    ----------
    arr : np.ndarray
        The ancillary data array to reshape.

    Returns
    -------
    np.ndarray
        The reshaped array.
    """
    return np.repeat(
        arr.reshape((arr.shape[0], arr.shape[1], arr.shape[2]))[:, :, np.newaxis, :],
        N_AZIMUTH,
        axis=2,
    )


def build_ancillary_dataset(
    delta_e: np.ndarray,
    geometry_factors: np.ndarray,
    efficiencies: np.ndarray,
    b: np.ndarray,
    species_array: xr.DataArray,
) -> xr.Dataset:
    """
    Build a xarray Dataset containing ancillary data for calculating intensity.

    This function builds a dataset containing the factors needed for calculating
    intensity for a given species. The dataset is built based on the dimensions
    and coordinates of the species data to align data along the epoch dimension.

    Parameters
    ----------
    delta_e : np.ndarray
        Delta E values which are energy bin widths.
    geometry_factors : np.ndarray
        Geometry factor values.
    efficiencies : np.ndarray
        Efficiency values.
    b : np.ndarray
        Background intensity values.
    species_array : xr.Dataset
        Data array for the species to extract coordinates from.

    Returns
    -------
    ancillary_ds : xr.Dataset
        A dataset containing all ancillary data variables and coordinates that
        align with the L2 dataset.
    """
    data_vars = {}

    # Check if this is sectored data (i.e., has azimuth and declination dims)
    is_sectored = (
        "declination" in species_array.dims or "declination" in species_array.coords
    )

    # Build variables
    data_vars["delta_e"] = (species_array.dims, delta_e)
    data_vars["geometry_factor"] = (
        species_array.dims,
        geometry_factors,
    )
    data_vars["efficiency"] = (
        species_array.dims,
        efficiencies,
    )
    data_vars["b"] = (species_array.dims, b)
    data_vars["delta_time"] = (
        ["epoch"],
        np.full(
            len(species_array.epoch),
            SECONDS_PER_10_MIN if is_sectored else SECONDS_PER_MIN,
        ),
    )

    return xr.Dataset(data_vars, coords=species_array.coords)


def calculate_intensities_for_a_species(
    species_variable: str, l2_dataset: xr.Dataset, ancillary_data_frames: dict
) -> xr.Dataset:
    """
    Calculate the intensity for a given species in the dataset.

    This function orchestrates calculating the intensity for a given species
    in the L2 dataset using ancillary data determined by the dynamic threshold
    state (0-3).

    The intensity is calculated using the equation:
        (L1B Rates) / (Delta Time * Delta E * Geometry Factor * Efficiency) - b

        where the equation factors are retrieved from the ancillary data for
        the given species and dynamic threshold states.

    Parameters
    ----------
    species_variable : str
        The species variable to calculate the intensity for, which is either the species
        or a statistical uncertainty.
        (i.e. "h", "h_stat_uncert_minus", or "h_stat_uncert_plus").
    l2_dataset : xr.Dataset
        The L2 dataset containing the L1B rates needed to calculate the intensity.
    ancillary_data_frames : dict
        Dictionary containing ancillary data for each dynamic threshold state where
        the key is the dynamic threshold state and the value is a pandas DataFrame
        containing the ancillary data for all species.

    Returns
    -------
    updated_ds : xr.Dataset
        The updated dataset with intensities calculated for the given species.
    """
    updated_ds = l2_dataset.copy()
    dynamic_threshold_states = updated_ds["dynamic_threshold_state"].values
    unique_states = np.unique(dynamic_threshold_states)
    species_name = (
        species_variable.split("_")[0]
        if "_uncert_" in species_variable
        else species_variable
    )

    # Subset ancillary data for this species
    species_ancillary_by_state = {
        state: get_species_ancillary_data(state, ancillary_data_frames, species_name)
        for state in unique_states
    }

    # Extract parameters - 3D arrays (num_states, energy bins, values)
    delta_e = np.stack(
        [
            species_ancillary_by_state[state]["delta_e"]
            for state in dynamic_threshold_states
        ]
    )
    geometry_factors = np.stack(
        [
            species_ancillary_by_state[state]["geometry_factor"]
            for state in dynamic_threshold_states
        ]
    )
    efficiencies = np.stack(
        [
            species_ancillary_by_state[state]["efficiency"]
            for state in dynamic_threshold_states
        ]
    )
    b = np.stack(
        [species_ancillary_by_state[state]["b"] for state in dynamic_threshold_states]
    )

    # Reshape parameters for sectored rates to 4D arrays
    if "declination" in updated_ds[species_variable].dims:
        delta_e = reshape_for_sectored(delta_e)
        geometry_factors = reshape_for_sectored(geometry_factors)
        efficiencies = reshape_for_sectored(efficiencies)
        b = reshape_for_sectored(b)

    # Reshape parameters for summed and standard rates to 2D arrays
    # by removing last dimension of size one, (n, n, 1)
    else:
        delta_e = np.squeeze(delta_e, axis=-1)
        geometry_factors = np.squeeze(geometry_factors, axis=-1)
        efficiencies = np.squeeze(efficiencies, axis=-1)
        b = np.squeeze(b, axis=-1)

    # Build ancillary xarray dataset
    ancillary_ds = build_ancillary_dataset(
        delta_e, geometry_factors, efficiencies, b, l2_dataset[species_name]
    )

    # Calculate intensities
    updated_ds[species_variable] = calculate_intensities(
        updated_ds[species_variable], ancillary_ds
    )

    return updated_ds


def calculate_intensities_for_all_species(
    l2_dataset: xr.Dataset, ancillary_data_frames: dict, valid_data_variables: list
) -> xr.Dataset:
    """
    Calculate the intensity for each species in the dataset.

    Parameters
    ----------
    l2_dataset : xr.Dataset
        The L2 dataset.
    ancillary_data_frames : dict
        Dictionary containing ancillary data for each dynamic threshold state
        where the key is the dynamic threshold state and the value is a pandas
        DataFrame containing the ancillary data.
    valid_data_variables : list
        A list of valid data variables to calculate intensity for.

    Returns
    -------
    updated_ds : xr.Dataset
        The updated dataset with the intensity calculated for each species.
    """
    updated_ds = l2_dataset.copy()

    # Add statistical uncertainty variables to the list of valid variables
    data_variables = (
        valid_data_variables
        + [f"{var}_stat_uncert_minus" for var in valid_data_variables]
        + [f"{var}_stat_uncert_plus" for var in valid_data_variables]
    )

    # Calculate the intensity for each valid data variable
    for species_variable in data_variables:
        if species_variable in updated_ds.data_vars:
            updated_ds = calculate_intensities_for_a_species(
                species_variable, updated_ds, ancillary_data_frames
            )
        else:
            logger.warning(
                f"Variable {species_variable} not found in dataset. "
                f"Skipping intensity calculation."
            )

    return updated_ds


def add_systematic_uncertainties(dataset: xr.Dataset, particle: str) -> xr.Dataset:
    """
    Add systematic uncertainties to the dataset.

    Add systematic uncertainties to the dataset. Just zeros for now.
    To change if/when HIT determines there are systematic uncertainties.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to add the systematic uncertainties to
        which contain the particle data variables.
    particle : str
        The particle name.

    Returns
    -------
    updated_ds : xr.Dataset
        The dataset with the systematic uncertainties added.
    """
    updated_ds = dataset.copy()

    updated_ds[f"{particle}_sys_err_minus"] = xr.DataArray(
        data=np.zeros(updated_ds[particle].shape, dtype=np.float32),
        dims=updated_ds[particle].dims,
        name=f"{particle}_sys_err_minus",
    )
    updated_ds[f"{particle}_sys_err_plus"] = xr.DataArray(
        data=np.zeros(updated_ds[particle].shape, dtype=np.float32),
        dims=updated_ds[particle].dims,
        name=f"{particle}_sys_err_plus",
    )

    return updated_ds


def add_total_uncertainties(dataset: xr.Dataset, particle: str) -> xr.Dataset:
    """
    Add total uncertainties to the dataset.

    This function calculates the total uncertainties for a given particle
    by combining the statistical uncertainties and systematic uncertainties.

    The total uncertainties are calculated as the square root of the sum
    of the squares of the statistical and systematic uncertainties.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to add the total uncertainties to.
    particle : str
        The particle name.

    Returns
    -------
    updated_ds : xr.Dataset
        The dataset with the total uncertainties added.
    """
    updated_ds = dataset.copy()

    # Calculate the total uncertainties
    total_minus = np.sqrt(
        np.square(updated_ds[f"{particle}_stat_uncert_minus"])
        + np.square(updated_ds[f"{particle}_sys_err_minus"])
    )
    total_plus = np.sqrt(
        np.square(updated_ds[f"{particle}_stat_uncert_plus"])
        + np.square(updated_ds[f"{particle}_sys_err_plus"])
    )

    updated_ds[f"{particle}_total_uncert_minus"] = xr.DataArray(
        data=total_minus.astype(np.float32),
        dims=updated_ds[particle].dims,
        name=f"{particle}_total_uncert_minus",
    )
    updated_ds[f"{particle}_total_uncert_plus"] = xr.DataArray(
        data=total_plus.astype(np.float32),
        dims=updated_ds[particle].dims,
        name=f"{particle}_total_uncert_plus",
    )

    return updated_ds


def get_species_ancillary_data(
    dynamic_threshold_state: int, ancillary_data_frames: dict, species: str
) -> dict:
    """
    Get the ancillary data for a given species and dynamic threshold state.

    Parameters
    ----------
    dynamic_threshold_state : int
        The dynamic threshold state for the ancillary data (0-3).
    ancillary_data_frames : dict
        Dictionary containing ancillary data for each dynamic threshold state
        where the key is the dynamic threshold state and the value is a pandas
        DataFrame containing the ancillary data.
    species : str
        The species to get the ancillary data for.

    Returns
    -------
    dict
        The ancillary data for the species and dynamic threshold state.
    """
    ancillary_df = ancillary_data_frames[dynamic_threshold_state]

    # Remove any trailing spaces from all values in the DataFrame
    ancillary_df = ancillary_df.map(lambda x: x.strip() if isinstance(x, str) else x)

    # Get the ancillary data for the species and group by lower energy
    species_ancillary_df = ancillary_df[ancillary_df["species"] == species]
    grouped = species_ancillary_df.groupby("lower energy (mev)")
    return {
        "delta_e": np.array(grouped["delta e (mev)"].apply(list).tolist()),
        "geometry_factor": np.array(
            grouped["geometry factor (cm2 sr)"].apply(list).tolist()
        ),
        "efficiency": np.array(grouped["efficiency"].apply(list).tolist()),
        "b": np.array(grouped["b"].apply(list).tolist()),
    }


def load_ancillary_data(dynamic_threshold_states: set, path_prefix: Path) -> dict:
    """
    Load ancillary data based on the dynamic threshold state.

    The dynamic threshold state (0-3) determines which ancillary file to use.
    This function returns a dictionary with ancillary data for each state in
    the dataset.

    Parameters
    ----------
    dynamic_threshold_states : set
        A set of dynamic threshold states in the L2 dataset.
    path_prefix : Path
        The path prefix for ancillary data files.

    Returns
    -------
    dict
        A dictionary with ancillary data for each dynamic threshold state.
    """
    # Load ancillary data
    ancillary_data_frames = {
        int(state): pd.read_csv(f"{path_prefix}{state}-factors_20250219_v002.csv")
        for state in dynamic_threshold_states
    }

    # Convert column names and species values to lowercase
    for df in ancillary_data_frames.values():
        df.columns = df.columns.str.lower().str.strip()
        df["species"] = df["species"].str.lower()

    return ancillary_data_frames


def process_summed_intensity_data(l1b_summed_rates_dataset: xr.Dataset) -> xr.Dataset:
    """
    Will process L2 HIT summed intensity data from L1B summed rates.

    This function converts the L1B summed rates to L2 summed intensities
    using ancillary tables containing factors needed to calculate the
    intensity (energy bin width, geometry factor, efficiency, and b).

    Equation 12 from the HIT algorithm document:
    Summed Intensity = (L1B Summed Rate) /
                       (60 * Delta E * Geometry Factor * Efficiency) - b

    Parameters
    ----------
    l1b_summed_rates_dataset : xarray.Dataset
        HIT L1B summed rates dataset.

    Returns
    -------
    xr.Dataset
        The processed L2 summed intensity dataset.
    """
    # Create a new dataset to store the L2 summed intensity data
    l2_summed_intensity_dataset = l1b_summed_rates_dataset.copy(deep=True)

    # Load ancillary data for each dynamic threshold state into a dictionary
    ancillary_data_frames = load_ancillary_data(
        set(l2_summed_intensity_dataset["dynamic_threshold_state"].values),
        L2_SUMMED_ANCILLARY_PATH_PREFIX,
    )

    # Calculate the intensity for each species
    l2_summed_intensity_dataset = calculate_intensities_for_all_species(
        l2_summed_intensity_dataset, ancillary_data_frames, VALID_SPECIES
    )

    # Add total and systematic uncertainties to the dataset
    for var in l2_summed_intensity_dataset.data_vars:
        if var in VALID_SPECIES:
            l2_summed_intensity_dataset = add_systematic_uncertainties(
                l2_summed_intensity_dataset, var
            )
            l2_summed_intensity_dataset = add_total_uncertainties(
                l2_summed_intensity_dataset, var
            )

    return l2_summed_intensity_dataset


def process_standard_intensity_data(
    l1b_standard_rates_dataset: xr.Dataset,
) -> xr.Dataset:
    """
    Will process L2 standard intensity data from L1B standard rates data.

    This function converts L1B standard rates to L2 standard intensities for each
    particle type and energy range using ancillary tables containing factors
    needed to calculate the intensity (energy bin width, geometry factor, efficiency
    and b).

    First, rates from the l2fgrates, l3fgrates, and penfgrates data variables
    in the L1B standard rates data are summed. These variables represent rates
    for different detector penetration ranges (Range 2, Range 3, and Range 4
    respectively). Only the energy ranges specified in the
    STANDARD_PARTICLE_ENERGY_RANGE_MAPPING dictionary are included in this
    product.

    Intensity is then calculated from the summed standard rates:

        Equation 9 from the HIT algorithm document:
        Standard Intensity = (Summed L1B Standard Rates) /
                             (60 * Delta E * Geometry Factor * Efficiency) - b

    Parameters
    ----------
    l1b_standard_rates_dataset : xr.Dataset
        The L1B standard rates dataset.

    Returns
    -------
    xr.Dataset
        The L2 standard intensity dataset.
    """
    # Create a new dataset to store the L2 standard intensity data
    l2_standard_intensity_dataset = xr.Dataset()

    # Assign the epoch coordinate from the l1B dataset
    l2_standard_intensity_dataset = l2_standard_intensity_dataset.assign_coords(
        {"epoch": l1b_standard_rates_dataset.coords["epoch"]}
    )

    # Add dynamic threshold state to the dataset
    l2_standard_intensity_dataset["dynamic_threshold_state"] = (
        l1b_standard_rates_dataset["dynamic_threshold_state"]
    )

    # Load ancillary data for each dynamic threshold state into a dictionary
    ancillary_data_frames = load_ancillary_data(
        set(l2_standard_intensity_dataset["dynamic_threshold_state"].values),
        L2_STANDARD_ANCILLARY_PATH_PREFIX,
    )

    # Process each particle type and add rates and uncertainties to the dataset
    for particle, energy_ranges in STANDARD_PARTICLE_ENERGY_RANGE_MAPPING.items():
        # Add standard particle rates and statistical uncertainties to the dataset
        l2_standard_intensity_dataset = add_summed_particle_data_to_dataset(
            l2_standard_intensity_dataset,
            l1b_standard_rates_dataset,
            particle,
            energy_ranges,
        )

    l2_standard_intensity_dataset = calculate_intensities_for_all_species(
        l2_standard_intensity_dataset, ancillary_data_frames, VALID_SPECIES
    )

    # Add total and systematic uncertainties to the dataset
    for particle in STANDARD_PARTICLE_ENERGY_RANGE_MAPPING.keys():
        l2_standard_intensity_dataset = add_systematic_uncertainties(
            l2_standard_intensity_dataset, particle
        )
        l2_standard_intensity_dataset = add_total_uncertainties(
            l2_standard_intensity_dataset, particle
        )

    return l2_standard_intensity_dataset


def process_sectored_intensity_data(
    l1b_sectored_rates_dataset: xr.Dataset,
) -> xr.Dataset:
    """
    Will process L2 HIT sectored intensity data from L1B sectored rates data.

    This function converts the L1B sectored rates to L2 sectored intensities
    using ancillary tables containing factors needed to calculate the
    intensity (energy bin width, geometry factor, efficiency, and b).

    Equation 12 from the HIT algorithm document:
    Sectored Intensity = (Summed L1B Sectored Rates) /
                       (600 * Delta E * Geometry Factor * Efficiency) - b

    Parameters
    ----------
    l1b_sectored_rates_dataset : xr.Dataset
        The L1B sectored rates dataset.

    Returns
    -------
    xr.Dataset
        The processed L2 sectored intensity dataset.
    """
    # Create a new dataset to store the L2 sectored intensity data
    l2_sectored_intensity_dataset = l1b_sectored_rates_dataset.copy(deep=True)

    # Load ancillary data for each dynamic threshold state into a dictionary
    ancillary_data_frames = load_ancillary_data(
        set(l2_sectored_intensity_dataset["dynamic_threshold_state"].values),
        L2_SECTORED_ANCILLARY_PATH_PREFIX,
    )

    # Calculate the intensity for each species
    l2_sectored_intensity_dataset = calculate_intensities_for_all_species(
        l2_sectored_intensity_dataset, ancillary_data_frames, VALID_SECTORED_SPECIES
    )

    # Add total and systematic uncertainties to the dataset
    for var in l2_sectored_intensity_dataset.data_vars:
        if var in VALID_SECTORED_SPECIES:
            l2_sectored_intensity_dataset = add_systematic_uncertainties(
                l2_sectored_intensity_dataset, var
            )
            l2_sectored_intensity_dataset = add_total_uncertainties(
                l2_sectored_intensity_dataset, var
            )

    return l2_sectored_intensity_dataset
