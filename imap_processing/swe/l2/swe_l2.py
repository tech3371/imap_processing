"""SWE L2 processing module."""

import numpy.typing as npt
import xarray as xr

from imap_processing.swe.utils.swe_utils import read_lookup_table

# TODO: add these to instrument status summary
ENERGY_CONVERSION_FACTOR = 4.75
# 7 CEMs geometric factors in cm^2 sr eV/eV units.
GEOMETRIC_FACTORS = [
    435e-6,
    599e-6,
    808e-6,
    781e-6,
    876e-6,
    548e-6,
    432e-6,
]
ELECTRON_MASS = 9.10938356e-31  # kg


def calculate_particle_energy(esa_table_num: int) -> npt.NDArray:
    """
    Calculate particle energy.

    To convert Volts to Energy, multiply ESA voltage in Volts by 4.75 to get
    electron energy in eV.

    Parameters
    ----------
    esa_table_num : int
        ESA table number.

    Returns
    -------
    energy : np.ndarray
        720 step energy values.
    """
    # The lookup table gives voltage applied to analyzers.
    esa_table = read_lookup_table()

    # Convert voltage to electron energy in eV by apply conversion factor.
    energy = esa_table["esa_v"].values * ENERGY_CONVERSION_FACTOR
    return energy


def calculate_phase_space_density(l1b_dataset: xr.Dataset) -> None:
    """
    Convert counts to phase space density.

    Calculate phase space density, fv, in units of s^3/cm^6
        fv = 2 * (C/tau) / (G * v^4)
        where:
        C / tau = corrected count rate
        E = electron energy, in eV.
            (result from caluclate_particle_energy() function)
        v = electron speed, computed from energy, in cm/s
            E = 0.5 * m * v^2
            where m is mass of electron and v = sqrt(2 * E / m).
        G = geometric factor, in (cm^2 * ster). 7 CEMS value.

    Parameters
    ----------
    l1b_dataset : xr.Dataset
        The L1B dataset to process.
    """
    # L1B contains corrected counts rate and not the counts.
    #    Is that ok? TODO: ask Ruth.

    # # Calculate particle energy and then electron speed for all unique ESA
    # # table number.
    # esa_table_nums = np.unique(l1b_dataset["esa_table_num"].values)
    # particle_energy = {k: calculate_particle_energy(k) for k in esa_table_nums}
    # electron_speed = {
    #     k: np.sqrt(2 * energy / ELECTRON_MASS) for k,
    # energy in particle_energy.items()
    # }

    # # Electron speed has shape (720, ) as a result of how lookup table is.
    # # Convert (720) to (24, 30) to match dimension of
    # # L1B dataset, (epoch, 24, 30, 7).
    # electron_speed = {k: speed.reshape(24, 30) for k, speed in electron_speed.items()}

    # # Calculate phase space density.
    # for esa_table_num in electron_speed:
    #     # Look up all indices of data whose esa_table_num is equal to
    #     # esa_table_num. Use first quarter cycle's esa_table_num.
    #     indices = np.where(l1b_dataset
    # ["esa_table_num"].values[:, 0] == esa_table_num)[
    #         0
    #     ]
    #     # Calculate phase space density for all full sweep data
    #     # that has current esa_table_num.
    #     data = l1b_dataset.isel(epoch=indices)
    #     print(data)
    #     # TODO: figure out how to apply it.
    pass
