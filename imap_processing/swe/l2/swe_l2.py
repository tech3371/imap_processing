"""SWE L2 processing module."""

import xarray as xr


def calculate_particle_energy() -> None:
    """Read the lookup table and calculate the particle energy."""
    # I think here, it's just looking up energy values used
    # per full cycle (720 counts -> ())
    # return read_lookup_table(table_index_value=0)


def calculate_phase_space_density(l1b_dataset: xr.Dataset) -> None:
    """
    Calculate phase space density.

    Calculate phase space density fv in units of s^3/cm^6

        fv(v,theta,phi) = 2*C(E,theta,phi)/(G*v^4*tau)

        C = corrected counts
        E = electron energy, in eV
        v = electron speed, computed from energy, in cm/s
        tau = sampling time, in s
        G = geometric factor, in (cm^2 * ster)

    Parameters
    ----------
    l1b_dataset : xr.Dataset
        The L1B dataset to process.
    """
    # 1. L1B contains corrected counts rate and not the counts.
    #    Is that ok? TODO: ask Ruth.
    # 2. For each full sweep data, read energy values from LUT.
    # 3. Calculate the electron speed from energy. v^4 = 1.237e31 * energy[i, j, k]**2
    #    TODO: What is the constant number?
    # 4. Sampling time of each data point is already stored in L1B dataset
    #    as 'acquisition_time' variable.
    # 5. Geometric factor. TODO: Is this the seven constant values for the CEMs?
    # 6. Calculate phase space density fv using this formula:
    #    fv(v,theta,phi) = 2*C(E,theta,phi)/(G*v^4*tau)
    #    fv[i, j, k] = 2 * ccounts[i, j, k] / (gg[j] * tsampl * vto4)
    #    TODO: how to do with differently with count rate data?

    # Geometric factors for the CEMs
    # gg = np.array(
    # [255.6e-6, 511.0e-6, 633.4e-6, 659.5e-6, 733.5e-6, 540.3e-6, 272.9e-6]
    # )

    # L1B science data is store in this dimension:
    #   [energy, angle, cem]
    # document code uses this:
    #   N_ENERGIES, N_CEMS, N_PHI

    # TODO: what is this for?
    # # Special case for the first and last CEMS
    # for i in range(N_ENERGIES):
    #     for k in range(N_PHI):
    #         if k % 2 == 0:
    #             fv[i, 0, k] = 0.5 * fv[i, 0, k + 1]
    #             fv[i, 6, k] = 0.5 * fv[i, 6, k + 1]
    #         else:
    #             fv[i, 0, k] = 0.5 * fv[i, 0, k]
    #             fv[i, 6, k] = 0.5 * fv[i, 6, k]

    # return np.zeros((1, 1, 1))


def calculate_electron_flux() -> None:
    """
    Calculate the electron flux.

    Calculation of Electron Flux: In the heritage code,
    electron distributions are always saved in units of
    phase space distribution rather than flux. We note that
    distribution function f (units s3 cm-6) can easily be
    converted to differential flux (units 1/(cm2 s ster eV))
    by j = p2 f, where p is the momentum p2 = 2mE,
    where m = particle mass and E = particle energy.

    mass of electron = 9.10938356e-31 kg
    9.10938 x 10^-31 kg
    """
    pass


def calculate_spin_phase() -> None:
    """
    Calculate the spin phase.

    calculate spin phase using spin data from universal spin data
    and use acquisition_time from L1B dataset.
    """
    pass


def swe_l2(l1b_dataset: xr.Dataset, data_version: str) -> xr.Dataset:
    """
    Calculate the L2 data products for the SWE instrument.

    Parameters
    ----------
    l1b_dataset : xr.Dataset
        The L1B dataset to process.
    data_version : str
        The version of the data.

    Returns
    -------
    l2_dataset : xr.Dataset
        The L2 dataset.
    """
    calculate_phase_space_density(l1b_dataset)

    # TODO:
    # Finally organize data by energy and spin phase. What does this mean
    # to go through [energy, angle, cem] to above dimensions?
    return l1b_dataset
