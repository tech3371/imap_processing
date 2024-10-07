import numpy as np
import xarray as xr

from imap_processing.swe.l2.swe_l2 import (
    calculate_particle_energy,
    calculate_phase_space_density,
)


def test_calculate_particle_energy():
    """Test calculate_particle_energy function."""
    esa_table_num = 1
    energy = calculate_particle_energy(esa_table_num)
    assert energy.shape == (1440,)


def test_calculate_phase_space_density():
    """Test calculate_phase_space_density function."""
    # Create a dummy l1b dataset
    total_sweeps = 2
    np.random.seed(0)
    l1b_dataset = xr.Dataset(
        {
            "science_data": (
                ["epoch", "energy", "angle", "cem"],
                np.full((total_sweeps, 24, 30, 7), 1),
            ),
            "acq_duration": (["epoch", "cycle"], np.full((total_sweeps, 4), 80.0)),
            "esa_table_num": (
                ["epoch", "cycle"],
                np.random.randint(0, 2, total_sweeps * 4).reshape(total_sweeps, 4),
            ),
        }
    )
    print(l1b_dataset)
    calculate_phase_space_density(l1b_dataset)
    assert True
