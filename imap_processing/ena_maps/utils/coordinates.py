"""Define utils and classes related to coordinates of the ENA maps."""

from enum import Enum


class CoordNames(Enum):
    """Enumeration of the names of the coordinates in the L1C and L2 ENA datasets."""

    TIME = "epoch"
    ENERGY = "energy"
    HEALPIX_INDEX = "healpix_index"

    # The names of the az/el angular coordinates may differ between L1C and L2 data
    AZIMUTH_L1C = "longitude"
    ELEVATION_L1C = "latitude"
    AZIMUTH_L2 = "longitude"
    ELEVATION_L2 = "latitude"
