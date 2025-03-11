"""Define classes for handling pointing sets and maps for ENA data."""

from __future__ import annotations

import logging
import pathlib
from abc import ABC, abstractmethod
from enum import Enum

import astropy_healpix.healpy as hp
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from imap_processing.cdf.utils import load_cdf
from imap_processing.ena_maps.utils import map_utils, spatial_utils
from imap_processing.spice import geometry
from imap_processing.spice.time import ttj2000ns_to_et

logger = logging.getLogger(__name__)


class SkyTilingType(Enum):
    """Enumeration of the types of tiling used in the ENA maps."""

    RECTANGULAR = "Rectangular"
    HEALPIX = "Healpix"


class IndexMatchMethod(Enum):
    """
    Enumeration of the types of index matching methods used in the ENA sky maps.

    Notes
    -----
    Index matching is the process of determining which pixels in a map grid correspond
    to which pixels in a pointing set grid. The Ultra instrument team has determined
    that they must support two methods of index matching for rectangular grid maps:

    **Push Method**

    The "push" method takes each pixel in a pointing set and transforms its coordinates
    to the frame of the map, then determines into which pixel in the map grid the
    transformed pointing set pixel falls.
    This method ensures that all pointing set pixels (and thus all counts) are
    captured in the map, but does not ensure that all pixels in the map receive data.

    **Pull Method**

    The "pull" method takes each pixel in the map grid and transforms its coordinates
    to the frame of the pointing set, then determines into which pixel in the
    pointing set grid the transformed map pixel falls.
    This method ensures that all pixels in the map receive data, but can result in
    some pointing set pixels not being captured in the map, and others being captured
    multiple times.
    """

    PUSH = "Push"
    PULL = "Pull"


def match_coords_to_indices(
    input_object: PointingSet | AbstractSkyMap,
    output_object: PointingSet | AbstractSkyMap,
    event_et: float | None = None,
) -> NDArray:
    """
    Find the output indices corresponding to each input coord between 2 spatial objects.

    First, the pixel center coordinates of the input spatial object are
    transformed from the Spice coordinate frame of the input object to their
    corresponding coordinates in the Spice frame of the output object.
    Then, the transformed pixel centers are matched to the 1D indices of the spatial
    pixels in the output frame, either in an unwrapped rectangular grid or a Healpix
    tessellation of the sky.

    This function always "pushes" the pixels of the input object to corresponding pixels
    in the output object's unwrapped rectangular grid or healpix tessellation;
    however, by swapping the input and output objects, one can apply the "pull" method
    of index  matching.

    At present, the allowable inputs are either:
    - A PointingSet object and a SkyMap object, in either order of input/output.
    The event time will be taken from the PointingSet object.
    - Two SkyMap objects, in which case the event time must be specified.

    Parameters
    ----------
    input_object : PointingSet | AbstractSkyMap
        An object containing 1D spatial pixel centers in azimuth and elevation,
        which will be matched to 1D indices of spatial pixels in the output frame.
        Must contain the Spice frame in which the pixel centers are defined.
    output_object : PointingSet | AbstractSkyMap
        The object containing a grid or tessellation of spatial pixels
        into which the input spatial pixel centers will 'land', and be matched to
        corresponding pixel 1D indices in the output frame.
    event_et : float, optional
        Event time at which to transform the input spatial object to the output frame.
        This can be manually specified, e.g., for converting between Maps which do not
        contain an epoch value.
        If specified, must be in SPICE compatible ET.
        The default value is None, in which case the event time of the PointingSet
        object is used.

    Returns
    -------
    flat_indices_input_grid_output_frame : NDArray
        1D array of pixel indices of the output object corresponding to each pixel in
        the input object. The length of the array is equal to the number of pixels in
        the input object, and may contain 0, 1, or multiple occurrences of the same
        output index.

    Raises
    ------
    ValueError
        If both input and output objects are PointingSet objects.
    ValueError
        If the event time is not specified and both objects are SkyMaps.
    NotImplementedError
        If the output tiling type is HEALPIX. Will be implemented in the future.
    ValueError
        If the tiling type of the output frame is not RECTANGULAR or HEALPIX.
    """
    if isinstance(input_object, PointingSet) and isinstance(output_object, PointingSet):
        raise ValueError("Cannot match indices between two PointingSet objects.")

    # If event_et is not specified, use epoch of the PointingSet, if present.
    # The epoch will be in units of terrestrial time (TT) J2000 nanoseconds,
    # which must be converted to ephemeris time (ET) for SPICE.
    if event_et is None:
        if isinstance(input_object, PointingSet):
            event_et = ttj2000ns_to_et(input_object.data["epoch"].values)
        elif isinstance(output_object, PointingSet):
            event_et = ttj2000ns_to_et(output_object.data["epoch"].values)
        else:
            raise ValueError(
                "Event time must be specified if both objects are SkyMaps."
            )

    # Az/El pixel center coords of the input object in its own frame
    input_obj_az_el_input_frame = input_object.az_el_points

    # Transform the input pixel centers to the output frame
    input_obj_az_el_output_frame = geometry.frame_transform_az_el(
        et=event_et,
        az_el=input_obj_az_el_input_frame,
        from_frame=input_object.spice_reference_frame,
        to_frame=output_object.spice_reference_frame,
        degrees=True,
    )

    # The way indices are matched depends on the tiling type of the 2nd object
    if output_object.tiling_type is SkyTilingType.RECTANGULAR:
        # To match to a rectangular grid, we need to digitize the transformed az, el
        # pixel centers onto the bin edges of the output frame's grid, then
        # use ravel_multi_index to get the 1D indices of the pixels in the output frame.
        az_indices = (
            np.digitize(
                input_obj_az_el_output_frame[:, 0],
                output_object.sky_grid.az_bin_edges,
            )
            - 1
        )
        el_indices = (
            np.digitize(
                input_obj_az_el_output_frame[:, 1],
                output_object.sky_grid.el_bin_edges,
            )
            - 1
        )
        flat_indices_input_grid_output_frame = np.ravel_multi_index(
            multi_index=(az_indices, el_indices),
            dims=(
                len(output_object.sky_grid.az_bin_midpoints),
                len(output_object.sky_grid.el_bin_midpoints),
            ),
        )

    elif output_object.tiling_type is SkyTilingType.HEALPIX:
        # To match to a Healpix tessellation, we need to use the healpy function ang2pix
        # which directly returns the index on the output frame's Healpix tessellation.
        flat_indices_input_grid_output_frame = hp.ang2pix(
            nside=output_object.nside,
            theta=input_obj_az_el_output_frame[:, 0],  # Lon in degrees
            phi=input_obj_az_el_output_frame[:, 1],  # Lat in degrees
            nest=output_object.nested,
            lonlat=True,
        )
    else:
        raise ValueError(
            "Tiling type of the output frame must be either RECTANGULAR or HEALPIX."
            f"Received: {output_object.tiling_type}"
        )

    return flat_indices_input_grid_output_frame


# Define the pointing set classes
class PointingSet(ABC):
    """
    Abstract class to contain pointing set (PSET) data in the context of ENA sky maps.

    Any spatial axes - (azimuth, elevation) for Rectangularly gridded tilings or
    (pixel index) for Healpix - must be stored in the last axis/axes of each data array.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset containing the pointing set data.
    spice_reference_frame : geometry.SpiceFrame
        The reference Spice frame of the pointing set.
    """

    @abstractmethod
    def __init__(self, dataset: xr.Dataset, spice_reference_frame: geometry.SpiceFrame):
        """Abstract method to initialize the pointing set object."""
        self.spice_reference_frame = spice_reference_frame
        self.num_points = 0
        self.az_el_points = np.zeros((self.num_points, 2))
        self.data = xr.Dataset()

    def __repr__(self) -> str:
        """
        Return a string representation of the pointing set.

        Returns
        -------
        str
            String representation of the pointing set.
        """
        return (
            f"{self.__class__.__name__} PointingSet"
            f"(spice_reference_frame={self.spice_reference_frame})"
        )


class UltraPointingSet(PointingSet):
    """
    PSET object specifically for ULTRA data, nominally at Level 1C.

    Parameters
    ----------
    l1c_dataset : xr.Dataset | pathlib.Path | str
        L1c xarray dataset containing the pointing set data or the path to the dataset.
        Currently, the dataset is expected to be in a rectangular grid,
        with data_vars indexed along the coordinates:
            - 'epoch' : time value (1 value per PSET)
            - 'azimuth_bin_center' : azimuth bin center values
            - 'elevation_bin_center' : elevation bin center values
        Some data_vars may additionally be indexed by energy bin;
        however, only the spatial axes are used in this class.
    spice_reference_frame : geometry.SpiceFrame
        The reference Spice frame of the pointing set. Default is IMAP_DPS.

    Raises
    ------
    ValueError
        If the azimuth or elevation bin centers do not match the constructed grid.
        Or if the azimuth or elevation bin spacing is not uniform.
    ValueError
        If multiple epochs are found in the dataset.
    """

    def __init__(
        self,
        l1c_dataset: xr.Dataset | pathlib.Path | str,
        spice_reference_frame: geometry.SpiceFrame = geometry.SpiceFrame.IMAP_DPS,
    ):
        # Store the reference frame of the pointing set
        self.spice_reference_frame = spice_reference_frame

        # Read in the data and store the xarray dataset as data attr
        if isinstance(l1c_dataset, (str, pathlib.Path)):
            self.data = load_cdf(pathlib.Path(l1c_dataset))
        elif isinstance(l1c_dataset, xr.Dataset):
            self.data = l1c_dataset

        # A PSET must have a single epoch
        self.epoch = self.data["epoch"].values
        if len(np.unique(self.epoch)) > 1:
            raise ValueError("Multiple epochs found in the dataset.")

        # The rest of the constructor handles the rectangular grid
        # aspects of the Ultra PSET.
        # NOTE: This may be changed to Healpix tessellation in the future
        self.tiling_type = SkyTilingType.RECTANGULAR

        # Ensure 1D axes grids are uniformly spaced,
        # then set spacing based on data's azimuth bin spacing.
        az_bin_delta = np.diff(self.data["azimuth_bin_center"])
        el_bin_delta = np.diff(self.data["elevation_bin_center"])
        if not np.allclose(az_bin_delta, az_bin_delta[0], atol=1e-10, rtol=0):
            raise ValueError("Azimuth bin spacing is not uniform.")
        if not np.allclose(el_bin_delta, el_bin_delta[0], atol=1e-10, rtol=0):
            raise ValueError("Elevation bin spacing is not uniform.")
        if not np.isclose(az_bin_delta[0], el_bin_delta[0], atol=1e-10, rtol=0):
            raise ValueError(
                "Azimuth and elevation bin spacing do not match: "
                f"az {az_bin_delta[0]} != el {el_bin_delta[0]}."
            )
        self.spacing_deg = az_bin_delta[0]

        # Build the azimuth and elevation grids with an AzElSkyGrid object
        # and check that the 1D axes match the dataset's az and el.
        self.sky_grid = spatial_utils.AzElSkyGrid(
            spacing_deg=self.spacing_deg,
        )

        for dim, constructed_bins in zip(
            ["azimuth", "elevation"],
            [self.sky_grid.az_bin_midpoints, self.sky_grid.el_bin_midpoints],
        ):
            if not np.allclose(
                sorted(constructed_bins),
                self.data[f"{dim}_bin_center"],
                atol=1e-10,
                rtol=0,
            ):
                raise ValueError(
                    f"{dim} bin centers do not match."
                    f"Constructed: {constructed_bins}"
                    f"Dataset: {self.data[f'{dim}_bin_center']}"
                )

        # Unwrap the az, el grids to series of points tiling the sky and combine them
        # into shape (number of points in tiling of the sky, 2) where
        # column 0 (az_el_points[:, 0]) is the azimuth of that point and
        # column 1 (az_el_points[:, 1]) is the elevation of that point.
        self.az_el_points = np.column_stack(
            (
                self.sky_grid.az_grid.ravel(),
                self.sky_grid.el_grid.ravel(),
            )
        )
        self.num_points = self.az_el_points.shape[0]

        # Also store the bin edges for the pointing set to allow for "pull" method
        # of index matching (not yet implemented).
        # These are 1D arrays of different lengths and cannot be stacked.
        self.az_bin_edges = self.sky_grid.az_bin_edges
        self.el_bin_edges = self.sky_grid.el_bin_edges

    def __repr__(self) -> str:
        """
        Return a string representation of the UltraPointingSet.

        Returns
        -------
        str
            String representation of the UltraPointingSet.
        """
        return (
            f"UltraPointingSet\n\t(spice_reference_frame="
            f"{self.spice_reference_frame}, epoch={self.epoch}, "
            f"num_points={self.num_points})"
        )


# Define the Map classes
class AbstractSkyMap(ABC):
    """
    Abstract base class to contain map data in the context of ENA sky maps.

    Data values are stored in a dictionary, where the final (-1) axis
    is the only spatial dimension. If the map is rectangular,
    this axis is the raveled 2D grid.
    If the map is Healpix, this axis is the 1D array of Healpix pixel indices.
    """

    @abstractmethod
    def __init__(self) -> None:
        self.tiling_type: SkyTilingType
        self.sky_grid: spatial_utils.AzElSkyGrid
        self.num_points: int
        self.binning_grid_shape: tuple[int, ...]
        self.data_dict: dict[str, NDArray]

    def project_pset_values_to_map(
        self,
        pointing_set: PointingSet,
        value_keys: list[str] | None = None,
        index_match_method: IndexMatchMethod = IndexMatchMethod.PUSH,
    ) -> None:
        """
        Project a pointing set's values to the map grid.

        Here, the term "project" refers to the process of determining which pixels in
        the map grid correspond to which pixels in the pointing set grid, and then
        binning the values at those indices from the pointing set to the map.

        Parameters
        ----------
        pointing_set : PointingSet
            The pointing set containing the values to project to the map.
        value_keys : list[tuple[str, IndexMatchMethod]] | None
            The keys of the values in the PointingSet to project to the map.
            Ex.: ["counts", "flux"]
            data_vars named each key must be present, and of the same dimensionality in
            each pointing set which is to be projected to the map.
            Default is None, in which case all data_vars in the pointing set are used.
        index_match_method : IndexMatchMethod, optional
            The method of index matching to use for all values.
            Default is IndexMatchMethod.PUSH.

        Raises
        ------
        ValueError
            If a value key is not found in the pointing set.
        """
        if value_keys is None:
            value_keys = list(pointing_set.data.data_vars.keys())
        for value_key in value_keys:
            if value_key not in pointing_set.data.data_vars:
                raise ValueError(f"Value key {value_key} not found in pointing set.")

        if index_match_method is IndexMatchMethod.PUSH:
            # Determine the indices of the sky map grid that correspond to
            # each pixel in the pointing set.
            matched_indices_push = match_coords_to_indices(
                input_object=pointing_set,
                output_object=self,
            )
        elif index_match_method is IndexMatchMethod.PULL:
            # Determine the indices of the pointing set grid that correspond to
            # each pixel in the sky map.
            matched_indices_pull = match_coords_to_indices(
                input_object=self,
                output_object=pointing_set,
            )
        else:
            raise NotImplementedError(
                "Only PUSH and PULL index matching methods are supported."
            )

        for value_key in value_keys:
            pset_values = pointing_set.data[value_key]

            # If there is an epoch dim with size 1, flatten it out
            if "epoch" in pset_values.dims and pset_values["epoch"].size == 1:
                pset_values = pset_values.squeeze("epoch")

            # If multiple spatial axes present
            # (i.e (az, el) for rectangular coordinate PSET),
            # flatten them in the values array to match the raveled indices
            raveled_pset_data = pset_values.data.reshape(-1, pointing_set.num_points)

            if value_key not in self.data_dict:
                # Initialize the map data array if it doesn't exist (values start at 0)
                output_shape = (*raveled_pset_data.shape[:-1], self.num_points)
                self.data_dict[value_key] = np.zeros(output_shape)

            if index_match_method is IndexMatchMethod.PUSH:
                # Bin the values at the matched indices. There may be multiple
                # pointing set pixels that correspond to the same sky map pixel.
                pointing_projected_values = map_utils.bin_single_array_at_indices(
                    value_array=raveled_pset_data,
                    projection_grid_shape=self.binning_grid_shape,
                    projection_indices=matched_indices_push,
                )
            elif index_match_method is IndexMatchMethod.PULL:
                # We know that there will only be one value per sky map pixel,
                # so we can use the matched indices directly
                pointing_projected_values = raveled_pset_data[..., matched_indices_pull]
            else:
                raise NotImplementedError(
                    "Only PUSH and PULL index matching methods are supported."
                )

            self.data_dict[value_key] += pointing_projected_values


class RectangularSkyMap(AbstractSkyMap):
    """
    Map which tiles the sky with a 2D rectangular grid of azimuth/elevation pixels.

    Parameters
    ----------
    spacing_deg : float
        The spacing of the rectangular grid in degrees.
    spice_frame : geometry.SpiceFrame
        The reference Spice frame of the map.

    Notes
    -----
    Internally, the map is stored as a 1D array of pixels, and all data arrays
    are stored with the final (-1) axis as the only spatial axis, representing the
    pixel index in the 1D array (See Figs 1-2, which demonstrate the 1D pixel index
    corresponding to the 2D grid of coordinates).

    ^  |15,  75|45,  75|75,  75|105,  75|...|255,  75|285,  75|315,  75|345,  75|
    |  |15,  45|45,  45|75,  45|105,  45|...|255,  45|285,  45|315,  45|345,  45|
    |  |15,  15|45,  15|75,  15|105,  15|...|255,  15|285,  15|315,  15|345,  15|
    |  |15, -15|45, -15|75, -15|105, -15|...|255, -15|285, -15|315, -15|345, -15|
    |  |15, -45|45, -45|75, -45|105, -45|...|255, -45|285, -45|315, -45|345, -45|
    |  |15, -75|45, -75|75, -75|105, -75|...|255, -75|285, -75|315, -75|345, -75|
    |
    ---------------------------------------------------------------> Azimuth (degrees)
    Elevation (degrees)

    Fig. 1: Example of a rectangular grid of pixels in azimuth and elevation coordinates
    in degrees, with a spacing of 30 degrees. There will be 12 azimuth bins and 6
    elevation bins in this example, resulting in 72 pixels in the map.

    A multidimentional value (e.g. counts, with energy levels at each pixel)
    will be stored as a 2D array with the first axis as the energy dimension and the
    second axis as the pixel index.

    ^  |5|11|17|23|29|35|41|47|53|59|65|71|
    |  |4|10|16|22|28|34|40|46|52|58|64|70|
    |  |3|9 |15|21|27|33|39|45|51|57|63|69|
    |  |2|8 |14|20|26|32|38|44|50|56|62|68|
    |  |1|7 |13|19|25|31|37|43|49|55|61|67|
    |  |0|6 |12|18|24|30|36|42|48|54|60|66|
    ---------------------------------------> Azimuth
    Elevation

    Fig. 2: The 1D indices of the pixels in Fig. 1.
    Note that the indices are raveled from the 2D grid of (az, el) such that as one
    increases in pixel index, elevation increments first, then azimuth.
    """

    def __init__(
        self,
        spacing_deg: float,
        spice_frame: geometry.SpiceFrame,
    ):
        # Define the core properties of the map:
        self.tiling_type = SkyTilingType.RECTANGULAR  # Type of tiling of the sky

        # The reference Spice frame of the map, in which angles are defined
        self.spice_reference_frame = spice_frame

        # Angular spacing of the map grid (degrees) defines the number, size of pixels.
        self.spacing_deg = spacing_deg
        self.sky_grid = spatial_utils.AzElSkyGrid(
            spacing_deg=self.spacing_deg,
        )
        # The shape of the map (num_az_bins, num_el_bins) is used to bin the data
        self.binning_grid_shape = self.sky_grid.grid_shape

        # Unwrap the az, el grids to 1D array of points tiling the sky
        az_points = self.sky_grid.az_grid.ravel()
        el_points = self.sky_grid.el_grid.ravel()

        # Stack so axis 0 is different pixels, and axis 1 is (az, el) of the pixel
        self.az_el_points = np.column_stack((az_points, el_points))
        self.num_points = self.az_el_points.shape[0]

        # Calculate solid angles of each pixel in the map grid in units of steradians
        self.solid_angle_grid = spatial_utils.build_solid_angle_map(
            spacing_deg=self.spacing_deg,
        )
        self.solid_angle_points = self.solid_angle_grid.ravel()

        # Initialize empty data dictionary to store map data
        self.data_dict: dict[str, NDArray] = {}

    def __repr__(self) -> str:
        """
        Return a string representation of the RectangularSkyMap.

        Returns
        -------
        str
            String representation of the RectangularSkyMap.
        """
        return (
            f"{self.__class__.__name__}\n\t(reference_frame="
            f"{self.spice_reference_frame.name} ({self.spice_reference_frame.value}), "
            f"spacing_deg={self.spacing_deg}, num_points={self.num_points})"
        )


class HealpixSkyMap(AbstractSkyMap):
    """
    Map which tiles the sky with a Healpix tessellation of equal-area pixels.

    Parameters
    ----------
    nside : int
        The nside parameter of the Healpix tessellation.
    spice_frame : geometry.SpiceFrame
        The reference Spice frame of the map.
    nested : bool, optional
        Whether the Healpix tessellation is nested. Default is False.
    """

    def __init__(
        self, nside: int, spice_frame: geometry.SpiceFrame, nested: bool = False
    ):
        # Define the core properties of the map:
        self.tiling_type = SkyTilingType.HEALPIX
        self.spice_reference_frame = spice_frame

        # Tile the sky with a Healpix tessellation. Defined by nside, nested parameters.
        self.nside = nside
        self.nested = nested

        # Calculate how many pixels cover the sky and the approximate resolution (deg)
        self.num_points = hp.nside2npix(nside)
        self.approx_resolution = np.rad2deg(hp.nside2resol(nside, arcmin=False))
        # Define binning_grid_shape for consistency with RectangularSkyMap
        self.binning_grid_shape = (self.num_points,)

        # The centers of each pixel in the Healpix tessellation in azimuth (az) and
        # elevation (el) coordinates (degrees) within the map's Spice frame.
        pixel_az, pixel_el = hp.pix2ang(
            nside=nside, ipix=np.arange(self.num_points), nest=nested, lonlat=True
        )
        # Stack so axis 0 is different pixels, and axis 1 is (az, el) of the pixel
        self.az_el_points = np.column_stack((pixel_az, pixel_el))

        # Tracks Per-Pixel Solid Angle in steradians.
        self.solid_angle = hp.nside2pixarea(nside, degrees=False)

        # Solid angle is equal at all pixels, but define
        # solid_angle_points to be consistent with RectangularSkyMap
        self.solid_angle_points = np.full(self.num_points, self.solid_angle)

        # Initialize the data dictionary to store map data projected from pointing sets
        self.data_dict: dict[str, NDArray] = {}

    def __repr__(self) -> str:
        """
        Return a string representation of the HealpixSkyMap.

        Returns
        -------
        str
            String representation of the HealpixSkyMap.
        """
        return (
            f"{self.__class__.__name__}\n\t(reference_frame="
            f"{self.spice_reference_frame.name} ({self.spice_reference_frame.value}), "
            f"nside={self.nside}, num_points={self.num_points})"
        )
