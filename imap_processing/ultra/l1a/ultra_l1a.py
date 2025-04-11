"""Generate ULTRA L1a CDFs."""

import logging
from typing import Optional

import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.ultra.l0.decom_ultra import (
    process_ultra_events,
    process_ultra_rates,
    process_ultra_tof,
)
from imap_processing.ultra.l0.ultra_utils import (
    ULTRA_AUX,
    ULTRA_EVENTS,
    ULTRA_RATES,
    ULTRA_TOF,
)
from imap_processing.utils import packet_file_to_datasets

logger = logging.getLogger(__name__)


def ultra_l1a(packet_file: str, apid: Optional[int] = None) -> list[xr.Dataset]:
    """
    Will process ULTRA L0 data into L1A CDF files at output_filepath.

    Parameters
    ----------
    packet_file : str
        Path to the CCSDS data packet file.
    apid : Optional[int]
        Optional apid.

    Returns
    -------
    output_datasets : list[xarray.Dataset]
        List of xarray.Dataset.
    """
    xtce = str(
        f"{imap_module_directory}/ultra/packet_definitions/ULTRA_SCI_COMBINED.xml"
    )

    datasets_by_apid = packet_file_to_datasets(packet_file, xtce)

    output_datasets = []

    # This is used for two purposes currently:
    #    For testing purposes to only generate a dataset for a single apid.
    #    Each test dataset is only for a single apid while the rest of the apids
    #    contain zeros. Ideally we would have
    #    test data for all apids and remove this parameter.
    if apid is not None:
        apids = [apid]
    else:
        apids = list(datasets_by_apid.keys())

    # Update dataset global attributes
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs("ultra")
    attr_mgr.add_instrument_variable_attrs("ultra", "l1a")

    for apid in apids:  # noqa PLR1704 redefined apid variable from outer scope
        if apid in ULTRA_AUX.apid:
            decom_ultra_dataset = datasets_by_apid[apid]
            gattr_key = ULTRA_AUX.logical_source[ULTRA_AUX.apid.index(apid)]
        elif apid in ULTRA_TOF.apid:
            decom_ultra_dataset = process_ultra_tof(datasets_by_apid[apid])
            gattr_key = ULTRA_TOF.logical_source[ULTRA_TOF.apid.index(apid)]
        elif apid in ULTRA_RATES.apid:
            decom_ultra_dataset = process_ultra_rates(datasets_by_apid[apid])
            gattr_key = ULTRA_RATES.logical_source[ULTRA_RATES.apid.index(apid)]
        elif apid in ULTRA_EVENTS.apid:
            decom_ultra_dataset = process_ultra_events(datasets_by_apid[apid])
            gattr_key = ULTRA_EVENTS.logical_source[ULTRA_EVENTS.apid.index(apid)]
            # Add coordinate attributes
            attrs = attr_mgr.get_variable_attributes("event_id")
            decom_ultra_dataset.coords["event_id"].attrs.update(attrs)
        else:
            logger.error(f"APID {apid} not recognized.")
            # TODO: here we can put other apids
            continue

        decom_ultra_dataset.attrs.update(attr_mgr.get_global_attributes(gattr_key))

        # Add data variable attributes
        for key in decom_ultra_dataset.data_vars:
            attrs = attr_mgr.get_variable_attributes(key.lower())
            decom_ultra_dataset.data_vars[key].attrs.update(attrs)

        # Add coordinate attributes
        attrs = attr_mgr.get_variable_attributes("epoch")
        decom_ultra_dataset.coords["epoch"].attrs.update(attrs)

        output_datasets.append(decom_ultra_dataset)

    return output_datasets
