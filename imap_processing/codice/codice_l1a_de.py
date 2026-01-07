"""Processing functions for CoDICE L1A Direct Event data."""

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.codice import constants
from imap_processing.codice.decompress import decompress
from imap_processing.codice.utils import (
    CODICEAPID,
    CoDICECompression,
    SegmentedPacketOrder,
    ViewTabInfo,
    apply_replacements_to_attrs,
    get_codice_epoch_time,
)
from imap_processing.spice.time import met_to_ttj2000ns


def get_de_metadata(packets: xr.Dataset, packet_index: int) -> bytes:
    """
    Gather and return packet metadata (From packet_version through byte_count).

    Extract the metadata in the packet_indexed direct event packet, which is then
    used to construct the full data of the group of packet_indexs.

    Parameters
    ----------
    packets : xarray.Dataset
        The packet_indexed direct event packet data.
    packet_index : int
        The index of the packet_index of interest.

    Returns
    -------
    metadata : bytes
        The compressed metadata for the packet_indexed packet.
    """
    # String together the metadata fields and convert the data to a bytes obj
    metadata_str = ""
    for field, num_bits in constants.DE_METADATA_FIELDS.items():
        metadata_str += f"{packets[field].data[packet_index]:0{num_bits}b}"
    metadata_chunks = [metadata_str[i : i + 8] for i in range(0, len(metadata_str), 8)]
    metadata_ints = [int(item, 2) for item in metadata_chunks]
    metadata = bytes(metadata_ints)

    return metadata


def group_data(packets: xr.Dataset) -> list[bytes]:
    """
    Organize continuation packets into appropriate groups.

    Some packets are continuation packets, as in, they are packets that are
    part of a group of packets. These packets are marked by the `seq_flgs` field
    in the CCSDS header of the packet. For CoDICE, the values are defined as
    follows:

    3 = Packet is not part of a group
    1 = Packet is the first packet of the group
    0 = Packet is in the middle of the group
    2 = Packet is the last packet of the group

    For packets that are part of a group, the byte count associated with the
    first packet of the group signifies the byte count for the entire group.

    Parameters
    ----------
    packets : xarray.Dataset
        Dataset containing the packets to group.

    Returns
    -------
    grouped_data : list[bytes]
        The packet data, converted to bytes and grouped appropriately.
    """
    grouped_data = []  # Holds the properly grouped data to be decompressed
    current_group = bytearray()  # Temporary storage for current group
    group_byte_count = None  # Temporary storage for current group byte count

    for packet_index in range(len(packets.event_data.data)):
        packet_data = packets.event_data.data[packet_index]
        group_code = packets.seq_flgs.data[packet_index]
        byte_count = packets.byte_count.data[packet_index]

        # If the group code is 3, this means the data is unsegmented
        # and can be decompressed as-is
        if group_code == SegmentedPacketOrder.UNSEGMENTED:
            grouped_data.append(packet_data[:byte_count])

        # If the group code is 1, this means the data is the first data in a
        # group. Also, set the byte count for the group
        elif group_code == SegmentedPacketOrder.FIRST_SEGMENT:
            group_byte_count = byte_count
            current_group += packet_data

        # If the group code is 0, this means the data is part of the middle of
        # the group.
        elif group_code == SegmentedPacketOrder.CONTINUATION_SEGMENT:
            current_group += get_de_metadata(packets, packet_index)
            current_group += packet_data

        # If the group code is 2, this means the data is the last data in the
        # group
        elif group_code == SegmentedPacketOrder.LAST_SEGMENT:
            current_group += get_de_metadata(packets, packet_index)
            current_group += packet_data

            # The grouped data is now ready to be decompressed
            values_to_decompress = current_group[:group_byte_count]
            grouped_data.append(values_to_decompress)

            # Reset the current group
            current_group = bytearray()
            group_byte_count = None

    return grouped_data


def combine_segmented_packets(packets: xr.Dataset) -> xr.Dataset:
    """
    Combine segmented packets into unsegmented packets.

    All packets have (SHCOARSE, EVENT_DATA, CHKSUM) fields. To combine
    the segmented packets, we only concatenate along the EVENT_DATA field
    into the first packet of the group.

    Parameters
    ----------
    packets : xarray.Dataset
        Dataset containing the packets to combine.

    Returns
    -------
    combined_packets : xarray.Dataset
        Dataset containing the combined packets.
    """
    # Identification of group starts (UNSEGMENTED or FIRST_SEGMENT)
    is_group_start = (packets.seq_flgs.data == SegmentedPacketOrder.UNSEGMENTED) | (
        packets.seq_flgs.data == SegmentedPacketOrder.FIRST_SEGMENT
    )

    # Assign group IDs using cumulative sum - each group start increments the ID
    group_ids = np.cumsum(is_group_start)

    # Get indices of packets we'll keep (first packet of each group)
    group_start_indices = np.where(is_group_start)[0]

    # Concatenate event_data in-place for each group
    for group_id in np.unique(group_ids):
        # Find all packets belonging to this group
        group_mask = group_ids == group_id
        group_indices = np.where(group_mask)[0]

        # If multiple packets, concatenate into the first packet
        if len(group_indices) > 1:
            start_index = group_indices[0]
            packets["event_data"].data[start_index] = np.sum(
                packets.event_data.data[group_indices]
            )

    # Select only the first packet of each group
    combined_packets = packets.isel(epoch=group_start_indices)

    return combined_packets


def extract_initial_items_from_combined_packets(
    packets: xr.Dataset,
) -> xr.Dataset:
    """
    Extract metadata fields from the beginning of combined event_data packets.

    Extracts bit fields from the first 20 bytes of each event_data array
    and adds them as new variables to the dataset.

    Parameters
    ----------
    packets : xr.Dataset
        Dataset containing combined packets with event_data.

    Returns
    -------
    xr.Dataset
        Dataset with extracted metadata fields added.
    """
    # Initialize arrays for extracted fields
    n_packets = len(packets.epoch)

    # Preallocate arrays
    packet_version = np.zeros(n_packets, dtype=np.uint16)
    spin_period = np.zeros(n_packets, dtype=np.uint16)
    acq_start_seconds = np.zeros(n_packets, dtype=np.uint32)
    acq_start_subseconds = np.zeros(n_packets, dtype=np.uint32)
    spare_1 = np.zeros(n_packets, dtype=np.uint8)
    st_bias_gain_mode = np.zeros(n_packets, dtype=np.uint8)
    sw_bias_gain_mode = np.zeros(n_packets, dtype=np.uint8)
    priority = np.zeros(n_packets, dtype=np.uint8)
    suspect = np.zeros(n_packets, dtype=np.uint8)
    compressed = np.zeros(n_packets, dtype=np.uint8)
    num_events = np.zeros(n_packets, dtype=np.uint32)
    byte_count = np.zeros(n_packets, dtype=np.uint32)

    # Extract fields from each packet
    for pkt_idx in range(n_packets):
        event_data = packets.event_data.data[pkt_idx]

        # Byte-aligned fields using frombuffer
        packet_version[pkt_idx] = np.frombuffer(event_data[0:2], dtype=">u2")[0]
        spin_period[pkt_idx] = np.frombuffer(event_data[2:4], dtype=">u2")[0]
        acq_start_seconds[pkt_idx] = np.frombuffer(event_data[4:8], dtype=">u4")[0]

        # Non-byte-aligned fields (bytes 8-12 contain mixed bit fields)
        # Extract 4 bytes and unpack bit fields
        mixed_bytes = np.frombuffer(event_data[8:12], dtype=">u4")[0]

        # acq_start_subseconds: 20 bits (MSB)
        acq_start_subseconds[pkt_idx] = (mixed_bytes >> 12) & 0xFFFFF
        # spare_1: 2 bits
        spare_1[pkt_idx] = (mixed_bytes >> 10) & 0x3
        # st_bias_gain_mode: 2 bits
        st_bias_gain_mode[pkt_idx] = (mixed_bytes >> 8) & 0x3
        # sw_bias_gain_mode: 2 bits
        sw_bias_gain_mode[pkt_idx] = (mixed_bytes >> 6) & 0x3
        # priority: 4 bits
        priority[pkt_idx] = (mixed_bytes >> 2) & 0xF
        # suspect: 1 bit
        suspect[pkt_idx] = (mixed_bytes >> 1) & 0x1
        # compressed: 1 bit (LSB)
        compressed[pkt_idx] = mixed_bytes & 0x1

        # Remaining byte-aligned fields
        num_events[pkt_idx] = np.frombuffer(event_data[12:16], dtype=">u4")[0]
        byte_count[pkt_idx] = np.frombuffer(event_data[16:20], dtype=">u4")[0]

        # Remove the first 20 bytes from event_data
        packets.event_data.data[pkt_idx] = event_data[20:]
        # Trim to the byte_count field
        packets.event_data.data[pkt_idx] = packets.event_data.data[pkt_idx][
            : byte_count[pkt_idx]
        ]
        if compressed[pkt_idx]:
            packets.event_data.data[pkt_idx] = decompress(
                packets.event_data.data[pkt_idx],
                CoDICECompression.LOSSLESS,
            )

    # Add extracted fields to dataset
    packets["packet_version"] = xr.DataArray(packet_version, dims=["epoch"])
    packets["spin_period"] = xr.DataArray(spin_period, dims=["epoch"])
    packets["acq_start_seconds"] = xr.DataArray(acq_start_seconds, dims=["epoch"])
    packets["acq_start_subseconds"] = xr.DataArray(acq_start_subseconds, dims=["epoch"])
    packets["spare_1"] = xr.DataArray(spare_1, dims=["epoch"])
    packets["st_bias_gain_mode"] = xr.DataArray(st_bias_gain_mode, dims=["epoch"])
    packets["sw_bias_gain_mode"] = xr.DataArray(sw_bias_gain_mode, dims=["epoch"])
    packets["priority"] = xr.DataArray(priority, dims=["epoch"])
    packets["suspect"] = xr.DataArray(suspect, dims=["epoch"])
    packets["compressed"] = xr.DataArray(compressed, dims=["epoch"])
    packets["num_events"] = xr.DataArray(num_events, dims=["epoch"])
    packets["byte_count"] = xr.DataArray(byte_count, dims=["epoch"])

    return packets


def unpack_bits(bit_structure: dict, de_data: np.ndarray) -> dict:
    """
    Unpack 64-bit values into separate fields based on bit structure.

    Parameters
    ----------
    bit_structure : dict
        Dictionary mapping variable names to their bit lengths.
    de_data : np.ndarray
        1D array of 64-bit values to unpack.

    Returns
    -------
    dict
        Dictionary of field_name -> unpacked values array.
    """
    unpacked = {}
    # Data need to be unpacked in right to left order (LSB). Eg.
    #   binary string  - 0x03 → 00000011
    #   bit read order - Bit 7 → 0
    #                    Bit 6 → 0
    #                    Bit 5 → 0
    #                    Bit 4 → 0
    #                    Bit 3 → 0
    #                    Bit 2 → 0
    #                    Bit 1 → 1
    #                    Bit 0 (LSB) → 1
    #   bits chunks - [5, 1, ...., 7, 3, 16]
    #   vars - ['gain', 'apd_id', ...., 'energy_step', 'priority', 'spare']
    #   unpack data - [3, 0, 0, ....., 0, 0]

    # convert data into int type for bitwise operations
    de_data = de_data.astype(np.uint64)

    for name, data in bit_structure.items():
        mask = (1 << data["bit_length"]) - 1
        unpacked[name] = de_data & mask
        # Shift the data to the right for the next iteration
        de_data = de_data >> data["bit_length"]

    return unpacked


def process_de_data(
    packets: xr.Dataset,
    decompressed_data: list[list[int]],
    apid: int,
    cdf_attrs: ImapCdfAttributes,
) -> xr.Dataset:
    """
    Reshape the decompressed direct event data into CDF-ready arrays.

    Unpacking DE needs below for-loops because of many reasons, including:
        - Need of preserve fillval per field of various bit lengths
        - inability to use nan for 64-bits unpacking
        - num_events being variable length per epoch
        - binning priorities into its bins
        - unpacking 64-bits into fields and indexing correctly

    Parameters
    ----------
    packets : xarray.Dataset
        Dataset containing the packets, needed to determine priority order
        and data quality.
    decompressed_data : list[list[int]]
        The decompressed data to reshape, in the format <epoch>[<priority>[<event>]].
    apid : int
        The sensor type, used primarily to determine if the data are from
        CoDICE-Lo or CoDICE-Hi.
    cdf_attrs : ImapCdfAttributes
        The CDF attributes to be added to the dataset.

    Returns
    -------
    data : xarray.Dataset
        Processed Direct Event data.
    """
    # xr.Dataset to hold all the (soon to be restructured) direct event data
    de_data = xr.Dataset()

    # Extract some useful variables
    num_priorities = constants.DE_DATA_PRODUCT_CONFIGURATIONS[apid]["num_priorities"]
    bit_structure = constants.DE_DATA_PRODUCT_CONFIGURATIONS[apid]["bit_structure"]

    # Determine the number of epochs to help with data array initialization
    # There is one epoch per set of priorities
    num_epochs = len(decompressed_data) // num_priorities

    # Initialize data arrays for unpacked 64-bits fields
    for field in bit_structure:
        if field not in ["Priority", "Spare"]:
            # Update attrs based on fillval per field
            fillval = bit_structure[field]["fillval"]
            dtype = bit_structure[field]["dtype"]
            attrs = cdf_attrs.get_variable_attributes("de_3d_attrs")
            attrs = apply_replacements_to_attrs(
                attrs, {"num_digits": len(str(fillval)), "valid_max": fillval}
            )
            de_data[field] = xr.DataArray(
                np.full(
                    (num_epochs, num_priorities, 10000),
                    fillval,
                    dtype=dtype,
                ),
                name=field,
                dims=["epoch", "priority", "event_num"],
                attrs=attrs,
            )

    # Get num_events, data quality, and priorities data for beginning of packet_indexs
    packet_index_starts = np.where(
        (packets.seq_flgs.data == SegmentedPacketOrder.UNSEGMENTED)
        | (packets.seq_flgs.data == SegmentedPacketOrder.FIRST_SEGMENT)
    )[0]
    num_events_arr = packets.num_events.data[packet_index_starts]
    data_quality_arr = packets.suspect.data[packet_index_starts]
    priorities_arr = packets.priority.data[packet_index_starts]

    # Initialize other fields of l1a that we want to
    # carry in L1A CDF file
    de_data["num_events"] = xr.DataArray(
        np.full((num_epochs, num_priorities), 65535, dtype=np.uint16),
        name="num_events",
        dims=["epoch", "priority"],
        attrs=cdf_attrs.get_variable_attributes("de_2d_attrs"),
    )

    de_data["data_quality"] = xr.DataArray(
        np.full((num_epochs, num_priorities), 65535, dtype=np.uint16),
        name="data_quality",
        dims=["epoch", "priority"],
        attrs=cdf_attrs.get_variable_attributes("de_2d_attrs"),
    )

    # As mentioned above, epoch data is of this shape:
    #   (epoch, (num_events * <number of priorities>)).
    # num_events is a variable number per priority.
    for epoch_index in range(num_epochs):
        # current epoch's grouped data are:
        #   current group's start index * 8 to next group's start indices * 8
        epoch_start = packet_index_starts[epoch_index] * num_priorities
        epoch_end = packet_index_starts[epoch_index + 1] * num_priorities
        # Extract the decompressed data for current epoch.
        # epoch_data should be of shape ((num_priorities * num_events),)
        epoch_data = decompressed_data[epoch_start:epoch_end]

        # Extract these other data
        unordered_priority = priorities_arr[epoch_start:epoch_end]
        unordered_data_quality = data_quality_arr[epoch_start:epoch_end]
        unordered_num_events = num_events_arr[epoch_start:epoch_end]

        print(f"Unorder priority {unordered_priority}")
        print(f"Unorder num_events {unordered_num_events}")
        print(f"Unorder data_quality {unordered_data_quality}")
        # If priority array unique size is not same size as
        # num_priorities, then throw error. They should match.
        if len(np.unique(unordered_priority)) != num_priorities:
            raise ValueError(
                f"Priority array for epoch {epoch_index} contains "
                f"non-unique values: {unordered_priority}"
            )

        # Until here, we have the out of order priority data. Data could have been
        # collected in any priority order. Eg.
        #   priority - [0, 4, 5, 1, 3, 2, 6, 7]
        # Now, we need to put data into their respective priority indexes
        # in final arrays for the current epoch. Eg. put data into
        #   priority - [0, 1, 2, 3, 4, 5, 6, 7]
        de_data["num_events"][epoch_index, unordered_priority] = unordered_num_events
        de_data["data_quality"][epoch_index, unordered_priority] = (
            unordered_data_quality
        )

        # Fill the event data into it's bin in same logic as above. But
        # since the epoch has different num_events per priority,
        # we need to loop and index accordingly. Otherwise, numpy throws
        # 'The detected shape was (n,) + inhomogeneous part' error.
        for priority_index in range(len(unordered_priority)):
            # Get num_events
            priority_num_events = int(unordered_num_events[priority_index])

            # If epoch data at priority index is empty, then what to do?
            # Can I just check if the num_events is zero to skip processing?
            # Where is there case where there is num_events > 0 but no data?
            if len(epoch_data[priority_index]) == 0 or priority_num_events == 0:
                priority_num = int(unordered_priority[priority_index])
                for field_name, field_data in bit_structure.items():
                    if field_name not in ["Priority", "Spare"]:
                        de_data[field_name][
                            epoch_index, priority_num, :priority_num_events
                        ] = field_data["fillval"]
                continue

            print(f"current {priority_index}, num_events {priority_num_events}")
            print(
                f"epoch data being unpacked {np.array(epoch_data[priority_index]).shape}"
            )
            # Reshape epoch data into (num_events, 8). That 8 is 8-bytes that
            # make up 64-bits. Therefore, combine last 8 dimension into one to
            # get 64-bits event data that we need to unpack later. First,
            # combine last 8 dimension into one 64-bits value
            #   we need to make a copy and reverse the byte order
            #   to match LSB order before we use .view.
            events_in_bytes = (
                np.array(epoch_data[priority_index], dtype=np.uint8)
                .reshape(priority_num_events, 8)[:, ::-1]
                .copy()
            )
            combined_64bits = events_in_bytes.view(np.uint64)[:, 0]
            # Unpack 64-bits into fields
            unpacked_fields = unpack_bits(bit_structure, combined_64bits)
            # Put unpacked event data into their respective variable and priority
            # number bins
            priority_num = int(unordered_priority[priority_index])
            for field_name, field_data in unpacked_fields.items():
                if field_name not in ["Priority", "Spare"]:
                    de_data[field_name][
                        epoch_index, priority_num, :priority_num_events
                    ] = field_data

    return de_data


def l1a_direct_event(unpacked_dataset: xr.Dataset, apid: int) -> xr.Dataset:
    """
    Process CoDICE L1A Direct Event data.

    Parameters
    ----------
    unpacked_dataset : xarray.Dataset
        Input L1A Direct Event dataset.
    apid : int
        APID to process.

    Returns
    -------
    xarray.Dataset
        Processed L1A Direct Event dataset.
    """
    print(unpacked_dataset)
    new_dataset = combine_segmented_packets(unpacked_dataset)
    new_dataset = extract_initial_items_from_combined_packets(new_dataset)
    print(new_dataset)
    # Group segmented data.
    # TODO: this may get replaced with space_packet_parser's functionality
    # grouped_data = group_data(new_dataset)

    # # Decompress data shape is (epoch, priority * num_events)
    # decompressed_data = [
    #     decompress(
    #         group,
    #         CoDICECompression.LOSSLESS,
    #     )
    #     for group in new_dataset.event_data
    # ]

    # Gather the CDF attributes
    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_global_attrs("codice")
    cdf_attrs.add_instrument_variable_attrs("codice", "l1a")

    # Unpack DE packet data into CDF-ready variables
    de_dataset = process_de_data(new_dataset, decompressed_data, apid, cdf_attrs)

    # Determine the epochs to use in the dataset, which are the epochs whenever
    # there is a start of a segment and the priority is 0
    epoch_indices = np.where(
        (
            (new_dataset.seq_flgs.data == SegmentedPacketOrder.UNSEGMENTED)
            | (new_dataset.seq_flgs.data == SegmentedPacketOrder.FIRST_SEGMENT)
        )
        & (new_dataset.priority.data == 0)
    )[0]
    acq_start_seconds = new_dataset.acq_start_seconds[epoch_indices]
    acq_start_subseconds = new_dataset.acq_start_subseconds[epoch_indices]
    spin_periods = new_dataset.spin_period[epoch_indices]

    # Calculate epoch variables using sensor id and apid
    # Provide 0 as default input for other inputs but they
    # are not used in epoch calculation
    view_tab_info = ViewTabInfo(
        apid=apid,
        sensor=1 if apid == CODICEAPID.COD_HI_PHA else 0,
        collapse_table=0,
        three_d_collapsed=0,
        view_id=0,
    )
    epochs, epochs_delta = get_codice_epoch_time(
        acq_start_seconds, acq_start_subseconds, spin_periods, view_tab_info
    )

    # Define coordinates
    epoch = xr.DataArray(
        met_to_ttj2000ns(epochs),
        name="epoch",
        dims=["epoch"],
        attrs=cdf_attrs.get_variable_attributes("epoch", check_schema=False),
    )
    epoch_delta_minus = xr.DataArray(
        epochs_delta,
        name="epoch_delta_minus",
        dims=["epoch"],
        attrs=cdf_attrs.get_variable_attributes(
            "epoch_delta_minus", check_schema=False
        ),
    )
    epoch_delta_plus = xr.DataArray(
        epochs_delta,
        name="epoch_delta_plus",
        dims=["epoch"],
        attrs=cdf_attrs.get_variable_attributes("epoch_delta_plus", check_schema=False),
    )
    event_num = xr.DataArray(
        np.arange(constants.MAX_DE_EVENTS_PER_PACKET),
        name="event_num",
        dims=["event_num"],
        attrs=cdf_attrs.get_variable_attributes("event_num", check_schema=False),
    )
    event_num_label = xr.DataArray(
        np.arange(constants.MAX_DE_EVENTS_PER_PACKET).astype(str),
        name="event_num_label",
        dims=["event_num"],
        attrs=cdf_attrs.get_variable_attributes("event_num_label", check_schema=False),
    )
    priority = xr.DataArray(
        np.arange(constants.DE_DATA_PRODUCT_CONFIGURATIONS[apid]["num_priorities"]),
        name="priority",
        dims=["priority"],
        attrs=cdf_attrs.get_variable_attributes("priority", check_schema=False),
    )
    priority_label = xr.DataArray(
        np.arange(
            constants.DE_DATA_PRODUCT_CONFIGURATIONS[apid]["num_priorities"]
        ).astype(str),
        name="priority_label",
        dims=["priority"],
        attrs=cdf_attrs.get_variable_attributes("priority_label", check_schema=False),
    )

    # Logical source id to lookup global attributes
    if apid == CODICEAPID.COD_LO_PHA:
        attrs = cdf_attrs.get_global_attributes("imap_codice_l1a_lo-direct-events")
    elif apid == CODICEAPID.COD_HI_PHA:
        attrs = cdf_attrs.get_global_attributes("imap_codice_l1a_hi-direct-events")

    # Add coordinates and global attributes to dataset
    de_dataset = de_dataset.assign_coords(
        epoch=epoch,
        epoch_delta_minus=epoch_delta_minus,
        epoch_delta_plus=epoch_delta_plus,
        event_num=event_num,
        event_num_label=event_num_label,
        priority=priority,
        priority_label=priority_label,
    )
    de_dataset.attrs = attrs

    # Carry over these variables from unpacked dataset
    if apid == CODICEAPID.COD_LO_PHA:
        # Add k_factor
        de_dataset["k_factor"] = xr.DataArray(
            np.array([constants.K_FACTOR]),
            name="k_factor",
            dims=["k_factor"],
            attrs=cdf_attrs.get_variable_attributes("k_factor", check_schema=False),
        )

    de_dataset["sw_bias_gain_mode"] = xr.DataArray(
        new_dataset["sw_bias_gain_mode"].data[epoch_indices],
        name="sw_bias_gain_mode",
        dims=["epoch"],
        attrs=cdf_attrs.get_variable_attributes("sw_bias_gain_mode"),
    )

    de_dataset["st_bias_gain_mode"] = xr.DataArray(
        new_dataset["st_bias_gain_mode"].data[epoch_indices],
        name="st_bias_gain_mode",
        dims=["epoch"],
        attrs=cdf_attrs.get_variable_attributes("st_bias_gain_mode"),
    )

    return de_dataset
