import numpy as np
import pytest

from imap_processing.swe.l1a.swe_science import swe_science
from imap_processing.swe.l1b.swe_l1b_science import (
    get_indices_of_full_cycles,
)


@pytest.fixture(scope="session")
def l1a_test_data(decom_test_data):
    """Read test data from file and process to l1a"""
    processed_data = swe_science(decom_test_data, "001")
    return processed_data


def test_get_full_cycle_data_indices():
    q = np.array([0, 1, 2, 0, 1, 2, 3, 2, 3, 0, 2, 3, 0, 1, 2, 3, 2, 3, 1, 0])
    filtered_q = get_indices_of_full_cycles(q)
    np.testing.assert_array_equal(filtered_q, np.array([3, 4, 5, 6, 12, 13, 14, 15]))

    q = np.array([0, 1, 0, 1, 2, 3, 0, 2])
    filtered_q = get_indices_of_full_cycles(q)
    np.testing.assert_array_equal(filtered_q, np.array([2, 3, 4, 5]))

    q = np.array([0, 1, 2, 3])
    filtered_q = get_indices_of_full_cycles(q)
    np.testing.assert_array_equal(filtered_q, np.array([0, 1, 2, 3]))

    q = np.array([1, 2])
    filtered_q = get_indices_of_full_cycles(q)
    np.testing.assert_array_equal(filtered_q, np.array([]))
