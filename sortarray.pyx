cimport cython
cimport numpy as np
import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def sort(array, sorting_index):
    """
    :param array: some ndarray
    :param sorting_index: some other index array by which to sort the array
    :return: sorted array
    """
    return np.take_along_axis(array, sorting_index, axis=0)
