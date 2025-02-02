"""Time measuring functionality.

This was written when having JETM usability in mind.

Author: Moshe Lichman
"""
from __future__ import division
from pandas import DataFrame
import numpy as np
import time

_time_funcs = {}


class _Point(object):
    """Time measurement point instance.

    A point is created for a specific functionality that time measurement is needed.
    """
    def __init__(self, name):
        self.name = name
        self.total_time = 0
        self.num_obs = 0
        self._start = time.time()

    def reset_time(self):
        self._start = time.time()

    def collect(self):
        self.total_time += time.time() - self._start
        self.num_obs += 1


def print_summary():
    """Prints time measurements using pandas DataFrame."""
    print '\n\n\n  Time measurements summary'
    print '------------------------------'

    x = []
    col_names = ['total time', '#calls', 'avg. time']
    row_names = []
    for point in _time_funcs.values():
        row_names.append(point.name)
        x.append([point.total_time, point.num_obs, point.total_time / point.num_obs])

    x = np.array(x)
    print(DataFrame(x, columns=col_names, index=row_names))


def get_point(point_name):
    """Creates or re-uses a time measurement point.

    Args:
        point_name: string.
                    Unique identification name of the point.

    Returns:
        point: Point.
               Point instance for the identification.
    """
    if point_name not in _time_funcs:
        _time_funcs[point_name] = _Point(point_name)
    else:
        _time_funcs[point_name].reset_time()

    return _time_funcs[point_name]


def reset_tm():
    """Resets all the measurements.

    This should be called in each script. Otherwise it will remember points from previous runs if the script is running
    in a python console.
    """
    global _time_funcs
    _time_funcs = {}


