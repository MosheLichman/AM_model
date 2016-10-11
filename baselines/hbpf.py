"""
Author: Moshe Lichman
"""
from __future__ import division
import numpy as np

import argparse

from baselines import mf_commons
from commons import log_utils as log
from commons import file_utils as fu
from commons import time_measure as tm


def _fix_projection(mat, items, dim):
    """
    Both matrices needs to be fixed first. In the c++ code they ignore locations with
    no data and they change the entire projection.

     INPUT:
    -------
        1. mat:     <( <= items, dim + 2) ndarray>   htheta or hbeta. There could be less rows than items because
                                                     in the cpp code if a location didn't have data in training they
                                                     remove it. I don't.
                                                     Each row is [their_id, my_id, [factor_values]]
        2. items:    <int>                           number of individual or location.
        3. dims:     <int>                           number of hidden latent space.

     OUTPUT:
    --------
        1. fixed:   <(items, dim) ndarray>           fixed projection matrix
    """
    fixed = np.zeros([items, dim])
    my_ids = mat[:, 1].astype(int) - 1
    values = mat[:, 2:]
    for i in range(mat.shape[0]):
        fixed[my_ids[i]] = values[i]
    return fixed


def main_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('-test', type=str, help='Path to validation matrix.')

    parser.add_argument('-num_proc', type=int, help='Num processors', default=2)
    parser.add_argument('-obj', type=str, help='Objective function.', default='erank')
    parser.add_argument('-theta', type=str, help='Path to htheta file.')
    parser.add_argument('-beta', type=str, help='Path to hbeta file.')
    parser.add_argument('-r', type=int, help='Number of rows.')
    parser.add_argument('-c', type=int, help='Number of columns.')

    parser.add_argument('-v', help='Debug logging level', action='store_true')

    args = parser.parse_args()

    if args.v:
        log.set_verbose()

    tm.reset_tm()

    # Converting the validation data to [row_id, col_id] X number of obs. It's not as efficient but it's
    # much easier to work with that data.
    test_mat_raw = fu.pkl_load(args.test)
    test_data = np.repeat(test_mat_raw[:, :-1], test_mat_raw[:, -1].astype(int), axis=0)

    theta = fu.np_load_txt(args.theta, delimiter='\t')
    # This assumes that all users have data
    W = _fix_projection(theta, args.r, theta.shape[1] - 2)

    beta = fu.np_load_txt(args.beta, delimiter='\t')
    H = _fix_projection(beta, args.c, beta.shape[1] - 2).T

    results = mf_commons.evaluation(args.obj, test_data, W, H, args.num_proc)

    tm.print_summary()
    return results


if __name__ == '__main__':
    res = main_func()
