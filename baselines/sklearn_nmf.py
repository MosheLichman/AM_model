"""
Author: Moshe Lichman
"""
from __future__ import division
import numpy as np
from scipy.sparse import coo_matrix
from scipy import sparse

import argparse

from baselines import mf_commons

from commons import log_utils as log
from commons import file_utils as fu
from commons import time_measure as tm

from sklearn.decomposition import NMF


def factorize_mat(train_data, k):
    # First loading the mf
    factor_point = tm.get_point('nmf factor')
    model = NMF(n_components=k, solver='cd', init='nndsvd', alpha=4)
    W = model.fit_transform(train_data)
    H = model.components_
    factor_point.collect()

    return W, H


def main_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', type=str, help='Path to training matrix.')
    parser.add_argument('-val', type=str, help='Path to training matrix.')
    parser.add_argument('-test', type=str, help='Path to validation matrix.')

    parser.add_argument('-num_proc', type=int, help='Num processors', default=2)
    parser.add_argument('-k', type=int, help='Latent space size', default=200)
    parser.add_argument('-r', type=int, help='Number of rows.')
    parser.add_argument('-c', type=int, help='Number of columns.')

    parser.add_argument('-v', help='Debug logging level', action='store_true')

    args = parser.parse_args()

    if args.v:
        log.set_verbose()

    tm.reset_tm()

    R, C = args.r, args.c
    obs_mat_raw = fu.pkl_load(args.train)
    # Combining the validation. I can treat it as train now.
    obs_mat_val_raw = fu.pkl_load(args.val)
    obs_mat_raw = np.vstack([obs_mat_raw, obs_mat_val_raw])

    train_data = sparse.coo_matrix((obs_mat_raw[:, 2], (obs_mat_raw[:, 0], obs_mat_raw[:, 1])),
                                   shape=(R, C)).tolil()

    # Converting the validation data to [row_id, col_id] X number of obs. It's not as efficient but it's
    # much easier to work with that data.
    test_mat_raw = fu.pkl_load(args.test)
    test_data = np.repeat(test_mat_raw[:, :-1], test_mat_raw[:, -1].astype(int), axis=0)

    W, H = factorize_mat(train_data, args.k)

    # Can't call it with logP
    results = mf_commons.evaluation('erank', test_data, W, H, args.num_proc)

    tm.print_summary()
    return np.array(results)


if __name__ == '__main__':
    res = main_func()
