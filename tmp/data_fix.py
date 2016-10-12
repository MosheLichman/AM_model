"""
Pre-processing the data Dimitris sent me as prelim data.

1. Splitting the test into validation and text 50-50
2. Fixing the similarity matrices.
3. Creating data for HBPF

Author: Moshe Lichman
"""
from __future__ import division
import numpy as np
from scipy import sparse

import argparse

from commons import file_utils as fu
from commons import helpers
from commons import log_utils as log


def fix_sim_mat(sim_mat, dim):
    sim_mat = sim_mat[sim_mat[:, 0] != sim_mat[:, 1], :]
    a, b, val = sim_mat[0]
    if sim_mat[np.where((sim_mat[:, 0] == b) & (sim_mat[:, 1] == a))[0], 2] != val:
        log.info('Triangle matrix, fix it first or fix the script')
        assert False
    mat = sparse.coo_matrix((sim_mat[:, 2], (sim_mat[:, 0], sim_mat[:, 1])), shape=(dim, dim))
    norm = mat.sum(axis=1)
    mask = np.where(norm > 0)[0]
    tmp = mat.tolil()
    tmp[mask, :] /= norm[mask, :]
    return np.vstack(sparse.find(mat)).T


def main_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('-val', type=str, default=None)
    parser.add_argument('-test', type=str, help='Path to test file.')
    parser.add_argument('-train', type=str, help='Path to train file.')
    parser.add_argument('-output', type=str, help='Output folder.')
    parser.add_argument('-row_sim', type=str, help='Output folder.')
    parser.add_argument('-col_sim', type=str, help='Output folder.')

    args = parser.parse_args()

    if args.val is None:
        # Step 1: Splitting test into validation and test
        full_test_raw = fu.pkl_load(args.test)
        full_test = np.repeat(full_test_raw[:, :-1], full_test_raw[:, -1].astype(int), axis=0)
        val = np.zeros([0, 2])
        test = np.zeros([0, 2])

        for u in np.unique(full_test[:, 0]):
            u_data = full_test[np.where(full_test[:, 0] == u)[0], :]
            n = u_data.shape[0]

            # Shuffling the data
            perm = np.random.permutation(n)
            u_data = u_data[perm]

            # Half test half validation
            nv = np.ceil(n / 2).astype(int)
            val = np.vstack([val, u_data[:nv, :]])
            test = np.vstack([test, u_data[nv:, :]])

        # Transforming into coo
        val = helpers.unique_two_cols(val[:, 0], val[:, 1])
        test = helpers.unique_two_cols(test[:, 0], test[:, 1])
    else:
        # No reason to split, it's already splitted
        test = fu.pkl_load(args.test)
        val = fu.pkl_load(args.val)

    # Saving all the files in the new folder
    fu.pkl_dump(args.output, 'validation.pkl', val)
    fu.pkl_dump(args.output, 'test.pkl', val)

    train = fu.pkl_load(args.train)
    fu.pkl_dump(args.output, 'train.pkl', train)

    rows = np.max([np.max(train[:, 0]), np.max(val[:, 0]), np.max(test[:, 0])]) + 1
    cols = np.max([np.max(train[:, 1]), np.max(val[:, 1]), np.max(test[:, 1])]) + 1

    fu.np_save(args.output, 'r=%d_c=%d.npy' % (rows, cols), [])

    # Fixing the similarity matrices:
    #   1. Making the sure the diagonal is 0.
    #   2. Making sure it's not a triangle matrix
    #   3. Normalinzing to 1
    row_sim = fu.pkl_load(args.row_sim)
    row_sim = fix_sim_mat(row_sim, rows)

    fu.pkl_dump(args.output, 'row_sim.pkl', row_sim)

    col_sim = fu.pkl_load(args.col_sim)
    col_sim = fix_sim_mat(col_sim, cols)

    fu.pkl_dump(args.output, 'col_sim.pkl', col_sim)

    # Creating the HBPF files:
    #   1. Adding +1 to make it one based
    #   2. Making sure all the rows and column exist in the train
    val[:, :-1] += 1
    fu.np_save_txt(args.output, 'validation.tsv', val.astype(int), delimiter='\t', fmt='%d')

    test[:, :-1] += 1
    fu.np_save_txt(args.output, 'test.tsv', test.astype(int), delimiter='\t', fmt='%d')

    missing_rows = np.where(np.in1d(np.arange(rows), np.unique(train[:, 0]), assume_unique=True, invert=True))[0]
    if len(missing_rows) > 0:
        log.info('Adding %d rows' % len(missing_rows))
        # sup_data = np.repeat(missing_rows.reshape([len(missing_rows), 1]), 2, axis=1)
        sup_data = np.zeros([len(missing_rows), 3])
        sup_data[:, 0] = missing_rows
        train = np.vstack([train, sup_data])

    missing_cols = np.where(np.in1d(np.arange(cols), np.unique(train[:, 1]), assume_unique=True, invert=True))[0]
    if len(missing_cols) > 0:
        log.info('Adding %d cols' % len(missing_cols))
        sup_data = np.zeros([len(missing_cols), 3])
        sup_data[:, 1] = missing_cols

        train = np.vstack([train, sup_data])

    # now adding the +1 and saving
    train[:, :-1] += 1
    fu.np_save_txt(args.output, 'train.tsv', train.astype(int), delimiter='\t', fmt='%d')


if __name__ == '__main__':
    main_func()
