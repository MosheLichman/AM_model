"""
Author: Moshe Lichman
"""
import argparse
import numpy as np
from scipy import sparse

from commons import file_utils as fu
from commons import log_utils as log
from commons import time_measure as tm

from am_model import am_factory


def evaluate_am(objective, model, train_data, test_data, params):
    results = model.evaluation(train_data, test_data, params, objective=objective)
    return results


def main_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', type=str, help='Path to training matrix.')
    parser.add_argument('-val', type=str, help='Path to validation matrix.')
    parser.add_argument('-test', type=str, help='Path to validation matrix.')
    parser.add_argument('-obj', type=str, help='Path to validation matrix.')
    parser.add_argument('-params', type=str, help='Path to learned params')
    parser.add_argument('-row_smooth', type=str, help='Path to row smoothing matrix.')
    parser.add_argument('-col_smooth', type=str, help='Path to column smoothing matrix.')

    parser.add_argument('-num_proc', type=int, help='Num processors', default=2)

    parser.add_argument('-model', type=str, help='Which models')

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

    model = am_factory.get_model(args)

    params = fu.pkl_load(args.params)
    results = evaluate_am(model, train_data, test_data, params, args.obj)

    tm.print_summary()
    return results


if __name__ == '__main__':
    res = main_func()

