"""
Author: Moshe Lichman
"""
import argparse
import os
import numpy as np
from scipy import sparse

from commons import file_utils as fu
from commons import log_utils as log
from commons import time_measure as tm

from am_model import am_factory
import matplotlib.pyplot as plt


def evaluate_am(objective, model, train_data, test_data, params, return_all=False):
    results = model.evaluation(train_data, test_data, params, objective=objective, return_all=return_all)
    return results


def compare_2_4_logP():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', type=str, help='Path to training matrix.')
    parser.add_argument('-val', type=str, help='Path to validation matrix.')
    parser.add_argument('-test', type=str, help='Path to validation matrix.')
    parser.add_argument('-row_smooth', type=str, help='Path to row smoothing matrix.')
    parser.add_argument('-col_smooth', type=str, help='Path to column smoothing matrix.')
    parser.add_argument('-output', type=str, help='Output folder of learning.')

    parser.add_argument('-num_proc', type=int, help='Num processors', default=4)
    parser.add_argument('-model', type=str, help='Which models')
    parser.add_argument('-v', help='Debug logging level', action='store_true')
    parser.add_argument('-r', type=int, help='Number of rows.')
    parser.add_argument('-c', type=int, help='Number of columns.')
    args = parser.parse_args()

    if args.v:
        log.set_verbose()

    log2 = get_detailed_log_results(args, 'gar2')
    log4 = get_detailed_log_results(args, 'gar4')
    f, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(log2, log4, c=".3")
    # ax.set(xlim=(-3, 3), ylim=(-3, 3))

    # Plot your initial diagonal line based on the starting
    # xlims and ylims.
    diag_line, = ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")

    # def on_change(axes):
    #     # When this function is called it checks the current
    #     # values of xlim and ylim and modifies diag_line
    #     # accordingly.
    #     x_lims = ax.get_xlim()
    #     y_lims = ax.get_ylim()
    #     diag_line.set_data(x_lims, y_lims)
    #
    # # Connect two callbacks to your axis instance.
    # # These will call the function "on_change" whenever
    # # xlim or ylim is changed.
    # ax.callbacks.connect('xlim_changed', on_change)
    # ax.callbacks.connect('ylim_changed', on_change)
    plt.savefig(os.path.join(args.output, 'log_scatter.pdf'))


def get_detailed_log_results(args, model):
    args.model = model
    args.params = os.path.join(args.output, '%s_params.pkl' % args.model)
    logP_filename = get_result_filename(args.model, 'logP')
    logP_path = os.path.join(args.output, logP_filename)
    if os.path.exists(logP_path):
        logP_results = fu.pkl_load(logP_path)
    else:
        R, C = args.r, args.c
        obs_mat_raw = fu.np_load(args.train)

        # Combining the validation. I can treat it as train now.
        obs_mat_val_raw = fu.np_load(args.val)
        obs_mat_raw = np.vstack([obs_mat_raw, obs_mat_val_raw])

        train_data = sparse.coo_matrix((obs_mat_raw[:, 2], (obs_mat_raw[:, 0], obs_mat_raw[:, 1])),
                                       shape=(R, C)).tolil()

        # Converting the validation data to [row_id, col_id] X number of obs. It's not as efficient but it's
        # much easier to work with that data.
        test_mat_raw = fu.np_load(args.test)
        test_data = np.repeat(test_mat_raw[:, :-1], test_mat_raw[:, -1].astype(int), axis=0)

        model = am_factory.get_model(args)

        params = fu.pkl_load(args.params)

        logP_results = evaluate_am('logP', model, train_data, test_data, params, return_all=True)
        fu.pkl_dump(args.output, logP_filename, logP_results)

    log2_array = np.zeros((0))
    for (id, results) in logP_results:
        log2_array = np.vstack((log2_array, results))
    return log2_array


def main_func():
    """Saves results. If results already exist in given name, loads and returns instead of calculating again."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', type=str, help='Path to training matrix.')
    parser.add_argument('-val', type=str, help='Path to validation matrix.')
    parser.add_argument('-test', type=str, help='Path to validation matrix.')
    parser.add_argument('-obj', type=str, help='Evaluation metric string. logP or erank')
    parser.add_argument('-params', type=str, help='Path to learned params')
    parser.add_argument('-row_smooth', type=str, help='Path to row smoothing matrix.')
    parser.add_argument('-col_smooth', type=str, help='Path to column smoothing matrix.')
    parser.add_argument('-output', type=str, help='Output folder of learning.')

    parser.add_argument('-num_proc', type=int, help='Num processors', default=2)

    parser.add_argument('-model', type=str, help='Which models')

    parser.add_argument('-r', type=int, help='Number of rows.')
    parser.add_argument('-c', type=int, help='Number of columns.')

    parser.add_argument('-v', help='Debug logging level', action='store_true')

    args = parser.parse_args()

    if args.v:
        log.set_verbose()

    tm.reset_tm()

    args.params = os.path.join(args.output, '%s_params.pkl' % args.model)
    logP_filename = get_result_filename(args.model, 'logP')
    erank_filename = get_result_filename(args.model, 'erank')
    logP_path = os.path.join(args.output, logP_filename)
    erank_path = os.path.join(args.output, erank_filename)
    if os.path.exists(logP_path) and os.path.exists(erank_path):
        logP_results = fu.pkl_load(logP_path)
        erank_results = fu.pkl_load(erank_path)
    else:
        R, C = args.r, args.c
        obs_mat_raw = fu.np_load(args.train)

        # Combining the validation. I can treat it as train now.
        obs_mat_val_raw = fu.np_load(args.val)
        obs_mat_raw = np.vstack([obs_mat_raw, obs_mat_val_raw])

        train_data = sparse.coo_matrix((obs_mat_raw[:, 2], (obs_mat_raw[:, 0], obs_mat_raw[:, 1])),
                                       shape=(R, C)).tolil()

        # Converting the validation data to [row_id, col_id] X number of obs. It's not as efficient but it's
        # much easier to work with that data.
        test_mat_raw = fu.np_load(args.test)
        test_data = np.repeat(test_mat_raw[:, :-1], test_mat_raw[:, -1].astype(int), axis=0)

        model = am_factory.get_model(args)

        params = fu.pkl_load(args.params)

        logP_results = evaluate_am('logP', model, train_data, test_data, params, return_all=True)
        erank_results = evaluate_am('erank', model, train_data, test_data, params, return_all=True)
        fu.pkl_dump(args.output, logP_filename, logP_results)
        fu.pkl_dump(args.output, erank_filename, erank_results)

    tm.print_summary()
    return logP_results, erank_results


def get_result_filename(model, obj):
    return '%s_results_detailed_%s.pkl' % (model, obj)


if __name__ == '__main__':
    res = main_func()
