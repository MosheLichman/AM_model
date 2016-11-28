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


def evaluate_am(objective, model, train_data, test_data, params):
    results = model.evaluation(train_data, test_data, params, objective=objective)
    return results


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
        print 'logP: %.4f' % np.mean(np.array(logP_results)[:, 1])
        print 'erank: %.4f' % np.mean(np.array(erank_results)[:, 1])
        return logP_results, erank_results

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

    logP_results = evaluate_am('logP', model, train_data, test_data, params)
    erank_results = evaluate_am('erank', model, train_data, test_data, params)
    print 'logP: %.4f' % np.mean(np.array(logP_results)[:, 1])
    fu.pkl_dump(args.output, logP_filename, logP_results)
    print 'erank: %.4f' % np.mean(np.array(erank_results)[:, 1])
    fu.pkl_dump(args.output, erank_filename, erank_results)

    tm.print_summary()
    return logP_results, erank_results


def get_result_filename(model, obj):
    return '%s_results_%s.pkl' % (model, obj)

    # def get_script_str(row='U', col='S', implicit_similarity=False, min_subscribers=1000, min_posts=1000,
    #                    max_words=25000, sample=False, min_col_sim=0.6, min_row_sim=0.8, rows=113557, cols=21386):
    #     input_dir = '/extra/dkotzias0/reddit_data/data/input/'
    #     run_name = '_%d_%d_%d' % (min_subscribers, min_posts, max_words)
    #     name = '%sx%s_' % (row, col)
    #
    #     train = '%s%sdata%s.npy' % (input_dir, name, run_name)
    #     validation = '%s%sdata%s_validation.npy' % (input_dir, name, run_name)
    #     test = '%s%sdata%s_test.npy' % (input_dir, name, run_name)
    #
    #     'category_similarity_data_0.600_1000_1000_25000.npy'
    #     if row == 'U':
    #         row_sim_name = 'user'
    #     elif row == 'S':
    #         row_sim_name = 'category'
    #     else:
    #         print 'Error i dont recognize row name %s' % row
    #         return
    #
    #     if col == 'S':
    #         col_sim_name = 'category'
    #     elif col == 'W':
    #         col_sim_name = 'word'
    #     else:
    #         print 'Error i dont recognize column name %s' % col
    #         return
    #
    #     implicit_str = ''
    #     if implicit_similarity:
    #         implicit_str = '_implicit'
    #
    #     row_sim = '%s%s_similarity%s_data_%.3f_%s.npy' % (input_dir, row_sim_name, implicit_str, min_row_sim, name)
    #
    #     col_sim = '%s%s_similarity%s_data_%.3f_%s.npy' % (input_dir, col_sim_name, implicit_str, min_col_sim, name)


if __name__ == '__main__':
    res = main_func()
