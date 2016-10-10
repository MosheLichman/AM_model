"""
Author: Moshe Lichman
"""
from __future__ import division
import numpy as np
from scipy.sparse import coo_matrix
from scipy import sparse

import argparse

from commons import log_utils as log
from commons import file_utils as fu
from commons import time_measure as tm
from commons import helpers
from commons import objectives
from sklearn.decomposition import NMF


def factorize_mat(train_data, k):
    # First loading the mf
    factor_point = tm.get_point('nmf factor')
    model = NMF(n_components=k, solver='cd', init='nndsvd', alpha=4)
    W = model.fit_transform(train_data)
    H = model.components_
    factor_point.collect()

    return W, H


def evaluation(train_data, test_data, k, num_proc):
    """
    Performs an evaluation of the test data using the learned parameters (object properties).

     INPUT:
    -------
        1. objective:       <string>            objective function to use for evaluation
        2. train_data:      <(U, L) csr_mat>    user observation data - used to learn the components
        3. test_data:       <(N, 2) ndarray>    each row is a test event [user_id, loc_id]

     OUTPUT:
    --------
        1. scores:              <(U_te, 2) ndarray>     avg. score for each user in the test data.
    """
    log.info('Running model evaluation')
    eval_point = tm.get_point('evaluation')
    W, H = factorize_mat(train_data, k)

    obj_scores = []

    uids = np.unique(test_data[:, 0])
    batch_size = int(np.ceil(uids.shape[0] / num_proc))
    scores = []
    helpers.quque_on_uids(num_proc, uids, batch_size, _mp_user_test, ('erank', test_data, W, H), scores.extend)

    obj_scores.append(np.array(scores))

    eval_point.collect()
    return obj_scores


def _mp_user_test(queue, uids, args):
    objective, test_data, W, H = args

    mf = np.dot(W, H)
    results = []
    for i in range(len(uids)):
        user_eval_point = tm.get_point('user_eval')
        uid = uids[i]
        u_test = test_data[np.where(test_data[:, 0] == uid)[0], 1].astype(int)

        user_mult = mf[uid]

        results.append([uid, objectives.obj_func[objective](user_mult, u_test)])
        user_eval_point.collect()

    queue.put(results)


def main_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', type=str, help='Path to training matrix.')
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
    train_data = sparse.coo_matrix((obs_mat_raw[:, 2], (obs_mat_raw[:, 0], obs_mat_raw[:, 1])),
                                   shape=(R, C)).tolil()

    # Converting the validation data to [row_id, col_id] X number of obs. It's not as efficient but it's
    # much easier to work with that data.
    test_mat_raw = fu.pkl_load(args.test)
    test_data = np.repeat(test_mat_raw[:, :-1], test_mat_raw[:, -1].astype(int), axis=0)

    results = evaluation(train_data, test_data, args.k, args.num_proc)

    tm.print_summary()
    return results


if __name__ == '__main__':
    res = main_func()
