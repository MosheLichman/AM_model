"""
Author: Moshe Lichman
"""
from __future__ import division
import numpy as np

from commons import time_measure as tm
from commons import objectives
from commons import helpers
from commons import log_utils as log


def evaluation(objective, test_data, W, H, num_proc):
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

    obj_scores = []

    uids = np.unique(test_data[:, 0])
    batch_size = int(np.ceil(uids.shape[0] / num_proc))
    scores = []
    helpers.quque_on_uids(num_proc, uids, batch_size, mp_user_test,
                          (objective, test_data, W, H), scores.extend)

    obj_scores.append(np.array(scores))

    eval_point.collect()
    return obj_scores


def mp_user_test(queue, uids, args):
    objective, test_data, W, H = args

    results = []
    for i in range(len(uids)):
        user_eval_point = tm.get_point('user_eval')
        uid = uids[i]
        u_test = test_data[np.where(test_data[:, 0] == uid)[0], 1].astype(int)

        u_mf = np.dot(W[uid], H)
        user_mult = u_mf        # Other places I'll have to normalize
        if objective is 'logP':
            if np.any(user_mult == 0):
                log.info('mf_commons.mp_user_test: User %d with 0 probability' % uid)
            user_mult /= np.sum(user_mult)

        results.append([uid, objectives.obj_func[objective](user_mult, u_test)])
        user_eval_point.collect()

    queue.put(results)

