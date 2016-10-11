"""
Author: Moshe Lichman
"""
from __future__ import division
from multiprocessing import Process, Queue
import numpy as np


def unique_two_cols(col_1, col_2):
    """
    Finding a unique instances of pairs. Basically it just creates a coo representation of the data.

     INPUT:
    -------
        1. col_1:       <(N, ) ndarray>    identifier 1
        2. col_2:       <(N, ) ndarray>    identifier 2

     OUTPUT:
    --------
        1. coo:     <(N', 3) ndarray>       coo representation [ident_1, ident_2, counts]
    """

    if col_1.shape[0] != col_2.shape[0]:
        raise AssertionError('Two columns are not the same length.')

    x, y = col_1, col_2
    b = x + y * 1.0j
    idx, counts = np.unique(b, return_index=True, return_counts=True)[1:]

    tmp = np.vstack([col_1[idx], col_2[idx], counts.astype(float)]).T

    return tmp


def quque_on_uids(num_proc, uids, batch_size, target, args, collect_func):
    queue = Queue()
    if num_proc > 1:
        proc_pool = []
        for i in range(num_proc):
            p_uids = uids[i * batch_size:(i + 1) * batch_size]
            if len(p_uids) == 0:
                break

            # Adding the p_uids and queue to args
            proc = Process(target=target, args=(queue, p_uids, args))
            proc_pool.append(proc)

        [proc.start() for proc in proc_pool]
        for _ in range(len(proc_pool)):
            resp = queue.get()
            collect_func(resp)

        [proc.join() for proc in proc_pool]
    else:
        target(queue, uids, args)
        resp = queue.get()
        collect_func(resp)
