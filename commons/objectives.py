"""
Objective functions that are used in our work.

Authors:
    1. Moshe Lichman
    2. Dimitrios Kotzias

"""
from __future__ import division
import numpy as np


def _obj_logP(user_mult, u_test, return_all=False):
    """
    Returns the avg. logP of the u_test data.

     INPUT:
    -------
        1. user_mult:   <(L, )> ndarray>    user probabilities. Sums to 1.
        2. u_test:      <(N_te, ) ndarray>  user test points.

     OUTPUT:
    --------
        1. avg_logP:    <float>     avg. logP of the test points
    """
    if return_all:
        return np.log(user_mult[u_test.astype(int)])
    return np.mean(np.log(user_mult[u_test.astype(int)]))


def _obj_Erank(u_scores, u_test, return_all=False):
    """
    Returns the avg. expected rank of the u_test points. 1 is the best.
    The expected part in it means that if several objects has the same score in the u_scores vector we will return
    the average of them.

    For example:
        u_scores = [2, 2, 0, 3]
        u_test = [1]    (This is the indexes, so it corresponds to the score 2 in the u_scores.

        First step would be to find the avg. ranks of the scores --> [ 2.5, 2.5, 4, 1]  (1 is the best)
        Then the avg score of u_test is 2.5 (only one point :) )

        The returned results is than (L - 2.5 + 1) / L (where L is the size of u_scores) --> (4 - 2.5 + 1) / 4 = 0.65.

        According to this 0 is the worst eRank (but you can't really get 0) and 1 is the best.


     INPUT:
    -------
        1. u_scores:    <(L, )> ndarray>    user scores. Doesn't have to be probability (to accommodate for SVD like)
        2. u_test:      <(N_te, ) ndarray>  user test points.

     OUTPUT:
    --------
        1. avg_Erank:    <float (0, 1]>     avg. logP of the test points (1 is the best)
    """
    L = u_scores.shape[0]
    rank = np.zeros(L)

    vals, idxs, counts = np.unique(u_scores, return_index=True, return_counts=True)

    # The expected thing makes non-trivial.
    # I need to find ranks that are equal, and average across their indexes. Only then I can look
    # at the test points
    prev_rank = 0
    for i in range(idxs.shape[0]):
        c = counts[i]
        if c == 1:
            rank[idxs[i]] = prev_rank + 1
            prev_rank += 1
        else:
            curr_rank = prev_rank + (1 + c) / 2
            mask = np.where(u_scores == vals[i])[0]
            rank[mask] = curr_rank
            prev_rank += c

    e_ranks = L - rank + 1
    u_ranks = e_ranks[u_test.astype(int)]  # The best score is here 1 and the worst is L
    u_ranks = (L - u_ranks + 1) / L  # Converting to [1, 0). 1 is perfect.

    if return_all:
        return u_ranks
    else:
        return np.mean(u_ranks)

# factory
obj_func = {'logP': _obj_logP,
            'erank': _obj_Erank}
