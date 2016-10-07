"""
Author: Moshe Lichman
"""
from __future__ import division

from scipy import sparse

from am_model import am_indiv
from am_model import am_global

from commons import file_utils as fu


def get_model(args):
    return _factory[args.model](args)


def _indiv_ar_2(args):
    return am_indiv.AMIndiv(args.U, args.I, 2, args.num_proc)


def _global_ar_2(args):
    return am_global.AMGlobal(args.U, args.I, 2, args.num_proc)


def _load_smooth_mat(path, dim):
    raw = fu.pkl_load(path)
    return sparse.coo_matrix((raw[:, 2], (raw[:, 0], raw[:, 1])), shape=(dim, dim))


def _indiv_ar_4(args):
    user_smooth = _load_smooth_mat(args.row_smooth, args.U).tocsr()
    item_smooth = _load_smooth_mat(args.col_smooth, args.I).tocsr()

    return am_indiv.AMIndiv(args.U, args.I, 4, args.num_proc, user_smooth, item_smooth)


def _global_ar_4(args):
    user_smooth = _load_smooth_mat(args.row_smooth, args.U).tocsr()
    item_smooth = _load_smooth_mat(args.col_smooth, args.I).tocsr()

    return am_global.AMGlobal(args.U, args.I, 4, args.num_proc, user_smooth, item_smooth)


_factory = {'iar2': _indiv_ar_2, 'gar2': _global_ar_2, 'iar4': _indiv_ar_4, 'gar4': _global_ar_4}