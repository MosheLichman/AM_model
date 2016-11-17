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


def _learn_params(model, train_data, val_data):
    """Learns the mixing components for each user as well as the hyper parameters of the model.

    In the learning process, the train data is used to compute the different components of the AR model. The val_data
    is used to optimize the mixing weights and the hyper parameters.The train data.

    Args:
        model: AR model instance.
                Initialized AR model ready for training.
        train_data: (R, C) lil_mat.
        val_data: (N, 3) ndarray.
                  Sparse representation, each entry is [user_id, item_id, #Observations].

    Returns:
        learned_params: dict.
                        Contains the learn parameters for the AR model.
    """
    model.training(train_data, val_data)
    return model.get_params()


def main_func():
    """Scripts main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', type=str, help='Path to training matrix.')
    parser.add_argument('-val', type=str, help='Path to validation matrix.')
    parser.add_argument('-row_smooth', type=str, help='Path to row smoothing matrix.')
    parser.add_argument('-col_smooth', type=str, help='Path to column smoothing matrix.')

    parser.add_argument('-output', type=str, help='Output folder.')

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
    obs_mat_raw = fu.np_load(args.train)
    train_data = sparse.coo_matrix((obs_mat_raw[:, 2], (obs_mat_raw[:, 0], obs_mat_raw[:, 1])),
                                shape=(R, C)).tolil()

    # Converting the validation data to [row_id, col_id] X number of obs. It's not as efficient but it's
    # much easier to work with that data.
    val_mat_raw = fu.np_load(args.val)
    val_data = np.repeat(val_mat_raw[:, :-1], val_mat_raw[:, -1].astype(int), axis=0)

    model = am_factory.get_model(args)
    model_params = _learn_params(model, train_data, val_data)

    fu.pkl_dump(args.output, '%s_params.pkl' % args.model, model_params)

    tm.print_summary()


if __name__ == '__main__':
    """
    %run scripts/learn_params.py -train /extra/mlichman0/all_data/redit_sample/train_data.pkl -val /extra/mlichman0/all_data/redit_sample/val_data.pkl -row_smooth /extra/mlichman0/all_data/redit_sample/user_smooth.pkl -col_smooth /extra/mlichman0/all_data/redit_sample/category_smooth.pkl -output /extra/mlichman0/all_data/redit_sample/ -p 10 -model gar2 -r 11100 -c 7782 -v
    """
    main_func()
