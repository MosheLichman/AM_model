"""
File methods wrapper. For the loading wrappers it just prints out loading times.
For the saving wrappers, if creates the dir if it doesn't exist and takes care of the permissions.

Author: Moshe Lichman
"""
from __future__ import division
import numpy as np
import time
import pickle
import os
from os.path import join

from commons import log_utils as log


def load(file_path):
    if file_path.endswith('.pkl'):
        return pkl_load(file_path)
    if file_path.endswith('.npy'):
        return np_load(file_path)
    print 'Dont know how to open %s' % file_path


def pkl_load(file_path):
    """
    Wrapper for pickle load that prints time as well.

     INPUT:
    -------
        1. file_path:   <string>    file path

     OUTPUT:
    --------
        1. data:    <?> whatever that was pickled

     RAISE:
    -------
        1. IOError
    """

    log.info('Loading %s' % file_path)
    start = time.time()
    with open(file_path, 'r') as f:
        tmp = pickle.load(f)

    log.info('Loading took %d seconds' % (time.time() - start))
    return tmp


def pkl_dump(path, file_name, data):
    """
    Wrapper for pickle.dump that also creates the dir if doesn't exist and fixes permissions.

     INPUT:
    -------
        1. path:        <sting>     dir path
        2. file_name:   <string>    file name
        3. data:        <?>         data
    """
    log.info('Saving file %s' % join(path, file_name))
    _make_dir(path)

    start = time.time()
    with open(join(path, file_name), 'w') as f:
        pickle.dump(data, f)

    os.chmod(join(path, file_name), 0770)
    log.info('Saving took %d seconds' % (time.time() - start))


def np_load_txt(file_path, delimiter=','):
    """
    Wrapper for np.loadtxt that also prints the time.

     INPUT:
    -------
        1. file_path:   <string>    file path
        2. delimiter:   <string>    delimiter in the file (default = ',' csv file)

     OUTPUT:
    --------
        1. data:    <ndarray>   numpy array of the data

     RAISE:
    -------
        1. IOError
    """
    log.info('Loading %s' % file_path)
    start = time.time()
    data = np.loadtxt(file_path, delimiter=delimiter)
    log.info('Loading took %d seconds' % (time.time() - start))

    return data


def _make_dir(path):
    """
    Making sure that the dir exist. If not, creating it with the write permissions.

     INPUT:
    -------
        1. path:    <string>    dir path
    """
    if not os.path.exists(path):
        os.makedirs(path)
        os.chmod(path, 0770)


def np_save_txt(path, file_name, data, delimiter=',', fmt='%.5f'):
    """
    Wrapper for np.savetxt that also creates the dir if doesn't exist

     INPUT:
    -------
        1. path:        <sting>     dir path
        2. file_name:   <string>    file name
        3. data:        <ndarray>   numpy array
        4. delimiter:   <string>    delimiter in the file (default = ',' csv file)
        5. fmt:         <string>    format
    """
    log.info('Saving file %s/%s' % (path, file_name))
    _make_dir(path)

    start = time.time()
    np.savetxt(join(path, file_name), data, delimiter=delimiter, fmt=fmt)
    os.chmod(join(path, file_name), 0770)
    log.info('Saving took %d seconds' % (time.time() - start))


def np_load(file_path):
    """
    Wrapper fpr the np.load that also prints time.

     INPUT:
    -------
        1. file_path:   <string>    file path

     OUTPUT:
    --------
        1. data:    <?>     whatever was saved

     RAISE:
    -------
        1. IOError
    """
    log.info('Loading %s' % file_path)
    start = time.time()
    data = np.load(file_path)
    log.info('Loading took %d seconds' % (time.time() - start))

    return data


def np_save(path, file_name, data):
    """
    Wrapper for np.save that also creates the dir if doesn't exist

     INPUT:
    -------
        1. path:        <sting>     dir path
        2. file_name:   <string>    file name
        3. data:        <ndarray>   numpy array
    """
    log.info('Saving file %s/%s' % (path, file_name))
    _make_dir(path)

    start = time.time()
    np.save(join(path, file_name), data)
    os.chmod(join(path, file_name), 0770)
    log.info('Saving took %d seconds' % (time.time() - start))
