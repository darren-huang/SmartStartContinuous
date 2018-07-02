"""Module that contains random utility functions used in the smartstart package

"""
import os

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../',
                   'data/default')
if not os.path.exists(DIR):
    os.makedirs(DIR)


def get_data_directory(file):
    """Creates and returns a data directory at the file's location

    Parameters
    ----------
    file :
        python file

    Returns
    -------
    :obj:`str`
        filepath to the data directory

    """
    fp = os.path.dirname(os.path.abspath(file))
    fp = os.path.join(fp, 'data')
    if not os.path.exists(fp):
        os.makedirs(fp)
    return fp

def get_default_directory(foldername):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../',
                 'data/')
    path = os.path.join(path, foldername)
    if not os.path.exists(path):
        os.makedirs(path)
    return path