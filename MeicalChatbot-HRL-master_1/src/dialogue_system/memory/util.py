import numpy as np
from collections import deque
import operator
import pydash as ps


def batch_get(arr, idxs):
    '''Get multi-idxs from an array depending if it's a python list or np.array'''
    if isinstance(arr, (list, deque)):
        return np.array(operator.itemgetter(*idxs)(arr))
    else:
        return arr[idxs]

def set_attr(obj, attr_dict, keys=None):
    '''Set attribute of an object from a dict'''
    if keys is not None:
        attr_dict = ps.pick(attr_dict, keys)
    for attr, val in attr_dict.items():
        setattr(obj, attr, val)
    return obj