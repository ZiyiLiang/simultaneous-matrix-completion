"""This file contains utility functions for processing largescale datasets.
    Functions are adpated from the Surprise repo: https://github.com/NicolasHug/Surprise/tree/master"""

import numpy as np
from surprise import Trainset
from collections import defaultdict
from typing import List, Tuple, Callable


class DataReader:
    """Reader class for parsing rating files.
    
    Simplified version without built-in datasets support.
    
    Parameters
    ----------
    line_format : str, default='user item rating'
        The field names in the order they appear in each line.
        Supported fields: 'user', 'item', 'rating', 'timestamp'
    sep : str, optional
        The separator between fields. If None, uses whitespace.
    rating_scale : tuple, default=(1, 5)
        The (min, max) rating scale used in the dataset.
    skip_lines : int, default=0
        Number of lines to skip at the beginning of the file.
    """
    
    def __init__(self, 
                 line_format='user item rating',
                 sep=None,
                 rating_scale=(1, 5),
                 skip_lines=0):
        
        self.sep = sep
        self.skip_lines = skip_lines
        self.rating_scale = rating_scale
        
        # Parse the line format
        splitted_format = line_format.split()
        
        # Define valid entities
        entities = ['user', 'item', 'rating']
        if 'timestamp' in splitted_format:
            self.with_timestamp = True
            entities.append('timestamp')
        else:
            self.with_timestamp = False
        
        # Validate all fields are recognized
        invalid_fields = [f for f in splitted_format if f not in entities]
        if invalid_fields:
            raise ValueError(f"Invalid fields in line_format: {invalid_fields}. "
                           f"Valid fields are: {entities}")
        
        # Check that required fields are present
        required = ['user', 'item', 'rating']
        missing = [f for f in required if f not in splitted_format]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        
        # Store the index of each entity in the line
        self.indexes = [splitted_format.index(entity) for entity in entities]
    
    def parse_line(self, line):
        """Parse a single line from the ratings file.
        
        Parameters
        ----------
        line : str
            The line to parse
            
        Returns
        -------
        tuple
            (user_id, item_id, rating)
        """
        line = line.split(self.sep)
        
        try:
            uid, iid, r = (line[i].strip() for i in self.indexes[:3])
                
        except IndexError:
            raise ValueError(f"Cannot parse line: {line}. "
                           f"Check line_format and sep parameters.")
        
        return uid, iid, float(r)


def construct_trainset(raw_trainset, rating_scale):
    raw2inner_id_users = {}
    raw2inner_id_items = {}

    current_u_index = 0
    current_i_index = 0

    ur = defaultdict(list)
    ir = defaultdict(list)

    # user raw id, item raw id, translated rating
    for urid, irid, r in raw_trainset:
        try:
            uid = raw2inner_id_users[urid]
        except KeyError:
            uid = current_u_index
            raw2inner_id_users[urid] = current_u_index
            current_u_index += 1
        try:
            iid = raw2inner_id_items[irid]
        except KeyError:
            iid = current_i_index
            raw2inner_id_items[irid] = current_i_index
            current_i_index += 1

        ur[uid].append((iid, r))
        ir[iid].append((uid, r))

    n_users = len(ur)  # number of users
    n_items = len(ir)  # number of items
    n_ratings = len(raw_trainset)

    trainset = Trainset(
        ur,
        ir,
        n_users,
        n_items,
        n_ratings,
        rating_scale,
        raw2inner_id_users,
        raw2inner_id_items,
    )

    return trainset


def create_batch_rating_predictor(algo) -> Callable[[List[Tuple]], Tuple[List[float], List[bool]]]:
    """
    Adapter Factory: Takes a trained Surprise algorithm and returns a unified
    batch prediction function.
    [Note] This adapter does not handle unseen user/item since surpirse give a 
    default prediction for unseen indices. Make adjustment if algo does not handle
    this corner case.

    Args:
        algo: A trained algorithm object from the Surprise library (or any
              object with a .predict(uid, iid) method that returns an
              object with an '.est' attribute).

    Returns:
        A function that adheres to the unified prediction interface.
    """
    def batch_predictor(indices: List[Tuple]) -> List[float]:
        results = [
            (p.est, p.details['was_impossible']) 
            for p in (algo.predict(uid, iid) for uid, iid in indices)
        ]
        predictions, was_impossible = zip(*results)
    
        return list(predictions), list(was_impossible)
    
    return batch_predictor


def default_weight(indices: List[Tuple]) -> List[float]:
    return np.ones(len(indices))


def evaluate_SCI_ls(lower, upper, k, ratings, is_inf=None, 
                    metric='mean', method=None):
    """ This function evaluates the coverage over test queries
    """

    val = ratings 
    n_test_queries = len(ratings) // int(k) 
    
    covered = np.array((lower <= val) & (upper >= val))
    query_covered = [np.prod(covered[k*i:k*(i+1)]) for i in range(n_test_queries)]
    query_coverage = np.mean(query_covered)
    coverage = np.mean(covered)
    sizes = upper - lower
    if metric == 'mean':
        size = np.mean(sizes)
    elif metric == 'median':
        size = np.median(sizes)
    elif metric == 'no_inf':
        if is_inf is None:
            print('Must provide inf locations for the no_inf metric.')
        else:
            query_no_inf = [not np.any(is_inf[k*i:k*(i+1)]) for i in range(n_test_queries)]
            idxs_no_inf = [element for element in query_no_inf for _ in range(int(k))]
        size = np.mean(sizes[idxs_no_inf])
    else:
        print(f'Unknown evaluation metric {metric}')
    results = pd.DataFrame({})
    results["Query_coverage"] = [query_coverage]
    results["Coverage"] = [coverage]
    results["Size"] = [size]
    results["metric"] = [metric]
    if type(is_inf) != type(None):
        query_is_inf = [np.any(is_inf[k*i:k*(i+1)]) for i in range(n_test_queries)]
        results["Inf_prop"] = [np.mean(query_is_inf)]
    if method:
        results["Method"] = [method]
    return results