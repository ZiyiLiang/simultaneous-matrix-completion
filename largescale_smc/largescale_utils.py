"""This file contains utility functions for processing largescale datasets.
    Functions are adpated from the Surprise repo: https://github.com/NicolasHug/Surprise/tree/master"""

import pdb
import pandas as pd
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


def create_batch_rating_predictor(algo) -> Callable[[List[Tuple]], Tuple[np.ndarray, np.ndarray]]:
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
    def batch_predictor(indices: List[Tuple]) -> Tuple[np.ndarray, np.ndarray]:
        n = len(indices)
        predictions = np.empty(n, dtype=np.float64)
        was_impossible = np.empty(n, dtype=np.bool_)

        for i, (uid, iid) in enumerate(indices):
            p = algo.predict(uid, iid)
            predictions[i] = p.est
            was_impossible[i] = bool(p.details.get('was_impossible', False))

        return predictions, was_impossible

    return batch_predictor


def default_weight(indices: List[Tuple]) -> np.ndarray:
    return np.ones(len(indices), dtype=np.float64)


def evaluate_SCI_ls(lower, upper, k, ratings, is_inf=None, is_impossible=None,
                    metric='mean', filter_impossible=True, method=None):
    """ This function evaluates the coverage over test queries
    """
    
    # Get the original number of queries before any filtering
    n_original_queries = len(ratings) // int(k)
    impossible_prop = 0.0

    # Filter if requested and the 'is_impossible' array is provided
    if filter_impossible and is_impossible is not None:
        # Reshape to (n_queries, k) and sum along the query axis to find impossible counts
        impossible_counts_per_query = np.sum(np.array(is_impossible).reshape(n_original_queries, int(k)), axis=1)

        valid_queries_mask = impossible_counts_per_query == 0
        
        # Calculate the proportion of queries that were filtered out
        n_impossible_queries = n_original_queries - np.sum(valid_queries_mask)
        if n_original_queries > 0:
            impossible_prop = n_impossible_queries / n_original_queries

        # Create a boolean mask for the original flat arrays by repeating the query mask
        valid_indices_mask = np.repeat(valid_queries_mask, int(k))

        # Apply the filter to all relevant data arrays
        lower = np.array(lower)[valid_indices_mask]
        upper = np.array(upper)[valid_indices_mask]
        ratings = np.array(ratings)[valid_indices_mask]
        if is_inf is not None:
            is_inf = np.array(is_inf)[valid_indices_mask]

        print(f'Filtered out {n_impossible_queries} impossible queries out of {n_original_queries} queries.')
    
    val = ratings 
    n_test_queries = len(ratings) // int(k) 
    if n_test_queries == 0:
        print(f'No valid test queries for evaluation!')
        return
    else:
        print(f'Evaluating {n_test_queries} queries...')

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
    results["Impossible_prop"] = [impossible_prop]
    if type(is_inf) != type(None):
        query_is_inf = [np.any(is_inf[k*i:k*(i+1)]) for i in range(n_test_queries)]
        results["Inf_prop"] = [np.mean(query_is_inf)]
    if method:
        results["Method"] = [method]

    print(f'Done!')
    return results