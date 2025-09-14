"""
Functions for loading largescale datasets and perform sample splittings.
"""

import os
import itertools
import numpy as np
from collections import defaultdict
from numpy.random import default_rng



class DataSplitter:
    """Handles data loading, splitting, and query sampling for matrix completion.

    This class reads interaction data once and builds a single efficient dictionary
    lookup (either user-centric or item-centric) to enable scalable sampling.

    Parameters
    ----------
    file_path : str
        Path to the ratings file.
    reader : Reader
        A reader object with a `parse_line` method.
    sampling_dim : {'user', 'item'}, optional
        The dimension to use as keys for grouping interactions. This choice is 
        fixed upon initialization. Defaults to 'item'.
    """
    
    def __init__(self, file_path, reader, sampling_dim='item'):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File {file_path} not found")
        if sampling_dim not in ('user', 'item'):
            raise ValueError("sampling_dim must be either 'user' or 'item'")
        
        self.file_path = file_path
        self.reader = reader
        self.sampling_dim = sampling_dim

        # Initialize mappings and model attributes
        self.raw2inner_id_users = {}
        self.raw2inner_id_items = {}
        self.n_users = 0
        self.n_items = 0

        # Directly build the single required dictionary upon initialization.
        self._load_and_build_dict()
    

    def _load_and_build_dict(self):
        """Reads the ratings file once and builds one dictionary mapping keys to entities."""
        self.key_to_entities_map = defaultdict(list)
        
        with open(os.path.expanduser(self.file_path)) as f:
            ratings_iterator = (
                self.reader.parse_line(line)
                for line in itertools.islice(f, self.reader.skip_lines, None)
            )

            # Build raw to inner ID mappings
            current_u_index = 0
            current_i_index = 0

            for urid, irid, r in ratings_iterator:
                # Map user ID
                try:
                    uid = self.raw2inner_id_users[urid]
                except KeyError:
                    uid = current_u_index
                    self.raw2inner_id_users[urid] = current_u_index
                    current_u_index += 1
                
                # Map item ID
                try:
                    iid = self.raw2inner_id_items[irid]
                except KeyError:
                    iid = current_i_index
                    self.raw2inner_id_items[irid] = current_i_index
                    current_i_index += 1

                if self.sampling_dim == 'user':
                    self.key_to_entities_map[uid].append((iid, r))
                else:
                    self.key_to_entities_map[iid].append((uid, r))

        self.n_users = len(self.raw2inner_id_users)
        self.n_items = len(self.raw2inner_id_items)


    def _validate_query_size(self, query_size, key2query_count, max_n_query, param_name):
        """
        validation helper to check if the query size is meaningful
        """


    def sample_train_calib(self, k, calib_size=0.5, random_state=0):
        """Splits observed ratings into training and calibration sets using inner IDs.

        The sampling dimension is determined by the `sampling_dim` parameter
        provided during the initialization of the class. The returned interactions
        use the internal integer IDs, not the raw IDs from the source file.

        Parameters
        ----------
        k : int
            The size of each query.
        calib_size : float or int
            The proportion or number of queries for the calibration set.
        random_state : int
            Seed for the random number generator.

        Returns
        -------
        train_interactions : list of tuples
            (inner_user_id, inner_item_id, rating) tuples for the training set.
        calib_interactions : list of tuples
            (inner_user_id, inner_item_id, rating) tuples for the calibration set.
        """
        rng = default_rng(random_state)

        # self.key_to_entities_map is already using inner IDs.
        # Create a temporary copy for shuffling and sampling.
        keys_to_sample_from = {key: list(entities) for key, entities in self.key_to_entities_map.items()}

        # 1. Prepare for sampling: identify eligible keys and potential queries.
        eligible_keys_map = {}
        key_to_query_count = {}
        
        for key, entities in keys_to_sample_from.items():
            if len(entities) >= k:
                rng.shuffle(entities)
                num_queries = len(entities) // k
                eligible_keys_map[key] = entities[:num_queries * k]
                key_to_query_count[key] = num_queries

        # 2. Calculate the number of calibration queries to sample.
        total_possible_queries = sum(key_to_query_count.values())
        if total_possible_queries == 0:
            raise Exception('Number of possible calibration query is 0, try a smaller group size k.')

        if isinstance(calib_size, float) and 0 < calib_size < 1:
            n_queries_to_sample = int(calib_size * total_possible_queries)
        elif isinstance(calib_size, int) and 0 <= calib_size <= total_possible_queries:
            n_queries_to_sample = calib_size
        else:
            raise ValueError(f"Invalid calib_size. Must be float in (0,1) or int "
                            f"in [0, {total_possible_queries}].")

        # 3. Perform sampling.
        calib_interactions = []
        keys = list(key_to_query_count.keys())
        query_counts = np.array(list(key_to_query_count.values()), dtype=float)

        for _ in range(n_queries_to_sample):
            if not keys: break
            
            prob = query_counts / np.sum(query_counts)
            chosen_key_idx = rng.choice(len(keys), p=prob)
            chosen_key = keys[chosen_key_idx]
            
            query_entities = eligible_keys_map[chosen_key][:k]
            for entity_id, rating in query_entities:
                if self.sampling_dim == 'user':
                    calib_interactions.append((chosen_key, entity_id, rating))
                else:  # 'item'
                    calib_interactions.append((entity_id, chosen_key, rating))
            
            eligible_keys_map[chosen_key] = eligible_keys_map[chosen_key][k:]
            query_counts[chosen_key_idx] -= 1
            
            if query_counts[chosen_key_idx] == 0:
                keys.pop(chosen_key_idx)
                query_counts = np.delete(query_counts, chosen_key_idx)

        # 4. Construct the training set from all remaining interactions.
        train_interactions = []
        eligible_key_ids = set(eligible_keys_map.keys())

        # Iterate over the original, unshuffled map to construct the training set.
        for key, original_entities in self.key_to_entities_map.items():
            if key in eligible_key_ids:
                # This key was part of sampling, so add its leftovers.
                remaining_entities = eligible_keys_map[key]
                for entity_id, rating in remaining_entities:
                    if self.sampling_dim == 'user':
                        train_interactions.append((key, entity_id, rating))
                    else:
                        train_interactions.append((entity_id, key, rating))
            else:
                # This key was never eligible, so all its entities go to training.
                for entity_id, rating in original_entities:
                    if self.sampling_dim == 'user':
                        train_interactions.append((key, entity_id, rating))
                    else:
                        train_interactions.append((entity_id, key, rating))

        return train_interactions, calib_interactions