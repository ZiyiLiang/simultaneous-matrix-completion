"""This file contains utility functions for processing largescale datasets.
    Functions are adpated from the Surprise repo: https://github.com/NicolasHug/Surprise/tree/master"""

from surprise import Trainset
from collections import defaultdict

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
