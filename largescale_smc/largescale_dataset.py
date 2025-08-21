"""
Functions for loading largescale datasets and perform sample splittings.
"""

import os
import itertools


class DataSplitter:
    """Handles data loading, splitting, and query sampling for matrix completion.
    
    Parameters
    ----------
    file_path : str
        Path to ratings file
    reader : Reader
        Reader object for parsing the file
    """
    
    def __init__(self, file_path, reader):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File {file_path} not found")
        
        self.file_path = file_path
        self.reader = reader

        # Load ratings
        self.raw_ratings = self._load_ratings()
        #print(f"Loaded {self.n_ratings} ratings: {self.n_users} users, {self.n_items} items")

    def _load_ratings(self):
        """Return a list of ratings (user, item, rating) read from
        file_name"""

        with open(os.path.expanduser(self.file_path)) as f:
            raw_ratings = [
                self.reader.parse_line(line)
                for line in itertools.islice(f, self.reader.skip_lines, None)
            ]
        return raw_ratings