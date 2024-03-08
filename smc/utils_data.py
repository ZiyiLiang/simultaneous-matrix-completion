import sys
import numpy as np
import pandas as pd


class DataSet:
    def __init__(self, base_path, data_name, replace_nan=-1, random_state=None):      
        # Load the data
        if data_name == "movielens":
            data_raw = pd.read_csv(base_path+"/ml-100k/u.data", sep='\t', names=['userid', 'movieid', 'rating', 'timestamp'])
            data = data_raw.pivot(index='movieid', columns='userid', values='rating')
        else:
            print("Unknown dataset!")

        self.replace = replace_nan
        self.random_state = random_state
        self.data = data
        self.rowid = data.index
        self.columnid = data.columns


    def sample(self, num_rows=None, num_columns=None):
        n1, n2 = self.data.shape
        rng = np.random.default_rng(self.random_state)

        if num_rows is not None:
            # Make sure the # of rows is meaningful
            num_rows = int(np.clip(num_rows, 1, n1))
            random_row_indices = rng.choice(self.data.index, size=num_rows, replace=False)
        else:
            random_row_indices = self.rowid
        if num_columns is not None:
            num_columns = int(np.clip(num_columns, 1, n2))
            random_column_indices = rng.choice(self.data.columns, size=num_columns, replace=False)
        else:
            random_column_indices = self.columnid
        
        # Filter the Data to include only the selected rows and columns
        subsample_data = self.data.loc[random_row_indices, random_column_indices]
        mask_obs = subsample_data.notna().values
        mask_miss = subsample_data.isna().values
        M = subsample_data.fillna(self.replace).values.copy()

        return M, mask_obs, mask_miss

        



