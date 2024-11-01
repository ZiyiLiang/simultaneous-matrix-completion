import sys
import json
import csv
import numpy as np
import pandas as pd
import pdb



class Load_MovieLens():
    """ 
    This class loads Movielens dataset and additional information regarding users and movies
    such as demographic information and movie genres etc. If one only needs the rating matrix,
    use the function load_data below (not the function within this class) to avoid additional memory usage. 
    """
    def __init__(self, data_path):
        self.data_path = data_path
        self.userID = None
        self.movieID = None
    
    def load_data(self, replace_nan=-1, num_rows=None, num_columns=None, random_state=None):
        data_raw = pd.read_csv(self.data_path+"/u.data", sep='\t', names=['userid', 'movieid', 'rating', 'timestamp'])
        data = data_raw.pivot(index='movieid', columns='userid', values='rating')

        n1, n2 = data.shape
        if random_state is not None:
            rng = np.random.default_rng(random_state)
        else:
            # Calculate non-missing values for each row and column
            row_notna_counts = data.notna().sum(axis=1)
            column_notna_counts = data.notna().sum(axis=0)

        if num_rows is not None:
            # Make sure the # of rows is meaningful
            num_rows = int(np.clip(num_rows, 1, n1))
            if random_state:
                selected_row_indices = rng.choice(data.index, size=num_rows, replace=False)
            else:
                selected_row_indices = row_notna_counts.nlargest(num_rows).index
        else:
            selected_row_indices = data.index

        if num_columns is not None:
            num_columns = int(np.clip(num_columns, 1, n2))
            if random_state:
                selected_column_indices = rng.choice(data.columns, size=num_columns, replace=False)
            else:
                selected_column_indices = column_notna_counts.nlargest(num_columns).index
        else:
            selected_column_indices = data.columns
        
        # Filter the Data to include only the selected rows and columns
        subsample_data = data.loc[selected_row_indices, selected_column_indices]
        mask_obs = subsample_data.notna().values
        mask_miss = subsample_data.isna().values
        M = subsample_data.fillna(replace_nan).values

        # Store the userID and movieID for retrival of other information
        self.userID = subsample_data.columns.tolist()
        self.movieID = subsample_data.index.tolist()

        return M, mask_obs, mask_miss
    

    def load_demographics(self):
        if self.userID is None:
            print("Error: Run load_data first to specify the user group.")
            return None

        try:
            user_data = pd.read_csv(
                self.data_path + "/u.user",
                sep='|',
                names=['userid', 'age', 'gender', 'occupation', 'zip_code']
            )
            # Filter for only selected user IDs in self.userID
            filtered_user_data = user_data[user_data['userid'].isin(self.userID)]
            return filtered_user_data
    
        except FileNotFoundError:
            print("Error: u.user file not found in the specified data path.")
            return None


    def load_movie_info(self, genre_only=True):
        if self.movieID is None:
            print("Error: Run load_data first to specify the movie group.")
            return None

        try:
            # Define column names based on the structure of u.item
            column_names = [
                'movieid', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
                'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
                'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
            ]
            
            # Load the data
            movie_data = pd.read_csv(
                self.data_path + "/u.item",
                sep='|', encoding="ISO-8859-1", names=column_names, usecols=range(24)
            )
            
            # Filter for selected movie IDs only
            filtered_movie_data = movie_data[movie_data['movieid'].isin(self.movieID)]
            
            if genre_only:
                # Select only genre columns if genre_only is True
                genre_columns = column_names[5:]  # Genres start from the 6th column
                filtered_movie_data = filtered_movie_data[['movieid'] + genre_columns]     
            return filtered_movie_data

        except FileNotFoundError:
            print("Error: u.item file not found in the specified data path.")
            return None


        

def load_data(base_path, data_name, replace_nan=-1, num_rows=None, num_columns=None, random_state=None):
    # Load the data
    if data_name == "movielens":
        data_raw = pd.read_csv(base_path+"/ml-100k/u.data", sep='\t', names=['userid', 'movieid', 'rating', 'timestamp'])
        data = data_raw.pivot(index='movieid', columns='userid', values='rating')    
    elif data_name == "books":
        data_raw = pd.read_csv(base_path+"/amazon/small_books.csv")
        data = data_raw.pivot_table(index='user_id', columns='item_id', values='rating')
    elif data_name == "myanimelist":
        data_raw = pd.read_csv(base_path+"/myanimelist/rating.csv")
        
        # replace -1 with NA since -1 indicates user watched but did not assign rating.
        data_raw['rating'].replace(-1, pd.NA, inplace=True)
        data = data_raw.pivot_table(index='user_id', columns='anime_id', values='rating')
    else:
        print("Unknown dataset!")

    n1, n2 = data.shape
    if random_state is not None:
        rng = np.random.default_rng(random_state)
    else:
        # Calculate non-missing values for each row and column
        row_notna_counts = data.notna().sum(axis=1)
        column_notna_counts = data.notna().sum(axis=0)

    if num_rows is not None:
        # Make sure the # of rows is meaningful
        num_rows = int(np.clip(num_rows, 1, n1))
        if random_state:
            selected_row_indices = rng.choice(data.index, size=num_rows, replace=False)
        else:
            selected_row_indices = row_notna_counts.nlargest(num_rows).index
    else:
        selected_row_indices = data.index

    if num_columns is not None:
        num_columns = int(np.clip(num_columns, 1, n2))
        if random_state:
            selected_column_indices = rng.choice(data.columns, size=num_columns, replace=False)
        else:
            selected_column_indices = column_notna_counts.nlargest(num_columns).index
    else:
        selected_column_indices = data.columns
    
    # Filter the Data to include only the selected rows and columns
    subsample_data = data.loc[selected_row_indices, selected_column_indices]
    mask_obs = subsample_data.notna().values
    mask_miss = subsample_data.isna().values
    M = subsample_data.fillna(replace_nan).values

    return M, mask_obs, mask_miss



def amazon_books_small(base_path):
    least_user_ratings = 30
    least_item_ratings = 800

    df = pd.read_csv(base_path+'/amazon_full/Books.csv', names=["item_id", "user_id", "rating", "timestamp"])

    # Extract item_id that start with "00" and sort the dataframe by timestamp in descending order
    df_small = df[df['item_id'].str.startswith('00')].sort_values(by='timestamp', ascending=False)
    del df

    # Drop duplicates ratings based on user_id and item_id, keeping the latest rating
    df_small.drop_duplicates(subset=['user_id', 'item_id'], keep='first', inplace=True)
    df_small.drop(columns=['timestamp'], inplace=True)

    # Count the number of ratings for each user_id and item_id
    user_ratings = df_small['user_id'].value_counts()
    item_ratings = df_small['item_id'].value_counts()

    # Filter out user_id and item_id
    valid_user_ids = user_ratings[user_ratings >= least_user_ratings].index
    valid_item_ids = item_ratings[item_ratings >= least_item_ratings].index

    filtered_df = df_small[(df_small['user_id'].isin(valid_user_ids)) & (df_small['item_id'].isin(valid_item_ids))]
    filtered_df.to_csv(base_path+'/amazon/small_books.csv', index=False)
    print("Small Amazon books dataset saved.")
