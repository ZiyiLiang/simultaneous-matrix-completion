import numpy as np
import pdb
import sys
from scipy.sparse import csr_matrix
import implicit
import warnings


class LogisticMFProbs:
    """
    Efficient probability computation from LogisticMF with raw ID support.
    Algorithm of the model is described in Logistic Matrix Factorization for 
    Implicit Feedback Data <https://web.stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf>
    """
    def __init__(self, factors=30, iterations=30, regularization=0.6, 
                 learning_rate=1.0, random_state=0):
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # Initialize mappings and model attributes
        self.raw2inner_id_users = {}
        self.raw2inner_id_items = {}
        self.n_users = 0
        self.n_items = 0
        self.shape = None
        self.user_factors = None
        self.item_factors = None
    
    def fit(self, raw_trainset):
        """
        Fit LogisticMF model on training data.
        
        Parameters:
        -----------
        raw_trainset : list of tuples
            List of (user_raw_id, item_raw_id, rating) tuples
        """
        # Build raw to inner ID mappings
        current_u_index = 0
        current_i_index = 0
        
        user_indices = []
        item_indices = []
        
        for urid, irid, r in raw_trainset:
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
            
            user_indices.append(uid)
            item_indices.append(iid)
        
        self.n_users = len(self.raw2inner_id_users)
        self.n_items = len(self.raw2inner_id_items)
        self.shape = (self.n_users, self.n_items)
        
        # Create binary CSR matrix (1 for any observed rating)
        binary_values = np.ones(len(raw_trainset), dtype=np.float32)
        obs_matrix = csr_matrix(
            (binary_values, (user_indices, item_indices)),
            shape=self.shape
        )
        
        # Train LogisticMF
        model = implicit.lmf.LogisticMatrixFactorization(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            random_state=self.random_state
        )
        model.fit(obs_matrix)
        
        # Store only the factors
        self.user_factors = model.user_factors
        self.item_factors = model.item_factors

        # # Get the corresponding factor vectors for only the observed pairs
        # observed_user_vecs = self.user_factors[user_indices]
        # observed_item_vecs = self.item_factors[item_indices]

        # # Calculate the logits for every observed pair and then find the mean
        # observed_logits = np.sum(observed_user_vecs * observed_item_vecs, axis=1)
        observed_logits = [np.sum(self.user_factors[u] * self.item_factors[i]) for u, i in zip(user_indices, item_indices)]
        self.global_mean_logit = np.mean(observed_logits)

        return self
        
    def __call__(self, indices):
        """
        Compute probabilities for given raw indices.
        
        Parameters:
        -----------
        indices : list of tuples
            List of (user_id, item_id) tuples where IDs are raw IDs
            e.g., [(uid1, iid1), (uid2, iid2), ...]
            
        Returns:
        --------
        probs : ndarray
            Probabilities for each (user, item) pair
        """
        if self.user_factors is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Extract user and item IDs from list of tuples
        user_raw = [uid for uid, _ in indices]
        item_raw = [iid for _, iid in indices]
        
        # Convert to internal indices
        user_idx = np.array([self.raw2inner_id_users.get(u, -1) for u in user_raw])
        item_idx = np.array([self.raw2inner_id_items.get(i, -1) for i in item_raw])
        
        logits = np.zeros(len(indices), dtype=float)

        # Boolean mask for impossible pairs.
        impossible_mask = (user_idx == -1) | (item_idx == -1)

        # Check for invalid IDs        
        if np.any(impossible_mask):
            missing_users = {u for u, m in zip(user_raw, user_idx == -1) if m}
            missing_items = {i for i, m in zip(item_raw, item_idx == -1) if m}
            warnings.warn(
                f"Estimating observation probability for some pairs were impossible. Using default. "
                f"Missing users: {missing_users or 'None'}. "
                f"Missing items: {missing_items or 'None'}.",
                UserWarning
            )
            
            # Assign the default value to all impossible slots.
            logits[impossible_mask] = self.global_mean_logit

        possible_mask = ~impossible_mask
        if np.any(possible_mask):
            valid_user_vecs = self.user_factors[user_idx[possible_mask]]
            valid_item_vecs = self.item_factors[item_idx[possible_mask]]
            
            # Compute all valid logits.
            possible_logits = np.sum(valid_user_vecs * valid_item_vecs, axis=1)
            
            # Assign the results to all possible slots.
            logits[possible_mask] = possible_logits
            
        probabilities = (1 / (1 + np.exp(-logits))).tolist()
        was_impossible = impossible_mask.tolist()
        
        return probabilities, was_impossible



