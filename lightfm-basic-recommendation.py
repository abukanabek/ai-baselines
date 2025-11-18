"""
recommendation_lightfm_basic.py
Basic LightFM pipeline using only user-item interactions.
"""

import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, auc_score
from sklearn.model_selection import train_test_split
import scipy.sparse as sp

class BasicLightFMRecommender:
    def __init__(self, no_components=30, learning_rate=0.05, loss='warp', random_state=42):
        self.dataset = Dataset()
        self.model = LightFM(
            no_components=no_components,
            learning_rate=learning_rate,
            loss=loss,
            random_state=random_state
        )
        self.interactions = None
        self.item_id_map = None
        self.user_id_map = None
        
    def load_data(self, interactions_path):
        """Load interactions CSV with columns: user_id, item_id, rating (optional)"""
        interactions_df = pd.read_csv(interactions_path)
        
        # If no rating column, create binary interactions
        if 'rating' not in interactions_df.columns:
            interactions_df['rating'] = 1
            
        self.interactions_df = interactions_df
        return interactions_df
    
    def prepare_interactions(self):
        """Fit dataset and create interactions matrix"""
        # Fit on user and item IDs
        self.dataset.fit(
            self.interactions_df['user_id'].unique(),
            self.interactions_df['item_id'].unique()
        )
        
        # Build interactions matrix
        self.interactions, _ = self.dataset.build_interactions(
            [(row['user_id'], row['item_id'], row['rating']) 
             for _, row in self.interactions_df.iterrows()]
        )
        
        self.user_id_map = self.dataset.mapping()[0]
        self.item_id_map = self.dataset.mapping()[2]
        self.user_id_map_inv = {v: k for k, v in self.user_id_map.items()}
        self.item_id_map_inv = {v: k for k, v in self.item_id_map.items()}
        
        return self.interactions
    
    def train_test_split(self, test_ratio=0.2):
        """Split interactions into train and test sets"""
        if self.interactions is None:
            self.prepare_interactions()
            
        # Convert to COO format for splitting
        interactions_coo = self.interactions.tocoo()
        
        # Generate train/test indices
        indices = np.random.permutation(interactions_coo.data.shape[0])
        test_size = int(len(indices) * test_ratio)
        
        train_indices = indices[test_size:]
        test_indices = indices[:test_size]
        
        # Create train matrix
        train_data = interactions_coo.data[train_indices]
        train_row = interactions_coo.row[train_indices]
        train_col = interactions_coo.col[train_indices]
        self.train_interactions = sp.coo_matrix(
            (train_data, (train_row, train_col)),
            shape=self.interactions.shape
        ).tocsr()
        
        # Create test matrix
        test_data = interactions_coo.data[test_indices]
        test_row = interactions_coo.row[test_indices]
        test_col = interactions_coo.col[test_indices]
        self.test_interactions = sp.coo_matrix(
            (test_data, (test_row, test_col)),
            shape=self.interactions.shape
        ).tocsr()
        
        return self.train_interactions, self.test_interactions
    
    def train(self, epochs=20, verbose=True):
        """Train the LightFM model"""
        if not hasattr(self, 'train_interactions'):
            self.train_test_split()
            
        self.model.fit(
            self.train_interactions,
            epochs=epochs,
            verbose=verbose
        )
    
    def evaluate(self, k=10):
        """Evaluate model performance"""
        train_precision = precision_at_k(
            self.model, self.train_interactions, k=k
        ).mean()
        test_precision = precision_at_k(
            self.model, self.test_interactions, k=k
        ).mean()
        
        train_auc = auc_score(self.model, self.train_interactions).mean()
        test_auc = auc_score(self.model, self.test_interactions).mean()
        
        print(f"Train Precision@{k}: {train_precision:.4f}")
        print(f"Test Precision@{k}: {test_precision:.4f}")
        print(f"Train AUC: {train_auc:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        
        return {
            'train_precision': train_precision,
            'test_precision': test_precision,
            'train_auc': train_auc,
            'test_auc': test_auc
        }
    
    def recommend_for_user(self, user_id, n=10, filter_interacted=True):
        """Generate recommendations for a specific user"""
        if user_id not in self.user_id_map:
            return []
            
        user_internal_id = self.user_id_map[user_id]
        
        # Get all items
        all_items = list(self.item_id_map.values())
        
        # Calculate scores
        scores = self.model.predict(user_internal_id, all_items)
        
        # Create item-score pairs
        item_scores = list(zip(all_items, scores))
        
        # Sort by score descending
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Filter out already interacted items if requested
        if filter_interacted:
            user_interactions = self.interactions[user_internal_id].indices
            item_scores = [(item, score) for item, score in item_scores 
                          if item not in user_interactions]
        
        # Get top N recommendations
        top_items = item_scores[:n]
        
        # Convert back to original IDs
        recommendations = [
            (self.item_id_map_inv[item_id], score) 
            for item_id, score in top_items
        ]
        
        return recommendations

# Usage example
if __name__ == "__main__":
    # Initialize recommender
    recommender = BasicLightFMRecommender()
    
    # Load data (assuming CSV with columns: user_id, item_id, rating)
    interactions_df = recommender.load_data("data/interactions.csv")
    print(f"Loaded {len(interactions_df)} interactions")
    
    # Prepare data
    recommender.prepare_interactions()
    
    # Train model
    recommender.train(epochs=20)
    
    # Evaluate
    metrics = recommender.evaluate()
    
    # Generate recommendations for a user
    sample_user = interactions_df['user_id'].iloc[0]
    recommendations = recommender.recommend_for_user(sample_user, n=5)
    print(f"\nTop 5 recommendations for user {sample_user}:")
    for item, score in recommendations:
        print(f"  Item: {item}, Score: {score:.4f}")