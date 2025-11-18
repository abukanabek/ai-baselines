"""
recommendation_lightfm_hybrid.py
Hybrid LightFM pipeline using interactions + user metadata + item metadata.
"""

import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, auc_score
import scipy.sparse as sp

class HybridLightFMRecommender:
    def __init__(self, no_components=30, learning_rate=0.05, loss='warp', random_state=42):
        self.dataset = Dataset()
        self.model = LightFM(
            no_components=no_components,
            learning_rate=learning_rate,
            loss=loss,
            random_state=random_state
        )
        
    def load_data(self, interactions_path, users_path=None, items_path=None):
        """Load all data sources"""
        # Load interactions
        self.interactions_df = pd.read_csv(interactions_path)
        if 'rating' not in self.interactions_df.columns:
            self.interactions_df['rating'] = 1
            
        # Load user metadata
        if users_path:
            self.users_df = pd.read_csv(users_path)
        else:
            self.users_df = None
            
        # Load item metadata
        if items_path:
            self.items_df = pd.read_csv(items_path)
        else:
            self.items_df = None
            
        return self.interactions_df, self.users_df, self.items_df
    
    def prepare_features(self):
        """Prepare interactions and features"""
        # Get all user and item IDs
        all_users = self.interactions_df['user_id'].unique()
        all_items = self.interactions_df['item_id'].unique()
        
        # Fit dataset with users and items
        self.dataset.fit(users=all_users, items=all_items)
        
        # Build interactions
        self.interactions, _ = self.dataset.build_interactions(
            [(row['user_id'], row['item_id'], row['rating']) 
             for _, row in self.interactions_df.iterrows()]
        )
        
        # Build user features if available
        if self.users_df is not None:
            self.dataset.fit_partial(users=all_users,
                                   user_features=self._get_feature_columns(self.users_df))
            user_features_list = []
            for _, user_row in self.users_df.iterrows():
                user_id = user_row['user_id']
                features = self._get_user_features(user_row)
                user_features_list.append((user_id, features))
            
            self.user_features = self.dataset.build_user_features(
                user_features_list, normalize=False
            )
        else:
            self.user_features = None
        
        # Build item features if available
        if self.items_df is not None:
            self.dataset.fit_partial(items=all_items,
                                   item_features=self._get_feature_columns(self.items_df))
            item_features_list = []
            for _, item_row in self.items_df.iterrows():
                item_id = item_row['item_id']
                features = self._get_item_features(item_row)
                item_features_list.append((item_id, features))
            
            self.item_features = self.dataset.build_item_features(
                item_features_list, normalize=False
            )
        else:
            self.item_features = None
        
        # Create mappings
        self.user_id_map = self.dataset.mapping()[0]
        self.item_id_map = self.dataset.mapping()[2]
        self.user_id_map_inv = {v: k for k, v in self.user_id_map.items()}
        self.item_id_map_inv = {v: k for k, v in self.item_id_map.items()}
        
        return self.interactions, self.user_features, self.item_features
    
    def _get_feature_columns(self, df):
        """Get feature columns excluding ID column"""
        return [col for col in df.columns if col != 'user_id' and col != 'item_id']
    
    def _get_user_features(self, user_row):
        """Extract user features from row"""
        features = []
        feature_cols = self._get_feature_columns(self.users_df)
        
        for col in feature_cols:
            value = user_row[col]
            if pd.isna(value):
                continue
                
            if isinstance(value, (int, float)):
                # Numerical feature
                features.append(f"{col}:{value}")
            else:
                # Categorical feature
                features.append(f"{col}_{str(value)}")
                
        return features
    
    def _get_item_features(self, item_row):
        """Extract item features from row"""
        features = []
        feature_cols = self._get_feature_columns(self.items_df)
        
        for col in feature_cols:
            value = item_row[col]
            if pd.isna(value):
                continue
                
            if isinstance(value, (int, float)):
                # Numerical feature
                features.append(f"{col}:{value}")
            else:
                # Categorical feature
                features.append(f"{col}_{str(value)}")
                
        return features
    
    def train_test_split(self, test_ratio=0.2):
        """Split data into train and test sets"""
        interactions_coo = self.interactions.tocoo()
        indices = np.random.permutation(interactions_coo.data.shape[0])
        test_size = int(len(indices) * test_ratio)
        
        train_indices = indices[test_size:]
        test_indices = indices[:test_size]
        
        # Train interactions
        train_data = interactions_coo.data[train_indices]
        train_row = interactions_coo.row[train_indices]
        train_col = interactions_coo.col[train_indices]
        self.train_interactions = sp.coo_matrix(
            (train_data, (train_row, train_col)),
            shape=self.interactions.shape
        ).tocsr()
        
        # Test interactions
        test_data = interactions_coo.data[test_indices]
        test_row = interactions_coo.row[test_indices]
        test_col = interactions_coo.col[test_indices]
        self.test_interactions = sp.coo_matrix(
            (test_data, (test_row, test_col)),
            shape=self.interactions.shape
        ).tocsr()
        
        return self.train_interactions, self.test_interactions
    
    def train(self, epochs=20, verbose=True):
        """Train the model with optional features"""
        if not hasattr(self, 'train_interactions'):
            self.train_test_split()
            
        self.model.fit(
            self.train_interactions,
            user_features=self.user_features,
            item_features=self.item_features,
            epochs=epochs,
            verbose=verbose
        )
    
    def evaluate(self, k=10):
        """Evaluate model performance"""
        train_precision = precision_at_k(
            self.model, self.train_interactions, 
            user_features=self.user_features,
            item_features=self.item_features,
            k=k
        ).mean()
        
        test_precision = precision_at_k(
            self.model, self.test_interactions,
            user_features=self.user_features,
            item_features=self.item_features,
            k=k
        ).mean()
        
        print(f"Train Precision@{k}: {train_precision:.4f}")
        print(f"Test Precision@{k}: {test_precision:.4f}")
        
        return {'train_precision': train_precision, 'test_precision': test_precision}
    
    def recommend_for_user(self, user_id, n=10, filter_interacted=True):
        """Generate recommendations for a user"""
        if user_id not in self.user_id_map:
            return []
            
        user_internal_id = self.user_id_map[user_id]
        all_items = list(self.item_id_map.values())
        
        # Get user features if available
        user_feats = None
        if self.user_features is not None:
            user_feats = self.user_features
        
        scores = self.model.predict(
            user_internal_id, 
            all_items,
            user_features=user_feats,
            item_features=self.item_features
        )
        
        item_scores = list(zip(all_items, scores))
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        if filter_interacted:
            user_interactions = self.interactions[user_internal_id].indices
            item_scores = [(item, score) for item, score in item_scores 
                          if item not in user_interactions]
        
        top_items = item_scores[:n]
        recommendations = [
            (self.item_id_map_inv[item_id], score) 
            for item_id, score in top_items
        ]
        
        return recommendations

# Usage example
if __name__ == "__main__":
    # Initialize hybrid recommender
    recommender = HybridLightFMRecommender()
    
    # Load all data
    interactions, users, items = recommender.load_data(
        "data/interactions.csv",
        "data/users.csv", 
        "data/items.csv"
    )
    
    # Prepare features
    recommender.prepare_features()
    
    # Train model
    recommender.train(epochs=20)
    
    # Evaluate
    metrics = recommender.evaluate()
    
    # Generate recommendations
    sample_user = interactions['user_id'].iloc[0]
    recommendations = recommender.recommend_for_user(sample_user, n=5)
    print(f"\nTop 5 recommendations for user {sample_user}:")
    for item, score in recommendations:
        print(f"  Item: {item}, Score: {score:.4f}")
