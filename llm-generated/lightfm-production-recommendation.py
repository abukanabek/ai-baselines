"""
recommendation_lightfm_production.py
Production-ready pipeline with model persistence and batch recommendations.
"""

import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
import scipy.sparse as sp
import pickle
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import os

class ProductionLightFMRecommender:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.dataset = Dataset()
        
    def load_data(self, interactions_path: str, 
                 users_path: Optional[str] = None,
                 items_path: Optional[str] = None) -> None:
        """Load data from CSV files"""
        self.interactions_df = pd.read_csv(interactions_path)
        if 'rating' not in self.interactions_df.columns:
            self.interactions_df['rating'] = 1
            
        self.users_df = pd.read_csv(users_path) if users_path else None
        self.items_df = pd.read_csv(items_path) if items_path else None
        
        # Data validation
        self._validate_data()
        
    def _validate_data(self) -> None:
        """Validate input data"""
        required_cols = ['user_id', 'item_id']
        for col in required_cols:
            if col not in self.interactions_df.columns:
                raise ValueError(f"Missing required column: {col}")
                
        if self.users_df is not None and 'user_id' not in self.users_df.columns:
            raise ValueError("users.csv must contain 'user_id' column")
            
        if self.items_df is not None and 'item_id' not in self.items_df.columns:
            raise ValueError("items.csv must contain 'item_id' column")
    
    def preprocess_features(self, df: pd.DataFrame, id_col: str) -> List[str]:
        """Preprocess features for LightFM"""
        features = []
        feature_cols = [col for col in df.columns if col != id_col]
        
        for _, row in df.iterrows():
            row_features = []
            for col in feature_cols:
                value = row[col]
                if pd.isna(value):
                    continue
                    
                if isinstance(value, (int, float)):
                    # Numerical feature - discretize if needed
                    if col in ['age', 'price']:  # Example numerical columns
                        # Discretize age into buckets
                        if col == 'age':
                            bucket = (value // 10) * 10
                            row_features.append(f"age_bucket_{bucket}")
                        else:
                            row_features.append(f"{col}:{value}")
                    else:
                        row_features.append(f"{col}:{value}")
                else:
                    # Categorical feature
                    row_features.append(f"{col}_{str(value).lower().replace(' ', '_')}")
            
            features.append((row[id_col], row_features))
            
        return features
    
    def build_dataset(self) -> None:
        """Build LightFM dataset with features"""
        # Fit on all users and items
        all_users = self.interactions_df['user_id'].unique()
        all_items = self.interactions_df['item_id'].unique()
        
        self.dataset.fit(users=all_users, items=all_items)
        
        # Build interactions
        self.interactions, _ = self.dataset.build_interactions(
            [(row['user_id'], row['item_id'], row['rating']) 
             for _, row in self.interactions_df.iterrows()]
        )
        
        # Build user features
        if self.users_df is not None:
            user_features_list = self.preprocess_features(self.users_df, 'user_id')
            feature_names = set()
            for _, features in user_features_list:
                feature_names.update(features)
            
            self.dataset.fit_partial(user_features=list(feature_names))
            self.user_features = self.dataset.build_user_features(
                user_features_list, normalize=True
            )
        else:
            self.user_features = None
        
        # Build item features
        if self.items_df is not None:
            item_features_list = self.preprocess_features(self.items_df, 'item_id')
            feature_names = set()
            for _, features in item_features_list:
                feature_names.update(features)
            
            self.dataset.fit_partial(item_features=list(feature_names))
            self.item_features = self.dataset.build_item_features(
                item_features_list, normalize=True
            )
        else:
            self.item_features = None
        
        # Store mappings
        self.user_id_map, _, self.item_id_map, _ = self.dataset.mapping()
        self.user_id_map_inv = {v: k for k, v in self.user_id_map.items()}
        self.item_id_map_inv = {v: k for k, v in self.item_id_map.items()}
    
    def train(self, no_components: int = 30, learning_rate: float = 0.05,
             loss: str = 'warp', epochs: int = 50) -> None:
        """Train the recommendation model"""
        self.model = LightFM(
            no_components=no_components,
            learning_rate=learning_rate,
            loss=loss,
            random_state=42
        )
        
        self.model.fit(
            self.interactions,
            user_features=self.user_features,
            item_features=self.item_features,
            epochs=epochs,
            verbose=True
        )
        
        self.training_time = datetime.now()
    
    def batch_recommendations(self, user_ids: List[str], n: int = 10,
                            filter_interacted: bool = True) -> Dict[str, List[Tuple]]:
        """Generate recommendations for multiple users in batch"""
        recommendations = {}
        
        for user_id in user_ids:
            if user_id not in self.user_id_map:
                recommendations[user_id] = []
                continue
                
            user_internal_id = self.user_id_map[user_id]
            all_items = list(self.item_id_map.values())
            
            scores = self.model.predict(
                user_internal_id, 
                all_items,
                user_features=self.user_features,
                item_features=self.item_features
            )
            
            item_scores = list(zip(all_items, scores))
            item_scores.sort(key=lambda x: x[1], reverse=True)
            
            if filter_interacted:
                user_interactions = self.interactions[user_internal_id].indices
                item_scores = [(item, score) for item, score in item_scores 
                              if item not in user_interactions]
            
            user_recommendations = [
                (self.item_id_map_inv[item_id], score) 
                for item_id, score in item_scores[:n]
            ]
            
            recommendations[user_id] = user_recommendations
        
        return recommendations
    
    def save_model(self, version: str = "v1") -> None:
        """Save model and metadata"""
        model_path = os.path.join(self.model_dir, version)
        os.makedirs(model_path, exist_ok=True)
        
        # Save model
        with open(os.path.join(model_path, "model.pkl"), "wb") as f:
            pickle.dump(self.model, f)
        
        # Save dataset
        with open(os.path.join(model_path, "dataset.pkl"), "wb") as f:
            pickle.dump(self.dataset, f)
        
        # Save metadata
        metadata = {
            'training_time': self.training_time.isoformat(),
            'num_users': len(self.user_id_map),
            'num_items': len(self.item_id_map),
            'user_features_shape': self.user_features.shape if self.user_features is not None else None,
            'item_features_shape': self.item_features.shape if self.item_features is not None else None,
            'model_params': {
                'no_components': self.model.no_components,
                'learning_rate': self.model.learning_rate,
                'loss': self.model.loss
            }
        }
        
        with open(os.path.join(model_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, version: str = "v1") -> None:
        """Load model and metadata"""
        model_path = os.path.join(self.model_dir, version)
        
        # Load model
        with open(os.path.join(model_path, "model.pkl"), "rb") as f:
            self.model = pickle.load(f)
        
        # Load dataset
        with open(os.path.join(model_path, "dataset.pkl"), "rb") as f:
            self.dataset = pickle.load(f)
        
        # Load mappings
        self.user_id_map, _, self.item_id_map, _ = self.dataset.mapping()
        self.user_id_map_inv = {v: k for k, v in self.user_id_map.items()}
        self.item_id_map_inv = {v: k for k, v in self.item_id_map.items()}
        
        # Reconstruct feature matrices
        self.user_features = self.dataset.build_user_features([], normalize=False)
        self.item_features = self.dataset.build_item_features([], normalize=False)
        
        print(f"Model loaded from {model_path}")
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'num_users': len(self.user_id_map),
            'num_items': len(self.item_id_map),
            'user_features_available': self.user_features is not None,
            'item_features_available': self.item_features is not None,
            'model_parameters': self.model.no_components if hasattr(self, 'model') else None
        }

# Usage example
if __name__ == "__main__":
    # Initialize production recommender
    recommender = ProductionLightFMRecommender()
