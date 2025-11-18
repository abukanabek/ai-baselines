"""
recommendation_lightfm_advanced.py
Advanced pipeline with cross-validation and hyperparameter tuning.
"""

import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, auc_score, recall_at_k
from sklearn.model_selection import KFold
import scipy.sparse as sp
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class AdvancedLightFMRecommender:
    def __init__(self):
        self.dataset = Dataset()
        
    def load_and_preprocess(self, interactions_path: str, 
                          users_path: str = None, 
                          items_path: str = None) -> Tuple:
        """Load and preprocess all data"""
        # Load data
        self.interactions_df = pd.read_csv(interactions_path)
        if 'rating' not in self.interactions_df.columns:
            self.interactions_df['rating'] = 1
            
        self.users_df = pd.read_csv(users_path) if users_path else None
        self.items_df = pd.read_csv(items_path) if items_path else None
        
        # Prepare dataset
        all_users = self.interactions_df['user_id'].unique()
        all_items = self.interactions_df['item_id'].unique()
        
        self.dataset.fit(users=all_users, items=all_items)
        
        # Build interactions
        self.interactions, _ = self.dataset.build_interactions(
            [(row['user_id'], row['item_id'], row['rating']) 
             for _, row in self.interactions_df.iterrows()]
        )
        
        # Build features
        self.user_features = self._build_user_features() if self.users_df is not None else None
        self.item_features = self._build_item_features() if self.items_df is not None else None
        
        # Create mappings
        self.user_id_map = self.dataset.mapping()[0]
        self.item_id_map = self.dataset.mapping()[2]
        self.user_id_map_inv = {v: k for k, v in self.user_id_map.items()}
        self.item_id_map_inv = {v: k for k, v in self.item_id_map.items()}
        
        return self.interactions, self.user_features, self.item_features
    
    def _build_user_features(self):
        """Build user features matrix"""
        feature_cols = [col for col in self.users_df.columns if col != 'user_id']
        self.dataset.fit_partial(users=self.users_df['user_id'].unique(),
                               user_features=feature_cols)
        
        user_features_list = []
        for _, row in self.users_df.iterrows():
            features = []
            for col in feature_cols:
                value = row[col]
                if pd.isna(value):
                    continue
                if isinstance(value, (int, float)):
                    features.append(f"{col}:{value}")
                else:
                    features.append(f"{col}_{str(value)}")
            user_features_list.append((row['user_id'], features))
        
        return self.dataset.build_user_features(user_features_list, normalize=False)
    
    def _build_item_features(self):
        """Build item features matrix"""
        feature_cols = [col for col in self.items_df.columns if col != 'item_id']
        self.dataset.fit_partial(items=self.items_df['item_id'].unique(),
                               item_features=feature_cols)
        
        item_features_list = []
        for _, row in self.items_df.iterrows():
            features = []
            for col in feature_cols:
                value = row[col]
                if pd.isna(value):
                    continue
                if isinstance(value, (int, float)):
                    features.append(f"{col}:{value}")
                else:
                    features.append(f"{col}_{str(value)}")
            item_features_list.append((row['item_id'], features))
        
        return self.dataset.build_item_features(item_features_list, normalize=False)
    
    def cross_validate(self, n_splits: int = 5, 
                      param_grid: Dict = None,
                      k: int = 10) -> pd.DataFrame:
        """Perform cross-validation with hyperparameter tuning"""
        if param_grid is None:
            param_grid = {
                'no_components': [20, 30, 50],
                'learning_rate': [0.01, 0.05, 0.1],
                'loss': ['warp', 'bpr', 'logistic']
            }
        
        results = []
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        interactions_coo = self.interactions.tocoo()
        
        # Convert to user-item pairs for splitting
        user_item_pairs = list(zip(interactions_coo.row, interactions_coo.col))
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(user_item_pairs)):
            print(f"Fold {fold + 1}/{n_splits}")
            
            # Create train/test matrices
            train_pairs = [user_item_pairs[i] for i in train_idx]
            test_pairs = [user_item_pairs[i] for i in test_idx]
            
            train_data = np.ones(len(train_pairs))
            train_rows, train_cols = zip(*train_pairs)
            train_interactions = sp.coo_matrix(
                (train_data, (train_rows, train_cols)),
                shape=self.interactions.shape
            ).tocsr()
            
            test_data = np.ones(len(test_pairs))
            test_rows, test_cols = zip(*test_pairs)
            test_interactions = sp.coo_matrix(
                (test_data, (test_rows, test_cols)),
                shape=self.interactions.shape
            ).tocsr()
            
            # Grid search
            for components in param_grid['no_components']:
                for lr in param_grid['learning_rate']:
                    for loss in param_grid['loss']:
                        model = LightFM(
                            no_components=components,
                            learning_rate=lr,
                            loss=loss,
                            random_state=42
                        )
                        
                        model.fit(
                            train_interactions,
                            user_features=self.user_features,
                            item_features=self.item_features,
                            epochs=20,
                            verbose=False
                        )
                        
                        # Evaluate
                        train_precision = precision_at_k(
                            model, train_interactions, k=k,
                            user_features=self.user_features,
                            item_features=self.item_features
                        ).mean()
                        
                        test_precision = precision_at_k(
                            model, test_interactions, k=k,
                            user_features=self.user_features,
                            item_features=self.item_features
                        ).mean()
                        
                        test_auc = auc_score(
                            model, test_interactions,
                            user_features=self.user_features,
                            item_features=self.item_features
                        ).mean()
                        
                        results.append({
                            'fold': fold,
                            'no_components': components,
                            'learning_rate': lr,
                            'loss': loss,
                            'train_precision': train_precision,
                            'test_precision': test_precision,
                            'test_auc': test_auc
                        })
        
        results_df = pd.DataFrame(results)
        return results_df
    
    def train_final_model(self, best_params: Dict, epochs: int = 50):
        """Train final model with best parameters"""
        self.model = LightFM(
            no_components=best_params['no_components'],
            learning_rate=best_params['learning_rate'],
            loss=best_params['loss'],
            random_state=42
        )
        
        self.model.fit(
            self.interactions,
            user_features=self.user_features,
            item_features=self.item_features,
            epochs=epochs,
            verbose=True
        )
    
    def get_similar_items(self, item_id: str, n: int = 10) -> List[Tuple]:
        """Find similar items using item embeddings"""
        if item_id not in self.item_id_map:
            return []
            
        item_internal_id = self.item_id_map[item_id]
        item_biases, item_embeddings = self.model.get_item_representations(
            features=self.item_features
        )
        
        # Calculate cosine similarities
        target_embedding = item_embeddings[item_internal_id]
        similarities = item_embeddings.dot(target_embedding)
        
        # Get top N similar items (excluding itself)
        similar_indices = np.argsort(similarities)[::-1][1:n+1]
        
        similar_items = []
        for idx in similar_indices:
            original_id = self.item_id_map_inv[idx]
            similarity = similarities[idx]
            similar_items.append((original_id, similarity))
        
        return similar_items
    
    def recommend_for_user(self, user_id: str, n: int = 10, 
                         filter_interacted: bool = True) -> List[Tuple]:
        """Generate recommendations for user"""
        if user_id not in self.user_id_map:
            return []
            
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
        
        recommendations = [
            (self.item_id_map_inv[item_id], score) 
            for item_id, score in item_scores[:n]
        ]
        
        return recommendations

# Usage example
if __name__ == "__main__":
    # Initialize advanced recommender
    recommender = AdvancedLightFMRecommender()
    
    # Load and preprocess data
    interactions, user_features, item_features = recommender.load_and_preprocess(
        "data/interactions.csv",
        "data/users.csv",
        "data/items.csv"
    )
    
    # Cross-validation
    print("Performing cross-validation...")
    cv_results = recommender.cross_validate(n_splits=3)
    
    # Find best parameters
    best_by_precision = cv_results.loc[cv_results.groupby(['no_components', 'learning_rate', 'loss'])['test_precision'].idxmax()]
    best_params = best_by_precision.sort_values('test_precision', ascending=False).iloc[0]
    
    print(f"\nBest parameters: {best_params[['no_components', 'learning_rate', 'loss']].to_dict()}")
    
    # Train final model
    recommender.train_final_model(best_params, epochs=50)
    
    # Generate recommendations
    sample_user = recommender.interactions_df['user_id'].iloc[0]
    recommendations = recommender.recommend_for_user(sample_user, n=5)
    print(f"\nTop 5 recommendations for user {sample_user}:")
    for item, score in recommendations:
        print(f"  Item: {item}, Score: {score:.4f}")
    
    # Get similar items
    sample_item = recommender.interactions_df['item_id'].iloc[0]
    similar_items = recommender.get_similar_items(sample_item, n=3)
    print(f"\nTop 3 similar items to {sample_item}:")
    for item, similarity in similar_items:
        print(f"  Item: {item}, Similarity: {similarity:.4f}")