import pandas as pd
import numpy as np
from surprise import Dataset, Reader, KNNBasic, SVD
from surprise.model_selection import train_test_split
from collections import defaultdict

class MovieRecommender:
    def __init__(self):
        # Initialize data structures
        self.movies_df = None
        self.ratings_df = None
        self.algo = None
        self.trainset = None
        
    def load_data(self, movies_path, ratings_path):
        """Load movie and rating data"""
        self.movies_df = pd.read_csv(movies_path)
        self.ratings_df = pd.read_csv(ratings_path)
        
        # Merge data for easier access
        self.movie_ratings = pd.merge(self.movies_df, self.ratings_df, on='movieId')
        
    def prepare_surprise_data(self):
        """Prepare data for Surprise library"""
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.ratings_df[['userId', 'movieId', 'rating']], reader)
        return data
    
    def train_model(self, algorithm='svd'):
        """Train the recommendation model"""
        data = self.prepare_surprise_data()
        self.trainset = data.build_full_trainset()
        
        if algorithm == 'svd':
            self.algo = SVD()
        elif algorithm == 'knn':
            self.algo = KNNBasic()
        else:
            raise ValueError("Unsupported algorithm")
            
        self.algo.fit(self.trainset)
    
    def get_top_n_recommendations(self, user_id, n=10):
        """Get top N recommendations for a user"""
        if not self.algo:
            raise Exception("Model not trained. Call train_model() first.")
            
        # Get list of all movie IDs
        all_movie_ids = self.movies_df['movieId'].unique()
        
        # Get list of movies the user has already rated
        rated_movies = self.ratings_df[self.ratings_df['userId'] == user_id]['movieId']
        
        # Get predictions for movies not rated by the user
        predictions = []
        for movie_id in all_movie_ids:
            if movie_id not in rated_movies.values:
                pred = self.algo.predict(user_id, movie_id)
                predictions.append((movie_id, pred.est))
        
       