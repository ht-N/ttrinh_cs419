"""
Data Loader Module for Movie Dataset
Handles loading and preprocessing of the Wikipedia movie plots dataset
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple
import streamlit as st

class MovieDataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.processed_df = None
        
    @st.cache_data
    def load_data(_self) -> pd.DataFrame:
        """Load the movie dataset from CSV file"""
        try:
            _self.df = pd.read_csv(_self.data_path)
            print(f"Loaded {len(_self.df)} movies from dataset")
            return _self.df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def preprocess_data(self) -> pd.DataFrame:
        """Preprocess the movie data for search"""
        if self.df is None:
            self.load_data()
            
        # Create a copy for processing
        self.processed_df = self.df.copy()
        
        # Handle missing values
        self.processed_df['Title'] = self.processed_df['Title'].fillna('')
        self.processed_df['Plot'] = self.processed_df['Plot'].fillna('')
        self.processed_df['Genre'] = self.processed_df['Genre'].fillna('')
        self.processed_df['Director'] = self.processed_df['Director'].fillna('')
        self.processed_df['Cast'] = self.processed_df['Cast'].fillna('')
        
        # Extract year from Release Year column
        if 'Release Year' in self.processed_df.columns:
            self.processed_df['Year'] = self.processed_df['Release Year']
        else:
            # Try to extract year from other columns if needed
            self.processed_df['Year'] = 0
            
        # Create combined text fields for search
        self.processed_df['title_text'] = self.processed_df['Title'].astype(str)
        self.processed_df['plot_text'] = self.processed_df['Plot'].astype(str)
        self.processed_df['genre_text'] = self.processed_df['Genre'].astype(str)
        self.processed_df['combined_text'] = (
            self.processed_df['Title'].astype(str) + ' ' + 
            self.processed_df['Plot'].astype(str) + ' ' + 
            self.processed_df['Genre'].astype(str) + ' ' +
            self.processed_df['Director'].astype(str) + ' ' +
            self.processed_df['Cast'].astype(str)
        )
        
        # Clean text
        for col in ['title_text', 'plot_text', 'genre_text', 'combined_text']:
            self.processed_df[col] = self.processed_df[col].apply(self._clean_text)
        
        return self.processed_df
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def get_genres(self) -> List[str]:
        """Get unique genres from the dataset"""
        if self.processed_df is None:
            self.preprocess_data()
            
        genres = set()
        for genre_str in self.processed_df['Genre'].dropna():
            # Split by common delimiters
            genre_list = re.split(r'[,;|]', str(genre_str))
            for genre in genre_list:
                genre = genre.strip().lower()
                if genre and len(genre) > 1:
                    genres.add(genre)
        
        return sorted(list(genres))
    
    def get_year_range(self) -> Tuple[int, int]:
        """Get the range of years in the dataset"""
        if self.processed_df is None:
            self.preprocess_data()
            
        years = self.processed_df['Year'].dropna()
        years = years[years > 1800]  # Filter out invalid years
        
        if len(years) > 0:
            return int(years.min()), int(years.max())
        else:
            return 1900, 2024
    
    def get_sample_movies(self, n: int = 5) -> pd.DataFrame:
        """Get a sample of movies for display"""
        if self.processed_df is None:
            self.preprocess_data()
            
        return self.processed_df.sample(min(n, len(self.processed_df)))
    
    def get_movie_stats(self) -> Dict:
        """Get basic statistics about the dataset"""
        if self.processed_df is None:
            self.preprocess_data()
            
        stats = {
            'total_movies': len(self.processed_df),
            'unique_genres': len(self.get_genres()),
            'year_range': self.get_year_range(),
            'avg_plot_length': self.processed_df['Plot'].str.len().mean() if 'Plot' in self.processed_df.columns else 0
        }
        
        return stats