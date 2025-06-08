"""
Search Engine Module
Implements BM25 and VSM (Vector Space Model) for movie retrieval
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from typing import List, Dict, Tuple, Any
import streamlit as st
import re

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class CustomSearchEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # BM25 models for different fields
        self.bm25_models = {}
        self.processed_corpus = {}
        
        # VSM models
        self.tfidf_vectorizers = {}
        self.tfidf_matrices = {}
        
        # Initialize models
        self._initialize_models()
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for BM25"""
        if pd.isna(text):
            return []
        
        # Convert to lowercase and tokenize
        text = str(text).lower()
        tokens = word_tokenize(text)
        
        # Remove punctuation, numbers, and short words
        tokens = [token for token in tokens if token.isalpha() and len(token) > 2]
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Stem words
        tokens = [self.stemmer.stem(token) for token in tokens]
        
        return tokens
    
    def _preprocess_for_tfidf(self, text: str) -> str:
        """Preprocess text for TF-IDF (VSM)"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    @st.cache_data
    def _initialize_models(_self):
        """Initialize BM25 and TF-IDF models for different fields"""
        fields = ['title_text', 'plot_text', 'genre_text', 'combined_text']
        
        for field in fields:
            if field in _self.df.columns:
                # Prepare corpus for BM25
                corpus = []
                for text in _self.df[field]:
                    processed_tokens = _self._preprocess_text(text)
                    corpus.append(processed_tokens)
                
                _self.processed_corpus[field] = corpus
                _self.bm25_models[field] = BM25Okapi(corpus)
                
                # Prepare TF-IDF for VSM
                processed_texts = [_self._preprocess_for_tfidf(text) for text in _self.df[field]]
                
                vectorizer = TfidfVectorizer(
                    max_features=5000,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.8
                )
                
                tfidf_matrix = vectorizer.fit_transform(processed_texts)
                _self.tfidf_vectorizers[field] = vectorizer
                _self.tfidf_matrices[field] = tfidf_matrix
        
        print("Search models initialized successfully!")
    
    def search_bm25(self, query: str, field: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search using BM25"""
        if field not in self.bm25_models:
            return []
        
        # Preprocess query
        query_tokens = self._preprocess_text(query)
        if not query_tokens:
            return []
        
        # Get BM25 scores
        scores = self.bm25_models[field].get_scores(query_tokens)
        
        # Get top k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(idx, scores[idx]) for idx in top_indices if scores[idx] > 0]
        
        return results
    
    def search_vsm(self, query: str, field: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search using Vector Space Model (TF-IDF + Cosine Similarity)"""
        if field not in self.tfidf_vectorizers:
            return []
        
        # Preprocess and vectorize query
        processed_query = self._preprocess_for_tfidf(query)
        query_vector = self.tfidf_vectorizers[field].transform([processed_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrices[field]).flatten()
        
        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(idx, similarities[idx]) for idx in top_indices if similarities[idx] > 0]
        
        return results
    
    def hybrid_search(self, query: str, search_weights: Dict[str, float], 
                     method: str = "bm25", top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform hybrid search across multiple fields with different weights
        
        Args:
            query: Search query
            search_weights: Weights for different fields
            method: "bm25", "vsm", or "hybrid"
            top_k: Number of results to return
        """
        if method == "hybrid":
            # Combine BM25 and VSM results
            return self._hybrid_bm25_vsm(query, search_weights, top_k)
        
        all_scores = {}
        
        for field, weight in search_weights.items():
            if weight > 0:
                if method == "bm25":
                    results = self.search_bm25(query, field, top_k * 2)
                else:  # vsm
                    results = self.search_vsm(query, field, top_k * 2)
                
                for idx, score in results:
                    if idx not in all_scores:
                        all_scores[idx] = 0
                    all_scores[idx] += score * weight
        
        # Sort by combined score
        sorted_results = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Format results
        formatted_results = []
        for idx, score in sorted_results:
            movie_data = self.df.iloc[idx].to_dict()
            movie_data['search_score'] = score
            movie_data['index'] = idx
            formatted_results.append(movie_data)
        
        return formatted_results
    
    def _hybrid_bm25_vsm(self, query: str, search_weights: Dict[str, float], 
                        top_k: int = 10, bm25_weight: float = 0.6) -> List[Dict[str, Any]]:
        """Combine BM25 and VSM results"""
        bm25_results = {}
        vsm_results = {}
        
        # Get BM25 results
        for field, weight in search_weights.items():
            if weight > 0:
                results = self.search_bm25(query, field, top_k * 2)
                for idx, score in results:
                    if idx not in bm25_results:
                        bm25_results[idx] = 0
                    bm25_results[idx] += score * weight
        
        # Get VSM results
        for field, weight in search_weights.items():
            if weight > 0:
                results = self.search_vsm(query, field, top_k * 2)
                for idx, score in results:
                    if idx not in vsm_results:
                        vsm_results[idx] = 0
                    vsm_results[idx] += score * weight
        
        # Normalize scores
        if bm25_results:
            max_bm25 = max(bm25_results.values())
            bm25_results = {k: v/max_bm25 for k, v in bm25_results.items()}
        
        if vsm_results:
            max_vsm = max(vsm_results.values())
            vsm_results = {k: v/max_vsm for k, v in vsm_results.items()}
        
        # Combine scores
        all_indices = set(bm25_results.keys()) | set(vsm_results.keys())
        combined_scores = {}
        
        for idx in all_indices:
            bm25_score = bm25_results.get(idx, 0)
            vsm_score = vsm_results.get(idx, 0)
            combined_scores[idx] = bm25_score * bm25_weight + vsm_score * (1 - bm25_weight)
        
        # Sort and format results
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        formatted_results = []
        for idx, score in sorted_results:
            movie_data = self.df.iloc[idx].to_dict()
            movie_data['search_score'] = score
            movie_data['index'] = idx
            movie_data['bm25_score'] = bm25_results.get(idx, 0)
            movie_data['vsm_score'] = vsm_results.get(idx, 0)
            formatted_results.append(movie_data)
        
        return formatted_results
    
    def search_by_year(self, year: int, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search movies by specific year"""
        if 'Year' not in self.df.columns:
            return []
        
        # Filter by year
        year_matches = self.df[self.df['Year'] == year]
        
        if len(year_matches) == 0:
            # Try approximate year search (±2 years)
            year_matches = self.df[
                (self.df['Year'] >= year - 2) & 
                (self.df['Year'] <= year + 2)
            ]
        
        # Limit results and convert to list of dicts
        results = year_matches.head(top_k)
        formatted_results = []
        
        for idx, row in results.iterrows():
            movie_data = row.to_dict()
            movie_data['search_score'] = 1.0  # Perfect match for year
            movie_data['index'] = idx
            formatted_results.append(movie_data)
        
        return formatted_results