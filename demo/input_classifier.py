"""
Input Classifier Module
Classifies user input to determine search strategy (title, plot, year, genre)
"""

import re
from typing import Dict, List, Tuple
from enum import Enum

class InputType(Enum):
    TITLE = "title"
    PLOT = "plot"
    YEAR = "year"
    GENRE = "genre"
    GENERAL = "general"

class InputClassifier:
    def __init__(self, known_genres: List[str] = None):
        self.known_genres = [g.lower() for g in known_genres] if known_genres else []
        
        # Common genre keywords
        self.genre_keywords = [
            'action', 'adventure', 'animation', 'comedy', 'crime', 'documentary',
            'drama', 'family', 'fantasy', 'horror', 'musical', 'mystery',
            'romance', 'sci-fi', 'science fiction', 'thriller', 'war', 'western',
            'biography', 'history', 'sport', 'music', 'film-noir'
        ]
        
        # Extend with known genres
        if self.known_genres:
            self.genre_keywords.extend(self.known_genres)
        
        self.genre_keywords = list(set(self.genre_keywords))  # Remove duplicates
    
    def classify_input(self, user_input: str) -> Tuple[InputType, float]:
        """
        Classify user input and return the most likely type with confidence score
        
        Args:
            user_input: User's search query
            
        Returns:
            Tuple of (InputType, confidence_score)
        """
        user_input = user_input.strip().lower()
        
        if not user_input:
            return InputType.GENERAL, 0.0
        
        # Check for year
        year_result = self._check_year(user_input)
        if year_result[1] > 0.8:
            return year_result
        
        # Check for genre
        genre_result = self._check_genre(user_input)
        if genre_result[1] > 0.7:
            return genre_result
        
        # Check for title vs plot
        title_vs_plot = self._classify_title_vs_plot(user_input)
        
        # Return the result with highest confidence
        results = [year_result, genre_result, title_vs_plot]
        best_result = max(results, key=lambda x: x[1])
        
        return best_result
    
    def _check_year(self, text: str) -> Tuple[InputType, float]:
        """Check if input is a year"""
        # Look for 4-digit years
        year_pattern = r'\b(19[0-9]{2}|20[0-2][0-9])\b'
        years = re.findall(year_pattern, text)
        
        if years:
            # If the text is just a year or year range
            if len(text.split()) <= 3 and any(year in text for year in years):
                return InputType.YEAR, 0.9
            else:
                # Year mentioned but not the main focus
                return InputType.YEAR, 0.3
        
        # Check for year-related keywords
        year_keywords = ['year', 'released', 'made in', 'from']
        if any(keyword in text for keyword in year_keywords):
            return InputType.YEAR, 0.5
        
        return InputType.YEAR, 0.0
    
    def _check_genre(self, text: str) -> Tuple[InputType, float]:
        """Check if input is about genre"""
        words = text.split()
        
        # Direct genre match
        for genre in self.genre_keywords:
            if genre in text:
                # Higher confidence if it's the main part of the query
                if len(words) <= 3:
                    return InputType.GENRE, 0.9
                else:
                    return InputType.GENRE, 0.6
        
        # Genre-related keywords
        genre_indicators = ['genre', 'type', 'kind of movie', 'category']
        if any(indicator in text for indicator in genre_indicators):
            return InputType.GENRE, 0.7
        
        return InputType.GENRE, 0.0
    
    def _classify_title_vs_plot(self, text: str) -> Tuple[InputType, float]:
        """Classify between title and plot search"""
        words = text.split()
        word_count = len(words)
        
        # Very short queries are likely titles
        if word_count <= 3:
            return InputType.TITLE, 0.8
        
        # Medium length could be either
        elif word_count <= 6:
            # Check for plot indicators
            plot_indicators = [
                'about', 'story', 'plot', 'character', 'man', 'woman', 'person',
                'who', 'where', 'when', 'how', 'what happens', 'involves',
                'follows', 'tells', 'depicts', 'shows', 'features'
            ]
            
            if any(indicator in text for indicator in plot_indicators):
                return InputType.PLOT, 0.7
            else:
                return InputType.TITLE, 0.6
        
        # Long queries are likely plot descriptions
        else:
            return InputType.PLOT, 0.8
    
    def get_search_weights(self, input_type: InputType) -> Dict[str, float]:
        """
        Get search weights for different fields based on input type
        
        Returns:
            Dictionary with weights for title, plot, genre, combined fields
        """
        weights = {
            InputType.TITLE: {
                'title_text': 1.0,
                'plot_text': 0.2,
                'genre_text': 0.1,
                'combined_text': 0.3
            },
            InputType.PLOT: {
                'title_text': 0.3,
                'plot_text': 1.0,
                'genre_text': 0.2,
                'combined_text': 0.5
            },
            InputType.GENRE: {
                'title_text': 0.1,
                'plot_text': 0.3,
                'genre_text': 1.0,
                'combined_text': 0.4
            },
            InputType.YEAR: {
                'title_text': 0.2,
                'plot_text': 0.2,
                'genre_text': 0.1,
                'combined_text': 0.3
            },
            InputType.GENERAL: {
                'title_text': 0.5,
                'plot_text': 0.7,
                'genre_text': 0.3,
                'combined_text': 1.0
            }
        }
        
        return weights.get(input_type, weights[InputType.GENERAL])
    
    def explain_classification(self, user_input: str) -> str:
        """Provide explanation for the classification"""
        input_type, confidence = self.classify_input(user_input)
        
        explanations = {
            InputType.TITLE: f"Detected as movie title search (confidence: {confidence:.1%}). Focusing on title matching.",
            InputType.PLOT: f"Detected as plot/story search (confidence: {confidence:.1%}). Focusing on plot descriptions.",
            InputType.GENRE: f"Detected as genre search (confidence: {confidence:.1%}). Focusing on movie genres.",
            InputType.YEAR: f"Detected as year-based search (confidence: {confidence:.1%}). Focusing on release years.",
            InputType.GENERAL: f"General search detected (confidence: {confidence:.1%}). Searching across all fields."
        }
        
        return explanations.get(input_type, "Unknown classification")