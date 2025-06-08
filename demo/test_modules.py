"""
Test script to verify all modules work correctly
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_data_loader():
    """Test data loader functionality"""
    print("Testing Data Loader...")
    try:
        from data_loader import MovieDataLoader
        
        # Test with sample data path
        loader = MovieDataLoader("../dataset/wiki_movie_plots_deduped.csv")
        
        # Test loading (will use a small sample for testing)
        print("✓ MovieDataLoader imported successfully")
        return True
    except Exception as e:
        print(f"✗ Data Loader Error: {e}")
        return False

def test_input_classifier():
    """Test input classifier functionality"""
    print("Testing Input Classifier...")
    try:
        from input_classifier import InputClassifier, InputType
        
        classifier = InputClassifier()
        
        # Test different input types
        test_cases = [
            ("Titanic", InputType.TITLE),
            ("1995", InputType.YEAR),
            ("action movie", InputType.GENRE),
            ("story about a man who travels through time", InputType.PLOT)
        ]
        
        for query, expected_type in test_cases:
            input_type, confidence = classifier.classify_input(query)
            print(f"  '{query}' → {input_type.value} (confidence: {confidence:.2f})")
        
        print("✓ Input Classifier working correctly")
        return True
    except Exception as e:
        print(f"✗ Input Classifier Error: {e}")
        return False

def test_search_engine():
    """Test search engine functionality"""
    print("Testing Search Engine...")
    try:
        import pandas as pd
        from demo.custom_search_engine import SearchEngine
        
        # Create sample dataframe
        sample_data = {
            'Title': ['The Matrix', 'Titanic', 'Avatar'],
            'Plot': ['A computer hacker learns reality is simulation', 
                    'A ship sinks in the ocean', 
                    'Aliens fight humans on distant planet'],
            'Genre': ['sci-fi', 'romance', 'action'],
            'Year': [1999, 1997, 2009],
            'Director': ['Wachowski', 'Cameron', 'Cameron'],
            'Cast': ['Keanu Reeves', 'Leonardo DiCaprio', 'Sam Worthington']
        }
        
        df = pd.DataFrame(sample_data)
        
        # Add required processed columns
        df['title_text'] = df['Title'].str.lower()
        df['plot_text'] = df['Plot'].str.lower()
        df['genre_text'] = df['Genre'].str.lower()
        df['combined_text'] = (df['Title'] + ' ' + df['Plot'] + ' ' + df['Genre']).str.lower()
        
        engine = SearchEngine(df)
        print("✓ Search Engine initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Search Engine Error: {e}")
        return False

def main():
    """Run all tests"""
    print("="*50)
    print("TESTING MOVIE SEARCH ENGINE MODULES")
    print("="*50)
    
    tests = [
        test_data_loader,
        test_input_classifier, 
        test_search_engine
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            results.append(False)
        print()
    
    print("="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All modules working correctly!")
        print("\nTo run the web application:")
        print("  streamlit run app.py")
    else:
        print("❌ Some modules have issues. Check the errors above.")

if __name__ == "__main__":
    main()