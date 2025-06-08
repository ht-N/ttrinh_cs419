# Movie Search Engine

A Streamlit web application that uses BM25 and Vector Space Model (VSM) for intelligent movie retrieval from the Wikipedia Movie Plots dataset.

## Features

- **Multiple Search Methods**: BM25, VSM (TF-IDF), and Hybrid approaches
- **Intelligent Input Classification**: Automatically detects whether user is searching by title, plot, genre, or year
- **Multi-field Search**: Searches across movie titles, plots, genres, directors, and cast
- **Interactive Web Interface**: Built with Streamlit for easy use
- **Advanced Search Options**: Configurable parameters and manual overrides
- **Visual Results**: Charts and cards displaying search results with scores

## Architecture

The application is built with modular design:

### Modules

1. **`data_loader.py`**: Handles loading and preprocessing of the movie dataset
2. **`input_classifier.py`**: Classifies user input to determine search strategy
3. **`search_engine.py`**: Implements BM25 and VSM search algorithms
4. **`app.py`**: Main Streamlit application interface

### Input Classification System

The system automatically detects input type using the following logic:

#### Hybrid Approach (Implemented)
- **Year Detection**: 4-digit numbers (1900-2025) → Release year search
- **Genre Detection**: Keywords matching known genres → Genre-focused search  
- **Title vs Plot**: 
  - Short queries (1-5 words) → Title search
  - Long queries (6+ words) → Plot description search
  - Medium queries → Analyzed for plot indicators
- **Manual Override**: Users can override auto-detection

#### Search Weight Distribution by Input Type:

| Input Type | Title Weight | Plot Weight | Genre Weight | Combined Weight |
|------------|--------------|-------------|--------------|-----------------|
| Title      | 1.0          | 0.2         | 0.1          | 0.3             |
| Plot       | 0.3          | 1.0         | 0.2          | 0.5             |
| Genre      | 0.1          | 0.3         | 1.0          | 0.4             |
| Year       | 0.2          | 0.2         | 0.1          | 0.3             |
| General    | 0.5          | 0.7         | 0.3          | 1.0             |

## Installation

1. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download NLTK Data** (automatic on first run):
   - punkt tokenizer
   - stopwords corpus

## Usage

1. **Start the Application**:
   ```bash
   streamlit run app.py
   ```

2. **Access the Web Interface**:
   - Open your browser to `http://localhost:8501`

3. **Search for Movies**:
   - Enter keywords in the search box
   - Choose search method (BM25, VSM, or Hybrid)
   - Configure number of results (1-50)
   - View results with scores and explanations

## Search Examples

### By Title
- Input: `"Titanic"`
- Classification: Title search
- Focus: Movie title matching

### By Plot
- Input: `"story about a man who can travel through time"`
- Classification: Plot search  
- Focus: Plot description matching

### By Genre
- Input: `"action movie"`
- Classification: Genre search
- Focus: Genre classification

### By Year
- Input: `"1995"`
- Classification: Year search
- Focus: Release year filtering

### General Search
- Input: `"romantic comedy with Tom Hanks"`
- Classification: General search
- Focus: Multi-field search across all attributes

## Technical Details

### BM25 Implementation
- **Parameters**: k1=1.5, b=0.75 (tunable)
- **Preprocessing**: Tokenization, stopword removal, stemming
- **Scoring**: Probabilistic ranking function with length normalization

### VSM Implementation  
- **Vectorization**: TF-IDF with n-grams (1,2)
- **Similarity**: Cosine similarity
- **Features**: Up to 5000 features, min_df=2, max_df=0.8

### Hybrid Search
- **Combination**: Weighted average of BM25 and VSM scores
- **Normalization**: Score normalization before combination
- **Default Weight**: 60% BM25, 40% VSM (configurable)

## Dataset

- **Source**: Wikipedia Movie Plots (Kaggle)
- **File**: `wiki_movie_plots_deduped.csv`
- **Fields**: Title, Plot, Genre, Director, Cast, Release Year
- **Size**: ~81MB, thousands of movies

## Performance

- **Indexing**: Cached for fast subsequent searches
- **Search Speed**: Typically <1 second for most queries
- **Memory**: Efficient with TF-IDF sparse matrices
- **Scalability**: Suitable for datasets up to 100K+ movies

## Configuration Options

### Search Settings
- **Method**: BM25, VSM, or Hybrid
- **Results Count**: 1-50 movies
- **BM25 Weight**: For hybrid search (0.0-1.0)

### Display Options
- **Show Scores**: Display search relevance scores
- **Show Explanation**: Show input classification reasoning
- **Auto-detect**: Enable/disable automatic input classification

## Future Enhancements

1. **Query Expansion**: Synonym and related term expansion
2. **User Feedback**: Learning from user interactions
3. **Advanced Filters**: By decade, rating, cast, etc.
4. **Recommendation System**: "Similar movies" feature
5. **Caching**: Query result caching for popular searches
6. **Analytics**: Search analytics and user behavior tracking

## File Structure

```
demo/
├── app.py                 # Main Streamlit application
├── data_loader.py         # Dataset loading and preprocessing
├── input_classifier.py    # Input type classification
├── search_engine.py       # BM25 and VSM implementation
├── requirements.txt       # Python dependencies
└── README.md             # This documentation
```

## Dependencies

- streamlit==1.28.0
- pandas==2.0.3  
- numpy==1.24.3
- scikit-learn==1.3.0
- nltk==3.8.1
- rank-bm25==0.2.2
- plotly==5.17.0