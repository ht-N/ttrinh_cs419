"""
Streamlit Movie Search Application
Main application file using BM25 and VSM for movie retrieval
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict
import time

# Import custom modules
from data_loader import MovieDataLoader
from custom_search_engine import CustomSearchEngine
from input_classifier import InputClassifier, InputType

# Page configuration
st.set_page_config(
    page_title="Movie Search Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .movie-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #ff6b6b;
    }
    
    .score-badge {
        background-color: #4CAF50;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .method-badge {
        background-color: #2196F3;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.7rem;
        margin-left: 0.5rem;
    }
    
    .input-type-badge {
        background-color: #FF9800;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.7rem;
    }
    
    .stButton > button {
        width: 100%;
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_movie_data():
    """Load and cache movie data - first 1000 movies only"""
    data_path = "../dataset/wiki_movie_plots_deduped.csv"
    
    # Load only first 1000 movies for faster performance
    df = pd.read_csv(data_path, nrows=1000)
    
    # Create loader instance and manually set the dataframe
    loader = MovieDataLoader(data_path)
    loader.df = df  # Set the limited dataset
    processed_df = loader.preprocess_data()
    return processed_df, loader

@st.cache_resource
def initialize_search_engine(df):
    """Initialize and cache search engine"""
    return CustomSearchEngine(df)

@st.cache_resource
def initialize_classifier(genres):
    """Initialize and cache input classifier"""
    return InputClassifier(genres)

def display_movie_card(movie: Dict, rank: int, show_scores: bool = False):
    """Display a movie card with clickable title for full information"""
    title = movie.get('Title', 'Unknown Title')
    
    # Create a unique key for this movie
    movie_key = f"movie_{rank}_{hash(title) % 10000}"
    
    with st.container():
        # Display title as a clickable button
        col1, col2 = st.columns([4, 1])
        
        with col1:
            # Use button for clickable title
            if st.button(f"🎬 #{rank} {title}", key=movie_key, help="Click to see full movie details"):
                st.session_state[f'show_details_{movie_key}'] = not st.session_state.get(f'show_details_{movie_key}', False)
        
        with col2:
            if show_scores:
                score = movie.get('search_score', 0)
                st.markdown(f'<span class="score-badge">Score: {score:.3f}</span>', 
                           unsafe_allow_html=True)
        
        # Show basic info always
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Year:** {movie.get('Year', 'Unknown')}")
        with col2:
            genre_text = str(movie.get('Genre', 'Unknown'))
            display_genre = f"{genre_text[:30]}..." if len(genre_text) > 30 else genre_text
            st.write(f"**Genre:** {display_genre}")
        with col3:
            if show_scores and 'bm25_score' in movie and 'vsm_score' in movie:
                st.write(f"BM25: {movie['bm25_score']:.3f}")
                st.write(f"VSM: {movie['vsm_score']:.3f}")
        
        # Show full details if clicked
        if st.session_state.get(f'show_details_{movie_key}', False):
            with st.expander("📋 Full Movie Details", expanded=True):
                st.markdown(f"### {title}")
                
                # Movie details in organized layout
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.write(f"**📅 Release Year:** {movie.get('Year', 'Unknown')}")
                    st.write(f"**🎭 Genre:** {movie.get('Genre', 'Unknown')}")
                    st.write(f"**🎬 Director:** {movie.get('Director', 'Unknown')}")
                    st.write(f"**🌍 Origin/Ethnicity:** {movie.get('Origin/Ethnicity', 'Unknown')}")
                
                with detail_col2:
                    st.write(f"**⭐ Cast:** {movie.get('Cast', 'Unknown')}")
                    if 'Wiki Page' in movie:
                        st.write(f"**🔗 Wiki Page:** {movie.get('Wiki Page', 'N/A')}")
                    if show_scores:
                        st.write(f"**📊 Search Score:** {movie.get('search_score', 0):.4f}")
                        st.write(f"**📈 Index:** {movie.get('index', 'N/A')}")
                
                # Plot section
                st.markdown("**📖 Plot Summary:**")
                plot = movie.get('Plot', 'No plot available')
                st.write(plot)
                
                # Additional metadata if available
                additional_fields = ['Release Year', 'Origin/Ethnicity', 'Wiki Page']
                additional_info = {k: v for k, v in movie.items() 
                                 if k not in ['Title', 'Plot', 'Genre', 'Director', 'Cast', 'Year', 
                                            'search_score', 'index', 'bm25_score', 'vsm_score',
                                            'title_text', 'plot_text', 'genre_text', 'combined_text'] 
                                 and v and str(v).strip()}
                
                if additional_info:
                    st.markdown("**📝 Additional Information:**")
                    for key, value in additional_info.items():
                        st.write(f"**{key}:** {value}")

def create_results_chart(results: List[Dict], method: str):
    """Create a chart showing search scores"""
    if not results:
        return None
    
    titles = [f"{movie['Title'][:30]}..." if len(movie['Title']) > 30 
              else movie['Title'] for movie in results[:10]]
    scores = [movie.get('search_score', 0) for movie in results[:10]]
    
    fig = px.bar(
        x=scores,
        y=titles,
        orientation='h',
        title=f"Top 10 Search Results - {method}",
        labels={'x': 'Search Score', 'y': 'Movie Title'}
    )
    
    fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
    return fig

def main():
    st.title("🎬 Movie Search Engine")
    st.markdown("Search for movies using BM25 and Vector Space Model (VSM)")
    st.info("📊 Loading first 1000 movies for optimal performance")
    
    # Load data
    with st.spinner("Loading movie database..."):
        df, loader = load_movie_data()
        search_engine = initialize_search_engine(df)
        genres = loader.get_genres()
        classifier = initialize_classifier(genres)
    
    # Display dataset statistics
    stats = loader.get_movie_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Movies", f"{stats['total_movies']:,}")
    with col2:
        st.metric("Unique Genres", stats['unique_genres'])
    with col3:
        st.metric("Year Range", f"{stats['year_range'][0]}-{stats['year_range'][1]}")
    with col4:
        st.metric("Avg Plot Length", f"{stats['avg_plot_length']:.0f} chars")
    
    # Main search interface
    st.header("🔍 Search Movies")
    
    # Search configuration in main area
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Search input
        query = st.text_input(
            "Enter your search query:",
            value=st.session_state.get('search_query', ''),
            placeholder="e.g., 'action movie', 'love story', '1995', 'Titanic'",
            help="Search by title, plot, genre, year, or any combination"
        )
    
    with col2:
        # Search method selection - MAIN DROPDOWN
        search_method = st.selectbox(
            "🔧 Retrieval Method",
            ["BM25", "Vector Space Model", "Hybrid (BM25 + VSM)"],
            help="Choose the search algorithm"
        )
    
    with col3:
        # Number of results
        top_k = st.selectbox(
            "📊 Results Count",
            [5, 10, 15, 20, 25, 30],
            index=1,  # Default to 10
            help="Number of movies to retrieve"
        )
    
    # Sidebar for advanced configuration
    st.sidebar.header("⚙️ Advanced Settings")
    
    # Advanced search options moved to sidebar
    show_scores = st.sidebar.checkbox("Show search scores", value=True)
    auto_detect = st.sidebar.checkbox("Auto-detect input type", value=True)
    show_explanation = st.sidebar.checkbox("Show classification explanation", value=True)
    
    # Hybrid search configuration
    bm25_weight = 0.6  # Default value
    if search_method == "Hybrid (BM25 + VSM)":
        bm25_weight = st.sidebar.slider(
            "BM25 Weight", 0.0, 1.0, 0.6, 0.1,
            help="Weight for BM25 vs VSM in hybrid search"
        )
    
    # Manual input type override
    manual_type = None
    if not auto_detect:
        manual_type = st.sidebar.selectbox(
            "Search Type Override",
            ["General", "Title", "Plot", "Genre", "Year"],
            help="Manually specify what you're searching for"
        )
    
    # Search execution and results
    if query:
        # Add search button for better UX
        search_button = st.button("🔍 Search Movies", type="primary")
        
        if search_button or query:  # Auto-search or manual search
            # Convert method names for internal use
            method_mapping = {
                "BM25": "bm25",
                "Vector Space Model": "vsm", 
                "Hybrid (BM25 + VSM)": "hybrid"
            }
            
            # Classify input
            if auto_detect:
                input_type, confidence = classifier.classify_input(query)
                
                if show_explanation:
                    explanation = classifier.explain_classification(query)
                    st.info(f"🤖 {explanation}")
                    
                    # Show classification details
                    with st.expander("Classification Details"):
                        st.write(f"**Detected Type:** {input_type.value}")
                        st.write(f"**Confidence:** {confidence:.1%}")
            else:
                # Manual type mapping
                type_mapping = {
                    "General": InputType.GENERAL,
                    "Title": InputType.TITLE,
                    "Plot": InputType.PLOT,
                    "Genre": InputType.GENRE,
                    "Year": InputType.YEAR
                }
                input_type = type_mapping[manual_type]
                confidence = 1.0
            
            # Get search weights based on input type
            search_weights = classifier.get_search_weights(input_type)
            
            # Perform search
            with st.spinner("Searching movies..."):
                start_time = time.time()
                
                if input_type == InputType.YEAR:
                    # Special handling for year search
                    try:
                        year = int(query.strip())
                        results = search_engine.search_by_year(year, top_k)
                    except ValueError:
                        # Fallback to regular search if year parsing fails
                        internal_method = method_mapping[search_method]
                        if internal_method == "hybrid":
                            results = search_engine._hybrid_bm25_vsm(
                                query, search_weights, top_k, bm25_weight
                            )
                        else:
                            results = search_engine.hybrid_search(
                                query, search_weights, internal_method, top_k
                            )
                else:
                    # Regular search
                    internal_method = method_mapping[search_method]
                    
                    if internal_method == "hybrid":
                        results = search_engine._hybrid_bm25_vsm(
                            query, search_weights, top_k, bm25_weight
                        )
                    else:
                        results = search_engine.hybrid_search(
                            query, search_weights, internal_method, top_k
                        )
                
                search_time = time.time() - start_time
            
            # Display results
            if results:
                # Results header with method info
                st.success(f"🎯 Found {len(results)} movies using **{search_method}** in {search_time:.2f} seconds")
                
                # Show method explanation
                method_explanations = {
                    "BM25": "Using BM25 probabilistic ranking function",
                    "Vector Space Model": "Using TF-IDF vectors with cosine similarity",
                    "Hybrid (BM25 + VSM)": f"Combining BM25 ({bm25_weight:.0%}) and VSM ({1-bm25_weight:.0%})"
                }
                st.info(f"ℹ️ {method_explanations[search_method]}")
                
                # Show results chart
                if show_scores and len(results) > 1:
                    chart = create_results_chart(results, search_method)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                
                # Display movie cards
                st.markdown("### 🎬 Search Results")
                st.markdown("*Click on any movie title to see full details*")
                
                for i, movie in enumerate(results, 1):
                    display_movie_card(movie, i, show_scores)
                    
                    if i < len(results):
                        st.divider()
            
            else:
                st.warning("❌ No movies found. Try different keywords or search terms.")
                
                # Suggestions
                with st.expander("💡 Search Suggestions"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Try these tips:**")
                        st.markdown("- Use broader search terms")
                        st.markdown("- Check spelling")
                        st.markdown("- Try different keywords")
                    with col2:
                        st.markdown("**Example searches:**")
                        st.markdown("- Genre: 'action', 'comedy', 'drama'")
                        st.markdown("- Year: '1995', '2000'")
                        st.markdown("- Plot: 'love story', 'time travel'")
    
    # Sample searches section
    st.sidebar.header("Sample Searches")
    sample_searches = [
        "Titanic",
        "action movie",
        "love story",
        "1995",
        "science fiction",
        "Batman",
        "comedy",
        "thriller"
    ]
    
    for sample in sample_searches:
        if st.sidebar.button(f"🔍 {sample}", key=f"sample_{sample}"):
            # Use session state to trigger search with sample query
            st.session_state['search_query'] = sample
            st.rerun()
    
    # About section
    with st.sidebar.expander("About"):
        st.markdown("""
        **Movie Search Engine**
        
        This application uses advanced information retrieval techniques:
        
        - **BM25**: Probabilistic ranking function
        - **Vector Space Model**: TF-IDF with cosine similarity
        - **Hybrid**: Combines both methods
        
        **Features:**
        - Automatic input classification
        - Multi-field search (title, plot, genre)
        - Clickable movie titles for full details
        - Intelligent ranking and scoring
        """)

if __name__ == "__main__":
    main()