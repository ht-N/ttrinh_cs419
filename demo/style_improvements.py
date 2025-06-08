"""
Additional styling improvements for the movie search app
"""

def get_custom_css():
    """Return custom CSS for better styling"""
    return """
    <style>
        /* Movie card styling */
        .movie-card {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 0.75rem;
            margin: 1rem 0;
            border-left: 5px solid #ff6b6b;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Score badges */
        .score-badge {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 1.5rem;
            font-size: 0.85rem;
            font-weight: bold;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        
        /* Method badges */
        .method-badge {
            background: linear-gradient(45deg, #2196F3, #1976D2);
            color: white;
            padding: 0.2rem 0.6rem;
            border-radius: 1rem;
            font-size: 0.75rem;
            margin-left: 0.5rem;
        }
        
        /* Input type badges */
        .input-type-badge {
            background: linear-gradient(45deg, #FF9800, #F57C00);
            color: white;
            padding: 0.2rem 0.6rem;
            border-radius: 1rem;
            font-size: 0.75rem;
        }
        
        /* Movie title buttons */
        .stButton > button {
            width: 100%;
            text-align: left;
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 0.5rem;
            padding: 0.75rem 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background: linear-gradient(45deg, #764ba2, #667eea);
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        /* Search button styling */
        .stButton > button[kind="primary"] {
            background: linear-gradient(45deg, #ff6b6b, #ee5a52);
            color: white;
            font-weight: bold;
            border-radius: 0.5rem;
            padding: 0.75rem 2rem;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #f0f2f6;
            border-radius: 0.5rem;
        }
        
        /* Metric styling */
        .metric-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 0.5rem;
            color: white;
            text-align: center;
        }
        
        /* Info boxes */
        .stAlert > div {
            border-radius: 0.5rem;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #f8f9fa;
        }
        
        /* Chart styling */
        .js-plotly-plot {
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Custom divider */
        .custom-divider {
            height: 2px;
            background: linear-gradient(90deg, transparent, #ff6b6b, transparent);
            margin: 1rem 0;
            border: none;
        }
        
        /* Movie details section */
        .movie-details {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 1.5rem;
            border-radius: 0.75rem;
            margin: 1rem 0;
        }
        
        /* Search stats */
        .search-stats {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
    </style>
    """

def get_movie_card_html(title, rank, score=None):
    """Generate HTML for movie card header"""
    score_html = f'<span class="score-badge">Score: {score:.3f}</span>' if score else ''
    
    return f"""
    <div class="movie-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h4 style="margin: 0; color: #333;">🎬 #{rank} {title}</h4>
            {score_html}
        </div>
    </div>
    """

def get_search_stats_html(count, method, time_taken):
    """Generate HTML for search statistics"""
    return f"""
    <div class="search-stats">
        <h4 style="margin: 0; color: #333;">🎯 Search Results</h4>
        <p style="margin: 0.5rem 0 0 0; color: #666;">
            Found <strong>{count}</strong> movies using <strong>{method}</strong> in <strong>{time_taken:.2f}s</strong>
        </p>
    </div>
    """