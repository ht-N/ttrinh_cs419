"""
Simple test to verify dropdown functionality and method selection
"""

import streamlit as st
import pandas as pd

def test_dropdown():
    st.title("🧪 Dropdown Test")
    
    # Test the dropdown options
    search_method = st.selectbox(
        "🔧 Choose Retrieval Method",
        ["BM25", "Vector Space Model", "Hybrid (BM25 + VSM)"],
        help="Select your preferred search algorithm"
    )
    
    # Show selected method
    st.write(f"**Selected Method:** {search_method}")
    
    # Method mapping test
    method_mapping = {
        "BM25": "bm25",
        "Vector Space Model": "vsm", 
        "Hybrid (BM25 + VSM)": "hybrid"
    }
    
    internal_method = method_mapping[search_method]
    st.write(f"**Internal Method Code:** {internal_method}")
    
    # Test results count dropdown
    top_k = st.selectbox(
        "📊 Number of Results",
        [5, 10, 15, 20, 25, 30],
        index=1  # Default to 10
    )
    
    st.write(f"**Results to show:** {top_k}")
    
    # Test search button
    if st.button("🔍 Test Search", type="primary"):
        st.success(f"✅ Search would use {search_method} to find {top_k} results!")
        
        # Show method explanation
        explanations = {
            "BM25": "🔍 BM25 uses probabilistic ranking with term frequency saturation",
            "Vector Space Model": "📊 VSM uses TF-IDF vectors with cosine similarity",
            "Hybrid (BM25 + VSM)": "🔄 Hybrid combines both BM25 and VSM for better results"
        }
        
        st.info(explanations[search_method])
    
    # Test clickable movie titles
    st.header("🎬 Test Movie Cards")
    
    # Sample movie data
    sample_movies = [
        {"Title": "The Matrix", "Year": 1999, "Genre": "Sci-Fi", "Score": 0.95},
        {"Title": "Titanic", "Year": 1997, "Genre": "Romance", "Score": 0.87},
        {"Title": "Avatar", "Year": 2009, "Genre": "Action", "Score": 0.82}
    ]
    
    for i, movie in enumerate(sample_movies, 1):
        # Create clickable title
        movie_key = f"test_movie_{i}"
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button(f"🎬 #{i} {movie['Title']}", key=movie_key):
                st.session_state[f'show_test_{movie_key}'] = not st.session_state.get(f'show_test_{movie_key}', False)
        
        with col2:
            st.markdown(f'<span style="background-color: #4CAF50; color: white; padding: 0.2rem 0.5rem; border-radius: 1rem; font-size: 0.8rem;">Score: {movie["Score"]:.2f}</span>', unsafe_allow_html=True)
        
        # Show basic info
        st.write(f"**Year:** {movie['Year']} | **Genre:** {movie['Genre']}")
        
        # Show details if clicked
        if st.session_state.get(f'show_test_{movie_key}', False):
            with st.expander("📋 Full Movie Details", expanded=True):
                st.write(f"### {movie['Title']}")
                st.write(f"**📅 Release Year:** {movie['Year']}")
                st.write(f"**🎭 Genre:** {movie['Genre']}")
                st.write(f"**📊 Search Score:** {movie['Score']:.4f}")
                st.write("**📖 Plot:** This is a sample plot description for testing purposes.")
        
        if i < len(sample_movies):
            st.divider()

if __name__ == "__main__":
    test_dropdown()