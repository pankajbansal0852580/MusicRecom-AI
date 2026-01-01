import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="MusicRecom AI", page_icon="🌙", layout="wide")

# --- 2. THE "ULTRA-CLEAN" CSS ---
st.markdown("""
    <style>
    /* 1. Background Image */
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0.85)), 
                    url('https://images.unsplash.com/photo-1470225620780-dba8ba36b745?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80');
        background-size: cover;
        background-attachment: fixed;
    }

    /* 2. Custom Title Styling (No grey bar) */
    .main-title {
        color: #ffffff;
        font-family: 'Montserrat', sans-serif;
        font-weight: 800;
        font-size: 50px;
        text-align: left;
        margin-bottom: 0px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .sub-title {
        color: #ffffff;
        font-size: 18px;
        font-weight: 400;
        margin-bottom: 30px;
    }

    /* 3. Glassmorphism Card (Pale Foreground) */
    .song-card {
        background: rgba(255, 255, 255, 0.9); /* Very clean pale foreground */
        padding: 20px 30px;
        border-radius: 15px;
        margin-bottom: 15px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        animation: slideIn 0.5s ease-out;
    }

    /* 4. Dark Font for Song Titles */
    .song-name {
        color: #1a1a1a; /* Deep dark font */
        font-size: 22px;
        font-weight: 700;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    .match-tag {
        background-color: #ff4b4e;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: bold;
    }

    /* 5. Animation */
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* 6. Hiding Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 3. DATA LOADING ---
@st.cache_resource
def load_assets():
    df = pd.read_csv('processed_songs.csv')
    vectors = np.load('song_vectors.npy')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return df, vectors, model

df, song_embeddings, model = load_assets()

# --- 4. INTERFACE ---
st.markdown('<p class="main-title">MusicRecom AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Semantic Music Discovery Engine</p>', unsafe_allow_html=True)

query = st.text_input("", placeholder="Enter a mood or a story...")

if query:
    query_vector = model.encode(query)
    scores = util.cos_sim(query_vector, song_embeddings)[0]
    top_indices = np.argsort(-scores)[:5]
    
    st.write("") # Spacer
    
    for idx in top_indices:
        song = df.iloc[idx.item()]
        score = scores[idx.item()]

        # Clean Card Layout
        st.markdown(f"""
            <div class="song-card">
                <div class="song-name">🎵 {song['Song Title']}</div>
                <div class="match-tag">{score:.1%} Match</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Action button
        search_url = f"https://www.youtube.com/results?search_query={song['Song Title'].replace(' ', '+')}+bollywood+song"
        st.link_button(f"Play {song['Song Title']}", search_url)