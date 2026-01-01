import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

# 1. Load the 'Map' we created earlier
df = pd.read_csv('processed_songs.csv')
song_embeddings = np.load('song_vectors.npy')

# 2. Load the Brain
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Get User Input
user_mood = input("\nDescribe the vibe or story you're looking for: ")

# 4. Turn the user's mood into a Vector
query_embedding = model.encode(user_mood)

# 5. The Math: Find the top 3 most similar songs
# This compares your mood vector against all 100 song vectors instantly
cosine_scores = util.cos_sim(query_embedding, song_embeddings)[0]

# Get the indices of the top 3 scores
top_results = np.argpartition(-cosine_scores, range(3))[:3]

print("\n--- AI Recommended Songs ---")
for idx in top_results:
    song_title = df.iloc[idx.item()]['Song Title']
    score = cosine_scores[idx.item()]
    # We'll show a snippet of the translation
    preview = str(df.iloc[idx.item()]['English Translations'])[:100] + "..."
    
    print(f"🎵 {song_title} (Match Score: {score:.4f})")
    print(f"   Vibe: {preview}\n")