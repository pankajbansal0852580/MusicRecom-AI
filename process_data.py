import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

# 1. Load your data (using the 'r' trick!)
df = pd.read_csv(r'C:\Users\ASUS\clean_data.csv')

# 2. Let's take the first 100 songs to keep it fast for now
df_subset = df.head(100).copy()

# 3. Clean the 'English Translations' column 
# (Since it looks like a list in your output, we'll make sure it's a string)
df_subset['English Translations'] = df_subset['English Translations'].astype(str)

# 4. Load the 'Brain'
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Mapping the soul of 100 Bollywood songs... please wait...")

# 5. Create the Vectors
vectors = model.encode(df_subset['English Translations'].tolist(), show_progress_bar=True)

# 6. Save the results so we are 'Production-Ready'
np.save('song_vectors.npy', vectors)
df_subset.to_csv('processed_songs.csv', index=False)

print("Success! You have created a mathematical map of your dataset.")