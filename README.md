# 🎵 MusicRecom AI: Semantic Music Discovery

A sophisticated Bollywood song recommendation engine that uses **Deep Learning (Transformers)** to map emotional user queries to lyrical content. Unlike traditional keyword-based search, Sangeet AI understands the "vibe" and "soul" of a song.

## 🚀 Key Features
- **Semantic Mapping**: Uses `all-MiniLM-L6-v2` Sentence-Transformers to encode lyrics into 384-dimensional vector space.
- **Vector Persistence**: Optimized retrieval using pre-computed NumPy embeddings for sub-millisecond search.
- **Glassmorphism UI**: A modern, responsive dashboard built with Streamlit and custom CSS.
- **Contextual Awareness**: Capable of identifying "Friendship" even when searching for "Unending bonds."

## 🛠️ Tech Stack
- **Language**: Python 3.13
- **AI/ML**: Sentence-Transformers, PyTorch
- **Data**: Pandas, NumPy
- **Frontend**: Streamlit (Custom CSS/HTML Injection)

## 📂 System Architecture
1. **Preprocessing**: Raw Kaggle CSV -> English Translation Extraction -> Batch Embedding.
2. **Storage**: Vectorized data saved as `.npy` binaries for production efficiency.
3. **Retrieval**: User query -> Vectorization -> Cosine Similarity Calculation -> Top-K Result Extraction.

## 📈 Scalability Roadmap (Million-Song Plan)
To scale this project to 1,000,000+ songs, the following architecture would be implemented:
- **Vector Database**: Transition from NumPy to **FAISS** or **ChromaDB** for Indexed Search.
- **Quantization**: Implementation of Product Quantization (PQ) to reduce memory footprint.
- **GPU Acceleration**: Using CUDA-enabled batches for high-speed lyrical encoding.