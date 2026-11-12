"""
Script to prepare chunk index and precompute dense embeddings.
This avoids slow startup times when deployed on Hugging Face Spaces CPU instances.
"""
import os
import re
import json
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "chunks.json")
EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "embeddings.npy")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking configuration: target 200-300 words with small overlap (~40 words)
TARGET_CHUNK_WORDS = 250
OVERLAP_WORDS = 40


def split_into_overlapping_chunks(text: str, filename: str, target_words: int = 250, overlap_words: int = 40):
    """
    Splits text into chunks of target_words with overlap_words overlap,
    preserving natural section headers where possible.
    """
    # Clean up excess whitespace while preserving paragraph boundaries
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    
    chunks = []
    current_words = []
    
    for para in paragraphs:
        para_words = para.split()
        if not para_words:
            continue
            
        # If adding this paragraph exceeds target words and we already have enough words:
        if len(current_words) + len(para_words) > target_words and len(current_words) >= (target_words - overlap_words):
            chunk_text = " ".join(current_words)
            chunks.append(chunk_text)
            # Retain the last `overlap_words` for context continuity
            current_words = current_words[-overlap_words:] + para_words
        else:
            current_words.extend(para_words)
            
    # Add any remaining words as the final chunk
    if current_words:
        chunk_text = " ".join(current_words)
        # Avoid creating an undersized trailing chunk (< 80 words) if previous chunk exists
        if len(current_words) < 80 and chunks:
            chunks[-1] = chunks[-1] + " " + chunk_text
        else:
            chunks.append(chunk_text)
            
    result = []
    for idx, c in enumerate(chunks):
        words = c.split()
        result.append({
            "source": filename,
            "chunk_index": idx,
            "word_count": len(words),
            "text": c
        })
    return result


def main():
    print(f"Reading knowledge base files from: {DATA_DIR}")
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory '{DATA_DIR}' not found.")
        
    all_chunks = []
    chunk_counter = 0
    
    doc_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".md") or f.endswith(".txt")])
    print(f"Found {len(doc_files)} documents: {doc_files}")
    
    for doc in doc_files:
        path = os.path.join(DATA_DIR, doc)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            
        doc_chunks = split_into_overlapping_chunks(content, doc, TARGET_CHUNK_WORDS, OVERLAP_WORDS)
        for c in doc_chunks:
            c["id"] = chunk_counter
            chunk_counter += 1
            all_chunks.append(c)
            
    print(f"Generated {len(all_chunks)} total chunks across {len(doc_files)} documents.")
    for c in all_chunks[:3]:
        print(f"  - Chunk #{c['id']} ({c['source']}, {c['word_count']} words): {c['text'][:90]}...")
        
    # Save chunks metadata
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved chunk metadata to {CHUNKS_PATH}")
    
    # Precompute sentence embeddings
    print(f"Loading embedding model '{MODEL_NAME}' to precompute embeddings...")
    embedder = SentenceTransformer(MODEL_NAME)
    
    chunk_texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(chunk_texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)
    
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Saved embeddings array to {EMBEDDINGS_PATH} with shape: {embeddings.shape}")
    print("Indexing and embedding precomputation finished successfully!")


if __name__ == "__main__":
    main()
