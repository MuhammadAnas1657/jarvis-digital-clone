"""
Core RAG Module: BM25 Lexical Search, Dense Semantic Search, and Reciprocal Rank Fusion (RRF).
Includes optional Cross-Encoder reranking and query intent guardrails.
"""
import os
import re
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Optional cross-encoder for bonus reranking
try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False

BASE_DIR = os.path.dirname(__file__)
CHUNKS_PATH = os.path.join(BASE_DIR, "chunks.json")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embeddings.npy")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def tokenize_text(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens."""
    return re.findall(r'\w+', text.lower())


class HybridRAG:
    def __init__(self, chunks_path: str = CHUNKS_PATH, embeddings_path: str = EMBEDDINGS_PATH):
        print("Initializing HybridRAG Engine...")
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"Missing {chunks_path}. Run prepare_index.py first.")
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Missing {embeddings_path}. Run prepare_index.py first.")

        # 1. Load chunks
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks: List[Dict[str, Any]] = json.load(f)
        print(f"Loaded {len(self.chunks)} knowledge base chunks.")

        # 2. Build BM25 index
        print("Building BM25 index...")
        self.corpus_tokens = [tokenize_text(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

        # 3. Load dense embeddings and sentence transformer
        print(f"Loading precomputed embeddings from {embeddings_path}...")
        self.embeddings = np.load(embeddings_path)
        print(f"Loaded embedding matrix shape: {self.embeddings.shape}")
        
        print(f"Loading embedding model: {MODEL_NAME}...")
        self.embedder = SentenceTransformer(MODEL_NAME)
        
        self.cross_encoder = None  # Lazily loaded if requested

    def _get_cross_encoder(self):
        if self.cross_encoder is None and HAS_CROSS_ENCODER:
            try:
                print(f"Loading Cross-Encoder model: {CROSS_ENCODER_NAME}...")
                self.cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)
            except Exception as e:
                print(f"Could not load CrossEncoder: {e}")
        return self.cross_encoder

    def bm25_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rank chunks using BM25 keyword matching.
        Returns top-k chunks with their BM25 scores.
        """
        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []
            
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            chunk = dict(self.chunks[idx])
            chunk["score"] = round(score, 4)
            chunk["retrieval_mode"] = "BM25"
            results.append(chunk)
        return results

    def dense_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rank chunks using cosine similarity of dense embeddings.
        Returns top-k chunks with their cosine similarity scores.
        """
        query_vec = self.embedder.encode([query], normalize_embeddings=True)[0]
        # Cosine similarity is the dot product of normalized vectors
        scores = np.dot(self.embeddings, query_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            chunk = dict(self.chunks[idx])
            chunk["score"] = round(score, 4)
            chunk["retrieval_mode"] = "Dense"
            results.append(chunk)
        return results

    def hybrid_search_rrf(self, query: str, top_k: int = 5, k_rrf: int = 60, candidate_k: int = 15) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF):
        Combines BM25 and Dense retrieval rankings using:
        RRF_score(d) = 1 / (k_rrf + rank_bm25(d)) + 1 / (k_rrf + rank_dense(d))
        """
        # Retrieve candidate pools from both retrievers
        bm25_results = self.bm25_search(query, top_k=candidate_k)
        dense_results = self.dense_search(query, top_k=candidate_k)

        rrf_scores: Dict[int, float] = {}
        chunk_map: Dict[int, Dict[str, Any]] = {}
        bm25_ranks: Dict[int, int] = {}
        dense_ranks: Dict[int, int] = {}
        bm25_score_map: Dict[int, float] = {}
        dense_score_map: Dict[int, float] = {}

        for rank, item in enumerate(bm25_results, start=1):
            cid = item["id"]
            chunk_map[cid] = item
            bm25_ranks[cid] = rank
            bm25_score_map[cid] = item["score"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k_rrf + rank))

        for rank, item in enumerate(dense_results, start=1):
            cid = item["id"]
            chunk_map[cid] = item
            dense_ranks[cid] = rank
            dense_score_map[cid] = item["score"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k_rrf + rank))

        # Sort by accumulated RRF score descending
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

        results = []
        for cid in sorted_cids:
            chunk = dict(self.chunks[cid])
            chunk["score"] = round(rrf_scores[cid], 5)
            chunk["rrf_score"] = round(rrf_scores[cid], 5)
            chunk["bm25_rank"] = bm25_ranks.get(cid, None)
            chunk["dense_rank"] = dense_ranks.get(cid, None)
            chunk["bm25_score"] = bm25_score_map.get(cid, 0.0)
            chunk["dense_score"] = dense_score_map.get(cid, 0.0)
            chunk["retrieval_mode"] = "Hybrid (RRF)"
            results.append(chunk)
            
        return results

    def hybrid_search_weighted(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        Alternative Hybrid Fusion: Min-Max Normalized Weighted Score
        score = alpha * norm(dense) + (1 - alpha) * norm(bm25)
        Provided for comparison in the evaluation report.
        """
        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []
            
        # BM25 scores
        bm25_raw = np.array(self.bm25.get_scores(query_tokens))
        # Dense scores
        query_vec = self.embedder.encode([query], normalize_embeddings=True)[0]
        dense_raw = np.dot(self.embeddings, query_vec)

        # Min-max normalization
        def min_max_norm(arr):
            rng = np.ptp(arr)
            return (arr - np.min(arr)) / (rng + 1e-9)

        bm25_norm = min_max_norm(bm25_raw)
        dense_norm = min_max_norm(dense_raw)

        combined = alpha * dense_norm + (1.0 - alpha) * bm25_norm
        top_indices = np.argsort(combined)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = dict(self.chunks[idx])
            chunk["score"] = round(float(combined[idx]), 4)
            chunk["retrieval_mode"] = f"Hybrid (Weighted alpha={alpha})"
            results.append(chunk)
        return results

    def rerank_with_cross_encoder(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """Optional cross-encoder reranking over hybrid candidates."""
        ce = self._get_cross_encoder()
        if not ce or not candidates:
            return candidates[:top_k]
            
        pairs = [[query, c["text"]] for c in candidates]
        ce_scores = ce.predict(pairs)
        
        ranked_indices = np.argsort(ce_scores)[::-1][:top_k]
        reranked = []
        for i in ranked_indices:
            item = dict(candidates[i])
            item["ce_score"] = round(float(ce_scores[i]), 4)
            reranked.append(item)
        return reranked

    def retrieve(self, query: str, mode: str = "Hybrid", top_k: int = 3, use_reranker: bool = False) -> List[Dict[str, Any]]:
        """
        Unified retrieval entry point.
        mode: 'Hybrid', 'Dense', or 'BM25'
        """
        mode_lower = mode.lower()
        if "bm25" in mode_lower:
            results = self.bm25_search(query, top_k=top_k)
        elif "dense" in mode_lower:
            results = self.dense_search(query, top_k=top_k)
        else:
            # Default to Hybrid RRF
            results = self.hybrid_search_rrf(query, top_k=top_k)

        if use_reranker and HAS_CROSS_ENCODER:
            results = self.rerank_with_cross_encoder(query, results, top_k=top_k)
            
        return results

    def classify_intent(self, query: str) -> Dict[str, Any]:
        """
        Agentic guardrail intent check:
        Determines whether the query is relevant to Anas or completely out-of-domain.
        """
        q = query.strip().lower()
        if len(q) < 3:
            return {"relevant": False, "reason": "Query is too short."}
            
        # Quick heuristic check for completely off-topic topics
        disallowed_keywords = ["recipe", "bake cake", "stock market", "weather in tokyo", "quantum physics gravity"]
        for kw in disallowed_keywords:
            if kw in q:
                return {
                    "relevant": False,
                    "reason": f"Topic '{kw}' is outside Anas's knowledge base."
                }
                
        return {"relevant": True, "reason": "Valid query."}

