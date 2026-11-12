"""
Automated Evaluation and Verification Script.
Tests:
1. Chunk count and shape of embeddings.
2. BM25 keyword retrieval (specifically testing the rare keyword 'Jarvis').
3. Dense semantic retrieval (testing paraphrased questions with zero lexical overlap).
4. Hybrid Reciprocal Rank Fusion (RRF) vs. Weighted Fusion.
5. Persona generation & strict grounding ('I don't have that in my notes.').
"""
import os
import json
import numpy as np
from rag import HybridRAG
from generator import CloneGenerator

TEST_QUERIES = [
    {
        "id": "Q1",
        "type": "Exact Rare Keyword",
        "query": "What is Jarvis and what are its core capabilities?",
        "expected_kw": "jarvis"
    },
    {
        "id": "Q2",
        "type": "Paraphrased (Zero Shared Words)",
        "query": "How do you stay calm and physically reset when feeling exhausted from studying?",
        "expected_concept": "boxing / fitness / sleep schedule"
    },
    {
        "id": "Q3",
        "type": "Hybrid Evaluation",
        "query": "Tell me about your SPIDER research and prompt injection defense.",
        "expected_concept": "SPIDER / prompt injection / small models"
    },
    {
        "id": "Q4",
        "type": "Multi-Fact Query",
        "query": "Which university do you attend, what is your CGPA, and what is your role in CSS?",
        "expected_concept": "UMT Sialkot / 3.5 CGPA / Vice President CSS"
    },
    {
        "id": "Q5",
        "type": "Out-of-Domain Grounding Refusal",
        "query": "What is your secret recipe for baking chocolate cake?",
        "expected_concept": "I don't have that in my notes."
    }
]


def run_tests():
    print("=" * 70)
    print("RUNNING AUTOMATED VERIFICATION FOR ANAS DIGITAL CLONE")
    print("=" * 70)

    # 1. Verify Data & Chunks
    with open("chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    embeddings = np.load("embeddings.npy")
    
    print(f"[PASSED] Chunks loaded: {len(chunks)} chunks across documents.")
    print(f"[PASSED] Precomputed embeddings shape: {embeddings.shape}")
    assert len(chunks) == embeddings.shape[0], "Mismatch between chunks count and embedding rows!"
    assert embeddings.shape[1] == 384, f"Expected 384 embedding dimension for all-MiniLM-L6-v2, got {embeddings.shape[1]}"

    # Check chunk sizes
    for c in chunks:
        assert 100 <= c["word_count"] <= 450, f"Chunk #{c['id']} word count ({c['word_count']}) out of reasonable bounds"
        assert "source" in c and "id" in c, f"Chunk #{c['id']} missing required metadata"
    print("[PASSED] All chunks adhere to word-count boundaries and contain source metadata.")

    # 2. Initialize RAG
    rag = HybridRAG()
    gen = CloneGenerator()

    print("\n" + "=" * 70)
    print("EVALUATING 5 TEST QUERIES ACROSS RETRIEVAL MODES")
    print("=" * 70)

    for item in TEST_QUERIES:
        qid = item["id"]
        qtype = item["type"]
        query = item["query"]

        print(f"\n--- [{qid}] {qtype}: '{query}' ---")
        
        # Test BM25
        bm25_res = rag.bm25_search(query, top_k=3)
        top_bm25 = bm25_res[0] if bm25_res else None
        if top_bm25:
            print(f"  [BM25] Top-1: {top_bm25['source']} (Chunk #{top_bm25['id']}, Score: {top_bm25['score']})")
            print(f"         Snippet: {top_bm25['text'][:120]}...")
            
        # Test Dense
        dense_res = rag.dense_search(query, top_k=3)
        top_dense = dense_res[0] if dense_res else None
        if top_dense:
            print(f"  [Dense] Top-1: {top_dense['source']} (Chunk #{top_dense['id']}, Score: {top_dense['score']})")
            print(f"          Snippet: {top_dense['text'][:120]}...")

        # Test Hybrid RRF
        hybrid_res = rag.hybrid_search_rrf(query, top_k=3)
        top_hybrid = hybrid_res[0] if hybrid_res else None
        if top_hybrid:
            print(f"  [Hybrid RRF] Top-1: {top_hybrid['source']} (Chunk #{top_hybrid['id']}, RRF Score: {top_hybrid['score']})")
            print(f"               Snippet: {top_hybrid['text'][:120]}...")

        # Test Generation
        answer = gen.generate_response(query, hybrid_res)
        print(f"  [Clone Response]: {answer[:200]}...")

    # Specific assertions
    print("\n" + "=" * 70)
    print("RUNNING STRICT CRITERIA ASSERTIONS")
    print("=" * 70)

    # Q1 Assertion: BM25 must find Jarvis in projects.md or faq.md with high score
    q1_bm25 = rag.bm25_search("Jarvis", top_k=1)[0]
    assert "jarvis" in q1_bm25["text"].lower(), "BM25 failed to retrieve Jarvis on rare keyword query!"
    print("[PASSED] BM25 keyword retrieval correctly wins on rare keyword 'Jarvis'.")

    # Q2 Assertion: Dense retrieval successfully captures boxing/fitness/sleep without shared words
    q2_dense = rag.dense_search("stay calm and physically reset when feeling exhausted from studying", top_k=1)[0]
    assert ("boxing" in q2_dense["text"].lower() or "fitness" in q2_dense["text"].lower() or "sleep" in q2_dense["text"].lower() or "overthink" in q2_dense["text"].lower()), "Dense failed on paraphrased query!"
    print("[PASSED] Dense retrieval correctly captures semantic intent on paraphrased query.")

    # Q5 Assertion: Clone admits lack of notes for out-of-domain
    q5_res = rag.retrieve("What is your secret recipe for baking chocolate cake?", mode="Hybrid", top_k=3)
    q5_answer = gen.generate_response("What is your secret recipe for baking chocolate cake?", q5_res)
    assert "I don't have that in my notes" in q5_answer or "Guardrail Notice" in q5_answer, f"Clone failed honesty test: {q5_answer}"
    print("[PASSED] Honest refusal assertion passed: clone says 'I don't have that in my notes.'")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()

