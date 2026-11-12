"""
JARVIS - Digital Clone of Anas (Hybrid RAG).
Custom HTML front-end (static/index.html) served by FastAPI, with a Gradio app mounted
at /gradio so the project still deploys as a standard Gradio Space.
"""
import os
import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional

from rag import HybridRAG
from generator import CloneGenerator

try:
    import spaces
    @spaces.GPU
    def gpu_init():
        return True
    try:
        gpu_init()
    except Exception:
        pass
except ImportError:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
rag_engine: Optional[HybridRAG] = None
generator: Optional[CloneGenerator] = None


def get_rag() -> HybridRAG:
    global rag_engine
    if rag_engine is None:
        rag_engine = HybridRAG()
    return rag_engine


def get_generator() -> CloneGenerator:
    global generator
    if generator is None:
        generator = CloneGenerator()
    return generator


def label_for(mode_label: str) -> str:
    m = mode_label.lower()
    if "hybrid" in m or "rrf" in m:
        return "RRF"
    return "BM25" if "bm25" in m else "Cosine"


def chat_logic(message: str, mode: str, top_k: int, history: List[Dict[str, str]], hf_token: Optional[str] = None) -> dict:
    rag, gen = get_rag(), get_generator()
    intent = rag.classify_intent(message)
    if not intent["relevant"]:
        return {
            "reply": "I only answer questions grounded in Anas's notes, projects and study habits. " + intent["reason"],
            "chunks": [],
            "note": "Query declined by the guardrail: " + intent["reason"],
        }
    retrieved = rag.retrieve(query=message, mode=mode, top_k=top_k, use_reranker=False)
    reply = gen.generate_response(message, retrieved, chat_history=history, token=hf_token)
    chunks = [
        {
            "source": c.get("source", "unknown"),
            "score": c.get("score", 0.0),
            "label": label_for(c.get("retrieval_mode", mode)),
            "text": c.get("text", ""),
        }
        for c in retrieved
    ]
    return {"reply": reply, "chunks": chunks}


class ChatRequest(BaseModel):
    message: str
    mode: str = "Hybrid (RRF)"
    top_k: int = 3
    history: List[Dict[str, str]] = []
    hf_token: Optional[str] = None


def index():
    with open(os.path.join(BASE_DIR, "static", "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


def api_chat(req: ChatRequest):
    try:
        return JSONResponse(chat_logic(req.message, req.mode, max(1, min(req.top_k, 5)), req.history, req.hf_token))
    except Exception as e:
        return JSONResponse({"reply": f"Something went wrong: {e}", "chunks": []}, status_code=200)


# Minimal Gradio app mounted for Gradio-SDK compatibility
with gr.Blocks(title="JARVIS") as demo:
    gr.HTML('<div style="text-align:center;padding:40px;font-family:sans-serif;"><h2>JARVIS is running.</h2><p><a href="/" style="color:#2563eb;font-weight:bold;">Click here to open the full JARVIS interface</a></p></div>')

# Attach custom routes directly to demo.app
demo.app.add_api_route("/", index, methods=["GET"])
demo.app.add_api_route("/chat", api_chat, methods=["POST"])
demo.app.add_api_route("/custom_api/chat", api_chat, methods=["POST"])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
