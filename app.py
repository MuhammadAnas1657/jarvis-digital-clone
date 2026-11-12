"""
JARVIS - Autonomous Digital Clone of Anas (Hybrid RAG).
Native Gradio 6.x Application designed for Hugging Face Spaces & Local Execution.
Features:
- BM25 Lexical + Dense Cosine (all-MiniLM-L6-v2) + Reciprocal Rank Fusion (RRF)
- Strict Persona Grounding & Honest Refusal
- Multi-Provider LLM: Mistral AI, Groq, Google Gemini, and Hugging Face Inference
- Real-time Retrieved Context Evidence & Score Inspection Panel
- ZeroGPU and CPU Basic Space compatibility
"""
import os
import re
from typing import List, Dict, Any, Optional
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from rag import HybridRAG
from generator import CloneGenerator

# Hugging Face ZeroGPU compatibility
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


def format_evidence_html(chunks: List[Dict[str, Any]], note: str = "") -> str:
    """Renders sleek cyber-styled evidence cards with score breakdowns."""
    if not chunks:
        msg = note if note else "Ask JARVIS a question to inspect retrieved context passages and similarity scores in real time."
        return f'<div style="border:1px dashed rgba(96,165,250,0.3);border-radius:12px;padding:16px;text-align:center;color:#8ea3c7;font-size:12.5px;">{msg}</div>'

    cards = []
    for idx, c in enumerate(chunks, start=1):
        source = c.get("source", "unknown")
        cid = c.get("id", idx)
        score = c.get("score", 0.0)
        mode_label = c.get("retrieval_mode", "RRF")
        text = c.get("text", "")
        # Short snippet preview
        snippet = text[:280] + ("..." if len(text) > 280 else "")

        card = f"""
        <div style="background:#09142e;border:1px solid rgba(96,165,250,0.22);border-radius:12px;padding:12px;margin-bottom:10px;">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:11.5px;">
            <span style="background:linear-gradient(135deg,#2563eb,#0ea5e9);color:#fff;font-weight:bold;padding:1px 7px;border-radius:6px;">#{idx}</span>
            <span style="color:#7dd3fc;font-family:monospace;font-weight:600;">{source} (Chunk #{cid})</span>
            <span style="margin-left:auto;color:#22d3ee;font-family:monospace;font-weight:bold;border:1px solid rgba(34,211,238,0.3);padding:1px 7px;border-radius:6px;">{mode_label}: {score}</span>
          </div>
          <div style="font-size:12px;line-height:1.5;color:#94a3b8;border-left:2px solid #38bdf8;padding-left:8px;">{snippet}</div>
        </div>
        """
        cards.append(card)
    return "".join(cards)


def chat_handler(message: str, chat_history: List[Dict[str, str]], mode: str, top_k: int, api_key: str):
    """Processes user query through intent guardrails, RAG retrieval, and LLM generation."""
    if not message or not message.strip():
        return "", chat_history, format_evidence_html([])

    rag = get_rag()
    gen = get_generator()
    chat_history = chat_history or []

    # 1. Guardrail Intent Check
    intent = rag.classify_intent(message)
    if not intent["relevant"]:
        decline_reply = "I only answer questions grounded in Anas's notes, projects and study habits. " + intent["reason"]
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": decline_reply})
        return "", chat_history, format_evidence_html([], note="Query declined by guardrail: " + intent["reason"])

    # 2. Retrieval
    retrieved = rag.retrieve(query=message, mode=mode, top_k=int(top_k), use_reranker=False)

    # 3. Generation (supports Mistral, Groq, Gemini, HF, and clean offline fallback)
    reply = gen.generate_response(
        query=message,
        retrieved_chunks=retrieved,
        chat_history=chat_history,
        token=api_key.strip() if api_key else None
    )

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": reply})

    evidence_html = format_evidence_html(retrieved)
    return "", chat_history, evidence_html


CUSTOM_CSS = """
body, .gradio-container {
    background-color: #050b1a !important;
    color: #eaf2ff !important;
    font-family: "Segoe UI", system-ui, -apple-system, sans-serif !important;
}
.header-box {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 16px 20px;
    background: radial-gradient(circle at 10% 20%, #0d1e44, #061129);
    border: 1px solid rgba(96, 165, 250, 0.25);
    border-radius: 16px;
    margin-bottom: 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}
.arc-logo {
    width: 48px;
    height: 48px;
    animation: rotateLogo 24s linear infinite;
}
@keyframes rotateLogo { to { transform: rotate(360deg); } }
.banner-notice {
    font-size: 11.5px;
    color: #64748b;
    margin-top: 4px;
}
"""

with gr.Blocks(theme=gr.themes.Ocean(), css=CUSTOM_CSS, title="JARVIS - Anas Digital Clone") as demo:
    # Header Banner
    gr.HTML("""
    <div class="header-box">
      <div style="width:48px;height:48px;display:grid;place-items:center;background:#061331;border-radius:12px;border:1px solid #38bdf8;box-shadow:0 0 16px rgba(56,189,248,0.35);">
        <svg class="arc-logo" viewBox="0 0 64 64" width="40" height="40">
          <circle cx="32" cy="32" r="28" fill="none" stroke="#38bdf8" stroke-width="2.5"/>
          <circle cx="32" cy="32" r="22" fill="none" stroke="#22d3ee" stroke-width="1.2" stroke-dasharray="3 4"/>
          <line x1="32" y1="6" x2="32" y2="12" stroke="#38bdf8" stroke-width="3" stroke-linecap="round"/>
          <line x1="32" y1="52" x2="32" y2="58" stroke="#38bdf8" stroke-width="3" stroke-linecap="round"/>
          <line x1="6" y1="32" x2="12" y2="32" stroke="#38bdf8" stroke-width="3" stroke-linecap="round"/>
          <line x1="52" y1="32" x2="58" y2="32" stroke="#38bdf8" stroke-width="3" stroke-linecap="round"/>
          <polygon points="32,16 46,24 46,40 32,48 18,40 18,24" fill="none" stroke="#7dd3fc" stroke-width="2" stroke-linejoin="round"/>
          <circle cx="32" cy="32" r="8" fill="#38bdf8"/>
          <circle cx="32" cy="32" r="3" fill="#061331"/>
        </svg>
      </div>
      <div>
        <h1 style="font-size:20px;margin:0;letter-spacing:1px;color:#f8fafc;">JARVIS</h1>
        <p style="font-size:12.5px;margin:2px 0 0 0;color:#38bdf8;font-weight:600;">Autonomous Digital Clone of Anas • Hybrid RAG Engine</p>
        <p class="banner-notice">⚠️ Clone Notice: Grounded in authentic documents authored by Anas (UMT Sialkot, CGPA 3.5, Atarion Solutions, SPIDER AI Security). Refuses unverified questions with <i>"I don't have that in my notes."</i></p>
      </div>
    </div>
    """)

    with gr.Row():
        # Left Column: Configuration & Retrieved Context Evidence
        with gr.Column(scale=4):
            with gr.Group():
                gr.Markdown("### ⚙️ Control Station")
                mode_radio = gr.Radio(
                    choices=["Hybrid (RRF)", "Dense only", "BM25 only"],
                    value="Hybrid (RRF)",
                    label="Retrieval Strategy",
                    info="RRF merges lexical BM25 and dense semantic similarity."
                )
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    label="Top-K Chunks to Retrieve"
                )
                api_key_box = gr.Textbox(
                    label="🔑 LLM API Key (Optional)",
                    placeholder="Mistral API, Groq, Gemini, or Hugging Face key...",
                    type="password",
                    info="Connects to real-time neural LLM. Without a key, high-speed grounded synthesis is used automatically."
                )

            with gr.Group():
                gr.Markdown("### 🔍 Retrieved Context Evidence")
                evidence_panel = gr.HTML(
                    value=format_evidence_html([])
                )

        # Right Column: Dialogue Stream
        with gr.Column(scale=6):
            chatbot = gr.Chatbot(
                label="Neural Dialogue Stream",
                height=520,
                type="messages",
                value=[
                    {
                        "role": "assistant",
                        "content": "Hello, I am JARVIS, the autonomous digital clone of Anas. I answer strictly from his verified notes, CV, Atarion Solutions projects, research and study habits. Ask me anything, or click a benchmark query below!"
                    }
                ]
            )

            # Quick Query Chips
            gr.Markdown("**Quick Benchmark Queries:**")
            with gr.Row():
                btn_q1 = gr.Button("⚡ Jarvis Capabilities", size="sm")
                btn_q2 = gr.Button("🏢 Atarion Solutions", size="sm")
                btn_q3 = gr.Button("🛡️ SPIDER & Security", size="sm")
                btn_q4 = gr.Button("🎓 University & CSS", size="sm")
                btn_q5 = gr.Button("🎂 Cake Recipe (Refusal)", size="sm")

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Message JARVIS...",
                    show_label=False,
                    scale=8,
                    container=False
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear", scale=1)

    # Wire event handlers
    msg_input.submit(
        chat_handler,
        inputs=[msg_input, chatbot, mode_radio, top_k_slider, api_key_box],
        outputs=[msg_input, chatbot, evidence_panel]
    )
    send_btn.click(
        chat_handler,
        inputs=[msg_input, chatbot, mode_radio, top_k_slider, api_key_box],
        outputs=[msg_input, chatbot, evidence_panel]
    )

    def clear_chat():
        initial_msg = [{
            "role": "assistant",
            "content": "Hello, I am JARVIS, the autonomous digital clone of Anas. I answer strictly from his verified notes, CV, Atarion Solutions projects, research and study habits. Ask me anything, or click a benchmark query below!"
        }]
        return "", initial_msg, format_evidence_html([])

    clear_btn.click(clear_chat, outputs=[msg_input, chatbot, evidence_panel])

    # Preset button click actions
    btn_q1.click(lambda: "What is Jarvis and what are its core capabilities?", outputs=msg_input).then(
        chat_handler, inputs=[msg_input, chatbot, mode_radio, top_k_slider, api_key_box], outputs=[msg_input, chatbot, evidence_panel]
    )
    btn_q2.click(lambda: "What is Atarion Solutions and what products does it build?", outputs=msg_input).then(
        chat_handler, inputs=[msg_input, chatbot, mode_radio, top_k_slider, api_key_box], outputs=[msg_input, chatbot, evidence_panel]
    )
    btn_q3.click(lambda: "Tell me about your research on SPIDER and prompt injection defense.", outputs=msg_input).then(
        chat_handler, inputs=[msg_input, chatbot, mode_radio, top_k_slider, api_key_box], outputs=[msg_input, chatbot, evidence_panel]
    )
    btn_q4.click(lambda: "Which university do you attend, what is your CGPA, and what is your role in CSS?", outputs=msg_input).then(
        chat_handler, inputs=[msg_input, chatbot, mode_radio, top_k_slider, api_key_box], outputs=[msg_input, chatbot, evidence_panel]
    )
    btn_q5.click(lambda: "What is your secret recipe for baking chocolate cake?", outputs=msg_input).then(
        chat_handler, inputs=[msg_input, chatbot, mode_radio, top_k_slider, api_key_box], outputs=[msg_input, chatbot, evidence_panel]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
