"""
JARVIS - Autonomous Digital Clone of Anas (Hybrid RAG).
Native Gradio Application designed for Hugging Face Spaces & Local Execution.
Features:
- BM25 Lexical + Dense Cosine (all-MiniLM-L6-v2) + Reciprocal Rank Fusion (RRF)
- Strict Persona Grounding & Honest Refusal
- Multi-Provider LLM: Mistral AI, Groq, Google Gemini, and Hugging Face Inference
- Full-width Dialogue Stream with Expandable Control Accordions & Evidence Panel
- Cybernetic Jarvis Arc Reactor Theme (Unified Deep Space Navy & Neon Cyan)
"""
import os
import re
from typing import List, Dict, Any, Optional
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from rag import HybridRAG
from generator import CloneGenerator

# Hugging Face ZeroGPU compatibility hook
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
        msg = note if note else "Ask JARVIS a question or click a benchmark query to inspect retrieved context passages and similarity scores in real time."
        return f'<div style="border:1px dashed rgba(56,189,248,0.3);background:#071126;border-radius:10px;padding:12px;text-align:center;color:#94a3b8;font-size:12px;">{msg}</div>'

    cards = []
    for idx, c in enumerate(chunks, start=1):
        source = c.get("source", "unknown")
        cid = c.get("id", idx)
        score = c.get("score", 0.0)
        mode_label = c.get("retrieval_mode", "RRF")
        text = c.get("text", "")
        snippet = text[:300] + ("..." if len(text) > 300 else "")

        card = f"""
        <div style="background:#07122b;border:1px solid #1e3a8a;border-radius:8px;padding:10px;margin-bottom:8px;">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;font-size:11.5px;">
            <span style="background:linear-gradient(135deg,#0284c7,#0369a1);color:#ffffff;font-weight:bold;padding:1px 6px;border-radius:4px;">#{idx}</span>
            <span style="color:#38bdf8;font-family:monospace;font-weight:600;">{source} (Chunk #{cid})</span>
            <span style="margin-left:auto;color:#7dd3fc;font-family:monospace;font-weight:bold;background:#0b1d44;border:1px solid rgba(56,189,248,0.4);padding:1px 6px;border-radius:4px;">{mode_label}: {score}</span>
          </div>
          <div style="font-size:12px;line-height:1.5;color:#cbd5e1;border-left:2px solid #0284c7;padding-left:8px;">{snippet}</div>
        </div>
        """
        cards.append(card)
    return "".join(cards)


def chat_handler(message: str, chat_history: List[Dict[str, str]], mode: str, top_k: int, api_key: str):
    """Processes user query through intent guardrails, RAG retrieval, and LLM generation with fail-safe error isolation."""
    if not message or not message.strip():
        return "", chat_history, format_evidence_html([])

    chat_history = chat_history or []

    try:
        rag = get_rag()
        gen = get_generator()

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

    except Exception as exc:
        error_msg = f"Diagnostics: {str(exc)}"
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": "I encountered a processing issue: " + error_msg})
        return "", chat_history, format_evidence_html([], note="Error: " + error_msg)


CUSTOM_CSS = """
body, .gradio-container {
    background-color: #030712 !important;
    color: #f1f5f9 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 !important;
    padding: 6px 14px !important;
    box-sizing: border-box !important;
}

/* Compact Header */
.header-box {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    background: radial-gradient(circle at 10% 20%, #0c1a3b, #050b1a);
    border: 1px solid #1e3a8a;
    border-radius: 10px;
    margin-bottom: 8px;
}
.arc-logo {
    width: 36px;
    height: 36px;
    animation: rotateLogo 24s linear infinite;
}
@keyframes rotateLogo { to { transform: rotate(360deg); } }

/* Quick Question Chip Buttons */
.benchmark-btn {
    background: #0b1739 !important;
    border: 1px solid #1e3a8a !important;
    color: #38bdf8 !important;
    border-radius: 6px !important;
    font-size: 11.5px !important;
    font-weight: 500 !important;
    padding: 4px 8px !important;
    transition: all 0.2s ease !important;
}
.benchmark-btn:hover {
    background: #0284c7 !important;
    color: #ffffff !important;
    border-color: #38bdf8 !important;
}

/* Primary Action Buttons */
button.primary {
    background: linear-gradient(135deg, #0284c7, #0369a1) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* Accordion Styling */
.gr-accordion {
    background: #071126 !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    margin-top: 8px !important;
}
.gr-accordion > .label-wrap {
    color: #38bdf8 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}

/* Hide clutter copy buttons on messages */
button[aria-label="Copy"], button[title="Copy"], button[aria-label="copy"], .copy_button {
    display: none !important;
}
"""

with gr.Blocks(title="JARVIS - Anas Digital Clone") as demo:
    # 1. Header Banner with Arc Reactor Branding
    gr.HTML("""
    <div class="header-box">
      <div style="width:40px;height:40px;display:grid;place-items:center;background:#050d24;border-radius:10px;border:1px solid #0284c7;box-shadow:0 0 12px rgba(2,132,199,0.4);flex-shrink:0;">
        <svg class="arc-logo" viewBox="0 0 64 64" width="32" height="32">
          <circle cx="32" cy="32" r="28" fill="none" stroke="#0284c7" stroke-width="2.5"/>
          <circle cx="32" cy="32" r="22" fill="none" stroke="#38bdf8" stroke-width="1.2" stroke-dasharray="3 4"/>
          <line x1="32" y1="6" x2="32" y2="12" stroke="#0284c7" stroke-width="3" stroke-linecap="round"/>
          <line x1="32" y1="52" x2="32" y2="58" stroke="#0284c7" stroke-width="3" stroke-linecap="round"/>
          <line x1="6" y1="32" x2="12" y2="32" stroke="#0284c7" stroke-width="3" stroke-linecap="round"/>
          <line x1="52" y1="32" x2="58" y2="32" stroke="#0284c7" stroke-width="3" stroke-linecap="round"/>
          <polygon points="32,16 46,24 46,40 32,48 18,40 18,24" fill="none" stroke="#7dd3fc" stroke-width="2" stroke-linejoin="round"/>
          <circle cx="32" cy="32" r="8" fill="#38bdf8"/>
          <circle cx="32" cy="32" r="3" fill="#050d24"/>
        </svg>
      </div>
      <div style="flex-grow:1;">
        <div style="display:flex;align-items:center;gap:8px;">
          <span style="font-size:17px;font-weight:bold;letter-spacing:1px;color:#f8fafc;">JARVIS</span>
          <span style="background:#0284c7;color:#ffffff;font-size:10px;font-weight:bold;padding:1px 6px;border-radius:6px;">ONLINE</span>
          <span style="color:#38bdf8;font-size:12px;font-weight:600;">• Autonomous Digital Clone of Anas</span>
        </div>
        <p style="font-size:11.5px;color:#94a3b8;margin:2px 0 0 0;">Strict Grounding: Verified notes on Atarion Solutions, SPIDER AI Security, and UMT Sialkot. Unverified questions refuse with <i>"I don't have that in my notes."</i></p>
      </div>
    </div>
    """)

    # 2. Main Dialogue Stream (Full Width)
    chatbot = gr.Chatbot(
        label="JARVIS Neural Dialogue",
        height=360,
        value=[
            {
                "role": "assistant",
                "content": "Hello, I am JARVIS, the autonomous digital clone of Anas. I answer strictly from his verified notes, CV, Atarion Solutions projects, research and study habits. Ask me anything, or click a benchmark query below!\n\n*(Tip: Instant grounded clone mode is active. You can also paste your Mistral, Groq, Gemini, or HF key into Settings below anytime.)*"
            }
        ]
    )

    # 3. Message Input Row (Directly Below Chatbot)
    with gr.Row():
        msg_input = gr.Textbox(
            placeholder="Ask JARVIS about Anas's projects, research, habits, or university...",
            show_label=False,
            scale=8,
            container=False
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)
        clear_btn = gr.Button("Clear", scale=1)

    # 4. Quick Benchmark Queries at the FULL END of the chat interface
    gr.Markdown("**Quick Benchmark Queries:**")
    with gr.Row():
        btn_q1 = gr.Button("⚡ Jarvis Capabilities", elem_classes=["benchmark-btn"], size="sm")
        btn_q2 = gr.Button("🏢 Atarion Solutions", elem_classes=["benchmark-btn"], size="sm")
        btn_q3 = gr.Button("🛡️ SPIDER & Security", elem_classes=["benchmark-btn"], size="sm")
        btn_q4 = gr.Button("🎓 University & CSS", elem_classes=["benchmark-btn"], size="sm")
        btn_q5 = gr.Button("🎂 Cake Recipe (Refusal)", elem_classes=["benchmark-btn"], size="sm")

    # 5. Expandable & Contractable Control Settings (Below the Chat Panel)
    with gr.Accordion("⚙️ RAG Control Station & Custom LLM API Key (Mistral, Groq, Gemini, HF)", open=False):
        with gr.Row():
            mode_radio = gr.Radio(
                choices=["Hybrid (RRF)", "Dense only", "BM25 only"],
                value="Hybrid (RRF)",
                label="Retrieval Strategy",
                info="RRF merges lexical BM25 and dense semantic similarity (k=60)."
            )
            top_k_slider = gr.Slider(
                minimum=1,
                maximum=5,
                value=3,
                step=1,
                label="Top-K Chunks to Retrieve"
            )
        api_key_box = gr.Textbox(
            label="🔑 Custom LLM API Key (Optional)",
            placeholder="Paste your Mistral API key, Groq (gsk_...), Gemini (AIza...), or Hugging Face (hf_...) token...",
            type="password",
            info="Supports Mistral AI, Groq (Free), Google Gemini (Free), or Hugging Face tokens. When left empty, instant offline grounded synthesis is used automatically."
        )

    # 6. Expandable & Contractable Retrieved Evidence Inspection (Below the Chat Panel)
    with gr.Accordion("🔍 Retrieved Context Evidence & Score Inspection", open=False):
        evidence_panel = gr.HTML(
            value=format_evidence_html([])
        )

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
    demo.launch(
        theme=gr.themes.Soft(primary_hue="sky", neutral_hue="slate"),
        css=CUSTOM_CSS,
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
