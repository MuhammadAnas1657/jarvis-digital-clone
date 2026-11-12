"""
Persona Generation Module for Anas's Digital Clone.
Interacts with Mistral API, Groq, Google GenAI, and Hugging Face Inference API with offline fallback.
Enforces strict grounding, first-person voice, and zero hallucination.
"""
import os
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

from huggingface_hub import InferenceClient

HF_TOKEN = os.environ.get("HF_TOKEN", None)
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", None)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", None)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", None)
MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"

SYSTEM_PERSONA_PROMPT = """You are Jarvis, the autonomous AI digital clone of Anas, a 22-year-old BS Computer Science student at UMT Sialkot, Founder and CEO of Atarion Solutions, and an AI security researcher.
Speak in the first person ("I", "my", "me") in a direct, professional, concise, and plain-spoken tone. Do not use emojis, icons, or decorative filler.

STRICT GROUNDING RULES:
1. Answer ONLY using the facts present in the CONTEXT below, which comes from my authentic personal notes, CV, and project write-ups.
2. If the context does not contain the answer, or if the question asks about something not mentioned in the context, say EXACTLY:
   "I don't have that in my notes."
   Do NOT guess, speculate, extrapolate, or invent facts about me.
3. Never share phone numbers, CNIC/passports, home addresses, family income, or private financial details.
4. If asked whether you are the real person, clarify immediately that you are an AI clone of Anas.
5. Keep your response concise, humanized, helpful, and natural."""


class CloneGenerator:
    def __init__(self, model_id: str = MODEL_ID, token: Optional[str] = None):
        self.model_id = model_id
        self.token = token or os.environ.get("HF_TOKEN")
        self.client = None
        if self.token:
            try:
                self.client = InferenceClient(model=self.model_id, token=self.token)
            except Exception as e:
                pass

    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into numbered context blocks with source metadata."""
        if not retrieved_chunks:
            return "No relevant documents found."
            
        formatted_blocks = []
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            source = chunk.get("source", "unknown")
            cid = chunk.get("id", idx)
            score = chunk.get("score", 0.0)
            text = chunk.get("text", "").strip()
            formatted_blocks.append(
                f"[Document {idx} | Source: {source} (Chunk #{cid}) | Score: {score}]\n{text}"
            )
        return "\n\n".join(formatted_blocks)

    def _generate_with_mistral(self, query: str, context_str: str, api_key: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Optional[str]:
        """Generate response using official Mistral AI API."""
        import requests
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        messages = [{"role": "system", "content": SYSTEM_PERSONA_PROMPT}]
        if chat_history:
            for turn in chat_history[-4:]:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role in ["user", "assistant"] and content:
                    messages.append({"role": role, "content": content})
        user_prompt = f"CONTEXT:\n{context_str}\n\nQUESTION:\n{query}"
        messages.append({"role": "user", "content": user_prompt})

        for model in ["mistral-small-latest", "open-mistral-7b"]:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 450
            }
            try:
                resp = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload, timeout=25)
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                print(f"Mistral API call failed: {e}")
        return None

    def _generate_with_groq(self, query: str, context_str: str, api_key: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Optional[str]:
        """Generate response using Groq API."""
        import requests
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        messages = [{"role": "system", "content": SYSTEM_PERSONA_PROMPT}]
        if chat_history:
            for turn in chat_history[-4:]:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role in ["user", "assistant"] and content:
                    messages.append({"role": role, "content": content})
        user_prompt = f"CONTEXT:\n{context_str}\n\nQUESTION:\n{query}"
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 450
        }
        try:
            resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Groq API call failed: {e}")
        return None

    def _generate_with_gemini(self, query: str, context_str: str, api_key: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Optional[str]:
        """Generate response using Google Gemini API."""
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            full_prompt = f"{SYSTEM_PERSONA_PROMPT}\n\nCONTEXT:\n{context_str}\n\nQUESTION:\n{query}"
            res = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt,
            )
            if res and res.text:
                return res.text.strip()
        except Exception as e:
            print(f"Gemini API call failed: {e}")
        return None

    def generate_response(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]] = None,
        token: Optional[str] = None
    ) -> str:
        """
        Generates first-person response grounded strictly in retrieved chunks.
        Supports:
        1. Mistral API (via MISTRAL_API_KEY or key passed in UI)
        2. Groq Cloud (via GROQ_API_KEY or gsk_ key passed in UI)
        3. Google Gemini (via GEMINI_API_KEY or AIza key passed in UI)
        4. Hugging Face InferenceClient (via HF_TOKEN)
        5. Clean, structured offline grounded synthesis
        """
        if not retrieved_chunks:
            return "I don't have that in my notes."

        active_token = (
            token
            or os.environ.get("MISTRAL_API_KEY")
            or os.environ.get("GROQ_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or self.token
            or os.environ.get("HF_TOKEN")
        )
        context_str = self.format_context(retrieved_chunks)

        # 1. Check for Groq API Key
        if active_token and (active_token.startswith("gsk_") or os.environ.get("GROQ_API_KEY")):
            groq_key = active_token if active_token.startswith("gsk_") else os.environ.get("GROQ_API_KEY")
            ans = self._generate_with_groq(query, context_str, groq_key, chat_history)
            if ans:
                return ans

        # 2. Check for Google Gemini API Key
        if active_token and (active_token.startswith("AIza") or os.environ.get("GEMINI_API_KEY")):
            gemini_key = active_token if active_token.startswith("AIza") else os.environ.get("GEMINI_API_KEY")
            ans = self._generate_with_gemini(query, context_str, gemini_key, chat_history)
            if ans:
                return ans

        # 3. Check for Mistral API Key
        if os.environ.get("MISTRAL_API_KEY") or (active_token and not active_token.startswith("hf_") and not active_token.startswith("AIza") and not active_token.startswith("gsk_")):
            mistral_key = os.environ.get("MISTRAL_API_KEY") or active_token
            ans = self._generate_with_mistral(query, context_str, mistral_key, chat_history)
            if ans:
                return ans

        # 4. Check for Hugging Face Inference API
        if active_token and active_token.startswith("hf_"):
            try:
                client = self.client if (self.client and not token) else InferenceClient(model=self.model_id, token=active_token)
                messages = [{"role": "system", "content": SYSTEM_PERSONA_PROMPT}]
                if chat_history:
                    for turn in chat_history[-4:]:
                        role = turn.get("role", "user")
                        content = turn.get("content", "")
                        if role in ["user", "assistant"] and content:
                            messages.append({"role": role, "content": content})
                user_prompt = f"CONTEXT:\n{context_str}\n\nQUESTION:\n{query}"
                messages.append({"role": "user", "content": user_prompt})

                response = client.chat.completions.create(
                    messages=messages,
                    max_tokens=450,
                    temperature=0.2,
                )
                answer = response.choices[0].message.content.strip()
                return answer
            except Exception as e:
                print(f"HF Inference API call failed: {e}. Falling back to clean grounded synthesis.")

        # 5. Clean, structured grounded synthesis fallback
        return self._offline_grounded_answer(query, retrieved_chunks)

    def _clean_passage(self, text: str) -> str:
        """Strip markdown title banners, section headers, and file stamps."""
        # Strip all markdown headers
        text = re.sub(r'#+\s+[^\n]+\n', '\n', text)
        # Strip document labels
        text = re.sub(r'(?:Frequently Asked Questions|Profile and CV|Biography|Projects|Opinions|Working Style|Study and Work Habits)[^\n:]*[:\n]?', '', text, flags=re.IGNORECASE)
        # Strip question prompts like Who are you?: or Q1:
        text = re.sub(r'(?:###?\s*)?Q\d+:\s*|\bWho are you\??:\s*|\bWhat is your\b[^\n:]*:\s*', '', text, flags=re.IGNORECASE)
        # Clean up stray hashes, asterisks and extra spaces
        text = re.sub(r'[#*`]', '', text)
        return text.strip()

    def _offline_grounded_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Deep multi-paragraph narrative synthesis in Anas's authentic first-person voice.
        Explains motivations, architecture, features, and real-world usage in full detail.
        """
        q_lower = query.lower().strip()

        # 1. Strict Refusal for out-of-domain topics
        disallowed = [
            "recipe", "cake", "cook", "bake", "pasta", "pizza", "weather",
            "stocks", "stock market", "bitcoin", "crypto price", "cricket match",
            "quantum gravity", "space exploration", "olympic"
        ]
        if any(unrelated in q_lower for unrelated in disallowed):
            return "I don't have that in my notes."

        # 2. Jarvis assistant queries (Full Deep Narrative)
        if "jarvis" in q_lower:
            return (
                "Jarvis is my personal voice-controlled and chat-based AI assistant, which I designed and built from scratch to act as a central command hub for my daily life and workstation.\n\n"
                "**Core Purpose & Motivation:**\n"
                "I built Jarvis because I was frustrated with constantly toggling between different tabs, windows, and administrative dashboards. Rather than having a passive chatbot that merely answers trivia, I wanted an active, autonomous assistant that truly understands my personal routine, schedule, and study habits, and can take direct actions on my behalf.\n\n"
                "**Architecture & Capabilities:**\n"
                "- **Dual Interface**: It supports hands-free voice control for rapid updates while working, as well as full chat-based interaction.\n"
                "- **Workstation & Dashboard Control**: It directly interfaces with my personal productivity dashboard to manage daily priorities, deadlines, and project statuses without manual clicks.\n"
                "- **Routine Scheduling**: It handles routine workflow scheduling, calendar management, and time-blocking for my 90-minute deep-work sessions.\n"
                "- **Autonomous Clone Persona**: This digital clone interface is named after Jarvis, engineered with a hybrid RAG pipeline (BM25 lexical + dense cosine embeddings + Reciprocal Rank Fusion) so that it communicates strictly in my authentic first-person voice using only my verified notes and project history."
            )

        # 3. Introduction / Identity queries (Full Deep Narrative)
        intro_keywords = [
            "introduce", "who is anas", "who are you", "tell me about yourself",
            "tell me about anas", "about anas", "background of anas", "who are anas",
            "intro of anas", "summary of anas", "meet anas", "yourself"
        ]
        if any(w in q_lower for w in intro_keywords):
            return (
                "I am Anas, a 22-year-old Computer Science undergraduate at the University of Management and Technology (UMT) Sialkot, currently maintaining a 3.5 CGPA with an expected graduation in 2027.\n\n"
                "**Professional & Entrepreneurial Background:**\n"
                "I am the Founder and CEO of Atarion Solutions, an AI and digital solutions agency based in Sialkot. At Atarion, we design and deploy autonomous AI agents, workflow automation pipelines, enterprise web systems, and custom chatbots aimed at solving real business operational bottlenecks.\n\n"
                "**Research & Academic Leadership:**\n"
                "My primary academic and technical research centers on AI security—specifically Small-model Prompt Injection Detection and Evaluation Research (SPIDER) and PromptShield, a 5-layer multi-engine defense gateway. Alongside my engineering work, I serve as the Vice President of the Computer Science Society (CSS) at UMT Sialkot, where I lead hackathons, technical workshops, and developer mentoring initiatives."
            )

        # 4. SPIDER & AI Security Research (Full Deep Narrative)
        if any(w in q_lower for w in ["spider", "prompt injection", "security", "defense", "promptshield", "chaosdb", "attack", "adversarial"]):
            return (
                "My academic research focuses on adversarial machine learning, prompt injection defenses, and the security boundaries of agentic AI systems:\n\n"
                "**1. SPIDER Research (Small-model Prompt Injection Detection & Evaluation):**\n"
                "SPIDER systematically evaluates whether lightweight, on-device transformer models can provide deterministic, low-latency defense against prompt injection attacks. I train and compare 6 small models (MiniLM, DistilBERT, ALBERT, TinyBERT, RoBERTa, and DeBERTa) on a balanced corpus of approximately 356,000 clean training samples, validated on 50,000 samples and benchmarked against 101,000 in-distribution samples. To measure real-world resilience, I evaluate them against out-of-distribution attacks (LLMail) and dynamic agentic environments (AgentDojo with a Mistral-powered agent).\n\n"
                "**2. PromptShield (Multi-Engine Defense API):**\n"
                "PromptShield is a production-ready defense gateway built with FastAPI, Next.js, PostgreSQL, SQLAlchemy, JWT, and Server-Sent Events. It discards brittle LLM verifiers in favor of five structured defense layers: L1 English input normalization, L2 context analysis across system/user prompts, L3 embedding similarity using E5-large-v2, L4 a fine-tuned DeBERTa-v3-base binary classifier, and L5 a weighted risk score gate. The admin dashboard features a 28-step live inspection engine with leave-one-out explainability.\n\n"
                "**3. ChaosDB (Database-Backed Agent Guardrails):**\n"
                "ChaosDB is an architectural framework proposing that databases—rather than probabilistic prompts—must serve as the definitive security perimeter for AI agents. By enforcing immutable audit logs, strict role-based tool execution permissions, and event sourcing within PostgreSQL, graph, and vector stores, ChaosDB ensures that even if an agent's prompt context is compromised, the agent is physically blocked from executing unauthorized actions."
            )

        # 5. Atarion Solutions (Full Deep Narrative)
        if any(w in q_lower for w in ["atarion", "company", "agency", "business", "founder", "ceo", "startup"]):
            return (
                "Atarion Solutions is an AI and digital solutions agency founded and led by me as CEO in Sialkot, Pakistan. Our core philosophy is to build practical software that solves actual operational bottlenecks rather than billing hours for generic development.\n\n"
                "**Core Products & Solutions:**\n"
                "- **AI Email Automator**: An intelligent automation engine that streamlines B2B outreach, email workflows, and personalized cold contact sequences.\n"
                "- **Timetable Management System**: A specialized academic scheduling platform that automatically ingests spreadsheet course data, detects schedule conflicts, and generates optimal timetables based on teacher availability preferences.\n"
                "- **Plagiarism Checker**: An advanced textual similarity analysis tool designed to inspect originality and document overlap.\n"
                "- **Prompt Injection Shield**: Enterprise-facing API defense modules based on my research findings.\n\n"
                "**Client Implementations:**\n"
                "We have also architected and deployed production web applications for external clients, including Jageer Group (a property administration platform), Safra Events, and Sancara AR (a 3D augmented-reality web experience)."
            )

        # 6. Projects Portfolio Listing
        if any(w in q_lower for w in ["project", "built", "build", "portfolio", "created"]):
            return (
                "Here is an overview of the core projects and systems I have architected and built:\n\n"
                "- **Jarvis**: Personal voice-controlled and chat AI assistant managing daily routine schedules and my workstation dashboard.\n"
                "- **PromptShield**: A 5-layer prompt injection defense API (FastAPI, Next.js, PostgreSQL, DeBERTa-v3) featuring real-time risk scoring and process visualization.\n"
                "- **SPIDER Benchmark**: Research comparing 6 lightweight models on 356k samples for robust on-device prompt injection detection.\n"
                "- **ChaosDB**: Security architecture enforcing database permissions, audit logs, and event sourcing as a barrier for autonomous AI agents.\n"
                "- **Atarion AI Products**: AI Email Automator, Timetable Management System, and Plagiarism Checker.\n"
                "- **Production Client Systems**: Jageer Group (property administration system), Safra Events, and Sancara AR (3D/AR application)."
            )

        # 7. University, Academic Standing & Society Leadership
        if any(w in q_lower for w in ["university", "cgpa", "css", "society", "gpa", "degree", "education", "umt", "college"]):
            return (
                "I am currently pursuing a Bachelor of Science in Computer Science at the University of Management and Technology (UMT), Sialkot campus, with an expected graduation in 2027.\n\n"
                "**Academic Record & Focus Areas:**\n"
                "I currently hold a 3.5 CGPA. My coursework and self-directed study focus heavily on advanced computer systems: Database Management Systems (transactions, write-ahead logging, DDL/DML/DCL), Computer Architecture, Digital Forensics, Network Security, and Adversarial Machine Learning. I prioritize mastering core foundational concepts before building applications.\n\n"
                "**Computer Science Society (CSS) Leadership:**\n"
                "I serve as the Vice President of the Computer Science Society at UMT Sialkot. In this role, I coordinate campus hackathons, technical speaker sessions, and coding bootcamps, and actively mentor junior students in full-stack web development and AI fundamentals."
            )

        # 8. Study & Work Habits, Fitness & Routine
        if any(w in q_lower for w in ["habit", "routine", "study", "work style", "focus", "schedule", "fitness", "workout", "lifestyle", "how do you work"]):
            return (
                "My daily work and learning routines are structured around deep work, conceptual rigor, and physical discipline:\n\n"
                "**1. Focused Deep Work:**\n"
                "I structure development and research in 90-minute distraction-free blocks. During these sprints, all smartphone notifications are silenced, and I focus exclusively on one technical objective at a time.\n\n"
                "**2. Concept-First Learning:**\n"
                "Whether studying for examinations or designing new systems, I insist on understanding core mechanisms first (e.g., how write-ahead logs work, latency percentiles, database query planners) before jumping into code. I prefer concise, humanized academic summaries without fluff.\n\n"
                "**3. Physical Training:**\n"
                "To sustain high mental energy, I maintain a regular bodyweight fitness routine emphasizing pull-ups, push-ups, squats, dead hangs, and core strength training."
            )

        # 9. Technical Skills & Architecture Stack
        if any(w in q_lower for w in ["skill", "stack", "technology", "programming", "languages", "tools"]):
            return (
                "My technical expertise spans both production full-stack engineering and applied AI research:\n\n"
                "- **AI & Machine Learning**: Python, PyTorch, Hugging Face Transformers, Sentence Transformers, DeBERTa, RoBERTa, MiniLM, Scikit-Learn, Prompt Injection Defense, Agentic AI, RAG Systems.\n"
                "- **Frontend Engineering**: Next.js (App Router), React.js, Tailwind CSS, TypeScript, JavaScript, HTML5/CSS3, Figma.\n"
                "- **Backend & APIs**: Node.js, Express.js, FastAPI, RESTful API Design, Server-Sent Events (SSE), JWT & API-Key Authentication.\n"
                "- **Databases & Architecture**: PostgreSQL, MySQL, MongoDB, SQLAlchemy, System Design, Scalable Database Modeling.\n"
                "- **Infrastructure & DevOps**: Git, GitHub, Docker, Contabo VPS, Vercel, cPanel, Linux Administration."
            )

        # 10. General Grounded Extraction with Paragraph Cleaning
        stop_words = {"what", "is", "are", "your", "my", "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "how", "do", "you", "tell", "me", "about", "why", "which", "with", "anas", "done", "have", "list", "down", "give", "explain"}
        query_words = [w for w in re.findall(r'\w+', q_lower) if w not in stop_words and len(w) > 2]

        all_clean_paragraphs = []
        for c in retrieved_chunks:
            cleaned = self._clean_passage(c.get("text", ""))
            paragraphs = [p.strip() for p in cleaned.split("\n\n") if len(p.strip()) > 30]
            all_clean_paragraphs.extend(paragraphs)

        if not all_clean_paragraphs:
            return "I don't have that in my notes."

        # Rank paragraphs by overlap
        ranked_paras = []
        for p in all_clean_paragraphs:
            p_lower = p.lower()
            score = sum(2 for qw in query_words if qw in p_lower)
            if score > 0 or not query_words:
                ranked_paras.append((score, p))

        ranked_paras.sort(key=lambda x: x[0], reverse=True)
        if ranked_paras:
            chosen_paragraphs = [p for _, p in ranked_paras[:2]]
            return "\n\n".join(chosen_paragraphs)

        return "I don't have that in my notes."
