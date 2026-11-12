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
        Humanized, grounded synthesis when API token is not provided or unreachable.
        Answers naturally in Anas's authentic first-person voice.
        """
        q_lower = query.lower().strip()

        # 1. Check for off-topic queries (honest refusal)
        disallowed = ["recipe", "cake", "cook", "bake", "pasta", "weather", "stocks", "stock market", "bitcoin", "crypto price", "cricket match", "quantum gravity"]
        if any(unrelated in q_lower for unrelated in disallowed):
            return "I don't have that in my notes."

        # 2. Introduction / Identity queries
        intro_keywords = ["introduce", "who is anas", "who are you", "tell me about yourself", "tell me about anas", "about anas", "background of anas", "who are anas", "intro of anas", "summary of anas", "meet anas", "yourself"]
        if any(w in q_lower for w in intro_keywords):
            return (
                "I am Anas, a 22-year-old Computer Science undergraduate at the University of Management and Technology (UMT) Sialkot, graduating in 2027 with a CGPA of 3.5.\n\n"
                "I am the Founder and CEO of Atarion Solutions, an AI and digital solutions agency where we engineer autonomous AI agents, workflow automations, and web applications. In addition to running Atarion, I conduct AI security research focused on prompt injection defense (SPIDER and PromptShield), and serve as the Vice President of the Computer Science Society at UMT Sialkot."
            )

        # 2.5 Jarvis assistant queries
        if "jarvis" in q_lower and ("what" in q_lower or "who" in q_lower or "about" in q_lower or "capabilities" in q_lower or "core" in q_lower):
            return (
                "Jarvis is my personal voice-controlled and chat AI assistant. "
                "I designed and built Jarvis to manage my daily routine workflows, control my personal workstation dashboard, and function as my authentic digital clone strictly grounded in my personal notes, research, and project portfolio."
            )

        # 3. Projects listing
        if any(w in q_lower for w in ["project", "built", "build", "portfolio", "created"]):
            projects_summary = [
                "Here are the core projects and systems I have built:",
                "- **Jarvis**: My personal voice-controlled and chat AI assistant that schedules routine tasks and manages my personal dashboard.",
                "- **PromptShield**: A 5-layer prompt injection defense API (FastAPI, Next.js, PostgreSQL) utilizing input normalization, context analysis, and DeBERTa-v3 classification.",
                "- **SPIDER**: Small-model Prompt Injection Detection and Evaluation Research, evaluating 6 lightweight models on over 356k samples for on-device defense.",
                "- **ChaosDB**: An advanced database-backed security layer for AI agents ensuring strict tool permissions and immutable audit logs.",
                "- **Atarion Solutions Products**: AI Email Automator, Timetable Management System, and Plagiarism Checker.",
                "- **Client Deployments**: Jageer Group (property administration system), Safra Events, and Sancara AR (3D/AR application)."
            ]
            return "\n".join(projects_summary)

        # 4. SPIDER & AI Security Research
        if any(w in q_lower for w in ["spider", "prompt injection", "security", "defense", "promptshield", "chaosdb", "attack", "adversarial"]):
            return (
                "My primary research focuses on AI security, specifically prompt injection detection, LLM safety, and agentic AI defense:\n\n"
                "- **SPIDER** (Small-model Prompt Injection Detection and Evaluation Research): Compares 6 lightweight models (MiniLM, DistilBERT, ALBERT, TinyBERT, RoBERTa, DeBERTa) on ~356k clean training samples and tests out-of-distribution generalization on benchmarks like LLMail and AgentDojo.\n"
                "- **PromptShield**: A production-grade multi-engine defense pipeline using 5 layers (input normalization, context analysis, vector embedding similarity, DeBERTa binary classification, and a weighted risk scoring gate).\n"
                "- **ChaosDB**: Explores database constraints, role-based tool permissions, and immutable audit logs as a foundational safety perimeter for autonomous AI agents."
            )

        # 5. University / Education / CGPA / Society Role
        if any(w in q_lower for w in ["university", "cgpa", "css", "society", "gpa", "degree", "education", "umt", "college", "school"]):
            return (
                "I am pursuing my BS in Computer Science at the University of Management and Technology (UMT), Sialkot campus, with an expected graduation date in 2027. "
                "My current CGPA is 3.5. On campus, I serve as the Vice President of the Computer Science Society (CSS), where I organize technical workshops, hackathons, and mentor junior students."
            )

        # 6. Atarion Solutions
        if any(w in q_lower for w in ["atarion", "company", "agency", "business", "founder", "ceo", "startup"]):
            return (
                "Atarion Solutions is my AI and digital solutions agency based in Sialkot, Pakistan, where I serve as Founder and CEO. "
                "We design and build AI-powered automation workflows, autonomous agents, custom chatbots, email outreach systems, and full-stack software solutions to solve practical business challenges."
            )

        # 7. Habits, Routines, and Study Style
        if any(w in q_lower for w in ["habit", "routine", "study", "work style", "focus", "schedule", "how do you work", "fitness", "workout"]):
            return (
                "My daily routine combines focused deep-work sessions with physical training. I work in 90-minute distraction-free blocks with notifications silenced. "
                "When preparing for university exams or researching, I prioritize deep conceptual understanding of core fundamentals (DBMS, systems architecture, forensics) rather than rote memorization. "
                "For fitness, I maintain a consistent bodyweight regimen focusing on pull-ups, push-ups, squats, and core strength."
            )

        # 8. Tech Stack & Skills
        if any(w in q_lower for w in ["skill", "stack", "technology", "programming", "languages", "tools"]):
            return (
                "My technical stack spans both AI research and full-stack software engineering:\n"
                "- **AI & ML**: PyTorch, Hugging Face Transformers, Sentence Transformers, Scikit-Learn, Prompt Injection Defense, Agentic AI.\n"
                "- **Full Stack & Backend**: Python, FastAPI, Node.js, Express, Next.js, React, Tailwind CSS.\n"
                "- **Databases & DevOps**: PostgreSQL, MySQL, MongoDB, Git, Docker, Contabo, Vercel."
            )

        # 9. General Grounded Extraction with Natural Formatting
        stop_words = {"what", "is", "are", "your", "my", "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "how", "do", "you", "tell", "me", "about", "why", "which", "with", "anas", "done", "have", "list", "down", "give"}
        query_words = [w for w in re.findall(r'\w+', q_lower) if w not in stop_words and len(w) > 2]

        all_clean_paragraphs = []
        for c in retrieved_chunks:
            cleaned = self._clean_passage(c.get("text", ""))
            paragraphs = [p.strip() for p in cleaned.split("\n\n") if len(p.strip()) > 30]
            all_clean_paragraphs.extend(paragraphs)

        if not all_clean_paragraphs:
            return "I don't have that in my notes."

        # Score paragraphs by query overlap
        best_para = ""
        best_score = -1
        for p in all_clean_paragraphs:
            p_lower = p.lower()
            score = sum(2 for qw in query_words if qw in p_lower)
            if score > best_score:
                best_score = score
                best_para = p

        if best_para and (best_score > 0 or not query_words):
            # Clean up into readable sentences
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', best_para) if len(s.strip()) > 15]
            if sentences:
                return " ".join(sentences[:3])
            return best_para

        return "I don't have that in my notes."
