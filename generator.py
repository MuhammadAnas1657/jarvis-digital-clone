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
        """Strip markdown title banners, redundant section headers and file stamps."""
        text = re.sub(r'#+\s*(?:Biography|Profile and CV|Projects|Opinions|Frequently Asked Questions|Working Style).*?\n', '', text, flags=re.IGNORECASE)
        text = re.sub(r'###?\s*Q\d+:\s*([^\n?]+\??)', r'\1: ', text)
        text = re.sub(r'##+\s*', '', text)
        return text.strip()

    def _offline_grounded_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Clean, structured offline synthesis when API token is not yet provided or unreachable.
        Ensures persona voice, honest refusals, and clean formatting.
        """
        q_lower = query.lower().strip()
        
        # Check if asked about identity
        if any(w in q_lower for w in ["who are you", "what is your name", "are you the real", "are you real"]):
            return "I am JARVIS, the AI clone of Anas. I answer strictly from his verified notes, projects and documents."

        # Check for out-of-domain topics
        if any(unrelated in q_lower for unrelated in ["recipe", "cake", "cook", "bake", "weather", "stocks", "bitcoin", "cricket match"]):
            return "I don't have that in my notes."

        stop_words = {"what", "is", "are", "your", "my", "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "how", "do", "you", "tell", "me", "about", "why", "which", "with", "anas", "done", "have", "list", "down"}
        query_words = [w for w in re.findall(r'\w+', q_lower) if w not in stop_words and len(w) > 2]

        all_context_text = " ".join([c.get("text", "").lower() for c in retrieved_chunks])
        has_lexical_match = any(qw in all_context_text for qw in query_words) if query_words else False

        dense_scores = [c.get("dense_score", 0.0) for c in retrieved_chunks]
        if not dense_scores and retrieved_chunks:
            dense_scores = [c.get("score", 0.0) for c in retrieved_chunks if c.get("retrieval_mode") == "Dense"]
        max_dense = max(dense_scores) if dense_scores else 0.0

        if not has_lexical_match and max_dense < 0.35:
            return "I don't have that in my notes."

        # Handle specific common queries cleanly in first-person voice:
        # 1. Projects listing
        if "project" in q_lower:
            projects_summary = [
                "Here are the key projects I have worked on:",
                "- **Jarvis**: My personal voice-controlled and chat AI assistant that schedules routine tasks and controls my personal dashboard.",
                "- **PromptShield**: A 5-layer prompt injection defense API (FastAPI, Next.js, PostgreSQL) utilizing input normalization, context analysis, and DeBERTa-v3 classification.",
                "- **SPIDER**: Small-model Prompt Injection Detection and Evaluation Research, evaluating 6 lightweight models on over 356k samples for on-device defense.",
                "- **ChaosDB**: An advanced database-backed security layer for AI agents ensuring strict tool permissions and immutable audit logs.",
                "- **Atarion Solutions Products**: AI Email Automator, Timetable Management System, and Plagiarism Checker.",
                "- **Client Systems**: Jageer Group (property administration system), Safra Events, and Sancara AR (3D/AR application)."
            ]
            return "\n".join(projects_summary)

        # 2. SPIDER & AI security research
        if "spider" in q_lower or "prompt injection" in q_lower or ("security" in q_lower and "research" in q_lower):
            return (
                "My primary research focuses on AI security, specifically prompt injection detection, LLM security, and agentic AI protection.\n\n"
                "- **SPIDER** (Small-model Prompt Injection Detection and Evaluation Research): Compares six lightweight models (MiniLM, DistilBERT, ALBERT, TinyBERT, RoBERTa, DeBERTa) on ~356k clean training samples and evaluates them on out-of-distribution benchmarks (LLMail) and agent environments (AgentDojo).\n"
                "- **PromptShield**: A multi-engine defense pipeline using 5 layers (input normalization, context analysis, embedding similarity, DeBERTa binary classification, and a weighted risk gate).\n"
                "- **ChaosDB**: Explores using database constraints, tool permissions, and immutable audit logs as a foundational security boundary for AI agents."
            )

        # 3. University / CGPA / Society Role
        if any(w in q_lower for w in ["university", "cgpa", "css", "society", "gpa"]):
            return "I am currently pursuing my BS in Computer Science at the University of Management and Technology (UMT), Sialkot (expected graduation 2027), with a CGPA of 3.5. I also serve as the Vice President of the Computer Science Society at UMT Sialkot."

        # 4. Atarion Solutions
        if "atarion" in q_lower:
            return "Atarion Solutions is my AI and digital solutions company based in Sialkot, where I serve as Founder and CEO. We build AI automation pipelines, autonomous agents, chatbots, email outreach automation, and custom software systems designed to solve real operational business challenges."

        # 5. General grounded extraction with clean sentence formatting
        raw_sentences = []
        for c in retrieved_chunks:
            cleaned_text = self._clean_passage(c.get("text", ""))
            for s in re.split(r"(?<=[.!?])\s+|\n+", cleaned_text):
                s = s.strip(" #-*")
                if len(s) > 25 and not s.lower().startswith("profile and cv") and not s.lower().startswith("biography"):
                    overlap = sum(1 for w in query_words if w in s.lower())
                    raw_sentences.append((overlap, s))

        raw_sentences.sort(key=lambda x: x[0], reverse=True)
        chosen = []
        seen = set()
        for overlap, s in raw_sentences:
            normalized = s.lower()[:40]
            if normalized not in seen and overlap > 0:
                seen.add(normalized)
                chosen.append(s)
            if len(chosen) >= 3:
                break

        if chosen:
            return " ".join(chosen)

        return "I don't have that in my notes."
