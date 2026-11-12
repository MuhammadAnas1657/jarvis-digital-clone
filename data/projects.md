# Projects - Anas

## Jarvis
Jarvis is my personal AI assistant. It is both voice-controlled and chat-based. It schedules my routine tasks and has control of my dashboard, so I can manage my day by talking or typing to it instead of clicking through screens. I built it because I wanted one assistant that understands my routine and acts on it rather than only answering questions. This digital clone interface is also named after it.

## Atarion Solutions products
Atarion Solutions is an AI and digital solutions company in Sialkot where I am Founder and CEO. It offers AI automation, AI agents, AI chatbots, email outreach automation, marketing automation, web and software development, business process automation and custom AI solutions. The goal is to solve real operational problems, not to sell development hours.
- AI Email Automator: automates email workflows and outreach.
- Timetable Management System: academic scheduling and timetable generation.
- Plagiarism Checker: checks textual similarity and originality.
- Prompt Injection Shield: detects and mitigates prompt injection attacks.

## PromptShield (Prompt Guard multi-engine API)
A prompt injection defense API built with FastAPI, Next.js, PostgreSQL, SQLAlchemy, JWT, API keys and server-sent events. It uses five layers: L1 input normalization (English only), L2 system, user and external context analysis, L3 embedding similarity with E5-large-v2 (DeBERTa-v3-base also explored), L4 a DeBERTa-v3-base binary classifier, and L5 a weighted manual risk gate. I removed the LLM verifier in favour of structured layers. The dashboard shows a 28-step live process with leave-one-out explainability.

## SAID v2 dataset
Training set of 280,026 samples (144,265 class 0 and 135,761 class 1) and a validation set of 30,285. Mean length is about 389 characters or 82 tokens, and only 1.91 percent exceed 256 tokens. I trained DeBERTa-v3-base with max length 256, batch size 16, 3 epochs and learning rate 3e-6.

## SPIDER research
SPIDER stands for Small-model Prompt Injection Detection and Evaluation Research. I compare six small models (MiniLM, DistilBERT, ALBERT, TinyBERT, RoBERTa, DeBERTa) on about 356k clean balanced training samples, 50k validation and 101k in-distribution test samples, with LLMail for out-of-distribution evaluation and AgentDojo with a Mistral agent for agent evaluation. The aim is on-device detection baselines and honest robustness measurements.

## ChaosDB
ChaosDB is an advanced DBMS project proposal on using a database as the security layer for AI agents. Even if an attacker manipulates an agent through prompt injection, the database-backed layer (PostgreSQL, graph and vector stores, event sourcing, immutable audit logs, tool permissions) stops the agent from doing anything it is not authorized to do.

## Timetable Management System design
I want it simple and functional. Users pick a program, batch and semester, paste course data from a spreadsheet, and the system detects the columns automatically. Teachers enter preferred days and time slots on a preferences page. I also require debouncing, fewer API calls, authentication, authorization, input validation and rate limiting. I test real workflows such as adding a student or column, not just opening pages.
