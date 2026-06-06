# 🛡️ TruthBot: AI-Powered Fake News Detection Telegram Chatbot

TruthBot is a Telegram chatbot that helps users verify the authenticity of news claims and combat misinformation. It uses a **Hybrid RAG pipeline** — combining a local ChromaDB vector store, live Sebenarnya.my scraping, and the Google Fact Check API — all orchestrated by LangChain and synthesised by a single LLM call to **DeepSeek V4 Flash via NVIDIA NIM**.

This project was developed as a Final Year Project (FYP) for the Bachelor of Software Engineering programme at UNIMAS.

---

## 🎯 Objectives

- Combat the spread of misinformation and fake news in Malaysia.
- Provide fast, evidence-backed fact-checking through Telegram.
- Apply Hybrid RAG with weighted multi-source retrieval for higher accuracy.
- Support both English and Bahasa Melayu claims.

---

## 🏗️ System Architecture

```
┌──────────────────────┐
│    Telegram User     │
└──────────┬───────────┘
           │ claim text
           ▼
┌──────────────────────┐
│  chatbot_telegram.py │  (python-telegram-bot v20+)
│  verify_claim()      │
└──────┬───────────────┘
       │
       ├─────────────────────────────────────────┐
       │  ThreadPoolExecutor (parallel)           │
       ▼                                         ▼
┌─────────────────┐                 ┌────────────────────────┐
│ LangChain Chroma│                 │ Google Fact Check API  │
│ (ChromaDB)      │                 │ weight = 0.5           │
│ weight = 0.2    │                 └────────────┬───────────┘
│ top-3 chunks    │                              │
│ threshold ≥ 20% │                 ┌────────────▼───────────┐
└────────┬────────┘                 │ Live Sebenarnya.my     │
         │                         │ scraper (BeautifulSoup)│
         │                         │ weight = 0.3           │
         │                         │ top-2 articles         │
         └─────────────┬───────────┘
                       │ all evidence combined into one prompt
                       ▼
         ┌─────────────────────────────┐
         │   LangChain ChatOpenAI      │
         │   → NVIDIA NIM              │
         │   → DeepSeek V4 Flash       │
         │   temperature = 0.2         │
         │   max_tokens = 300          │
         └──────────────┬──────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │  Verdict + Confidence        │
         │  TRUE / FALSE /              │
         │  MISLEADING / UNVERIFIED     │
         └──────────────────────────────┘
```

---

## 🚀 Features

### ✅ Hybrid RAG Pipeline
Three independent sources are queried in parallel and their evidence is merged into a single LLM prompt — no chained calls, no guessing.

### ✅ Weighted Evidence Scoring
Each source contributes to the final verdict with a calibrated weight:

| Source | Weight | Rationale |
|---|---|---|
| Google Fact Check API | 0.50 | Authoritative verdicts from multiple independent publishers |
| Live Sebenarnya.my | 0.30 | Malaysia's official MCMC fact-checking portal, queried live |
| Local ChromaDB (RAG) | 0.20 | Offline fallback; useful but may lag behind recent articles |

### ✅ 4-Level Verdict System
Every response includes a verdict and a confidence score:
- 🟢 **TRUE** — evidence strongly supports the claim
- 🔴 **FALSE** — evidence strongly contradicts the claim
- 🟡 **MISLEADING** — evidence is mixed or partially contradicts
- ⚪ **UNVERIFIED** — insufficient evidence found across all sources

### ✅ Bilingual Support
Language is auto-detected from the user's input. TruthBot responds in **English** or **Bahasa Melayu** accordingly.

### ✅ Telegram Integration
No additional app required. Users interact via Telegram and receive responses with sources.

### ✅ Graceful Degradation
If ChromaDB is unavailable, the bot continues with live retrieval only. If the LLM hits a rate limit, a user-friendly fallback message is sent.

---

## 📂 Project Structure

```
TruthBot/
│
├── chatbot_telegram.py          # Main bot — Hybrid RAG pipeline + Telegram handlers
├── scraper.py                   # Scrapes Sebenarnya.my → sebenarnya_articles.json
├── build_vectordb.py            # Chunks articles → generates embeddings → populates ChromaDB
├── requirements.txt             # Python dependencies
│
├── sebenarnya_articles.json     # Scraped dataset (primary)
├── sebenarnya_articles_backup.json  # Backup copy of scraped dataset
│
├── chroma_db/                   # Persistent ChromaDB vector store
│   └── chroma.sqlite3
│
├── README.md
└── .gitignore
```

---

## 📊 Dataset

**Source:** [Sebenarnya.my](https://www.sebenarnya.my) — Malaysia's official fact-checking platform managed by the Malaysian Communications and Multimedia Commission (MCMC).

Each article contains a news headline, the claim under investigation, a fact-check verdict, a detailed explanation, publication date, category tags, and the source URL.

The scraper (`scraper.py`) crawls Sebenarnya.my and saves results to `sebenarnya_articles.json`. The vector database builder (`build_vectordb.py`) then chunks and embeds these articles into ChromaDB for offline retrieval.

| Component | Notes |
|---|---|
| Embedding model | `all-MiniLM-L6-v2` via HuggingFace |
| Vector store | ChromaDB (local, persistent) |
| Similarity threshold | ≥ 20% cosine similarity |
| Max chunks per query | Top 3 |

---

## 🤖 Technology Stack

| Component | Technology |
|---|---|
| Telegram bot | `python-telegram-bot >= 20.0` |
| LLM orchestration | LangChain (`langchain`, `langchain-core`, `langchain-openai`) |
| LLM | DeepSeek V4 Flash via NVIDIA NIM (`integrate.api.nvidia.com/v1`) |
| Embeddings | `langchain-huggingface` + `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB via `langchain-chroma` |
| Live scraping | `requests` + `beautifulsoup4` |
| Fact-check API | Google Fact Check Tools API |
| Parallelism | `concurrent.futures.ThreadPoolExecutor` |
| Environment | `python-dotenv` |

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/zfrtlsofea/TruthBot.git
cd TruthBot
```

### 2. Create a virtual environment

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration

Create a `.env` file in the project root:

```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
NVIDIA_API_KEY=your_nvidia_nim_api_key
GOOGLE_FACT_CHECK_API_KEY=your_google_fact_check_api_key

# Optional overrides (defaults shown)
NIM_MODEL=deepseek-ai/deepseek-v4-flash
NIM_API_BASE=https://integrate.api.nvidia.com/v1
```

> **Note:** `NVIDIA_API_KEY` and `TELEGRAM_BOT_TOKEN` are required. `GOOGLE_FACT_CHECK_API_KEY` is optional — the bot will skip Google Fact Check if it is not set.

---

## 📥 Build the Dataset

Run the scraper to collect articles from Sebenarnya.my:

```bash
python scraper.py
```

This crawls the site, extracts fact-checking articles, and saves them to `sebenarnya_articles.json`.

---

## 🧠 Generate Embeddings

Build the ChromaDB vector database:

```bash
python build_vectordb.py
```

This loads the scraped articles, generates embeddings using `all-MiniLM-L6-v2`, and stores them in `./chroma_db/`.

> **Skip this step?** TruthBot will still work using live Sebenarnya.my scraping and Google Fact Check — local RAG will simply be disabled.

---

## ▶️ Run TruthBot

```bash
python chatbot_telegram.py
```

On success you will see:

```
TruthBot is running. Press Ctrl+C to stop.
About to start polling...
```

---

## 📱 Using the Bot

Open Telegram and search for **@zs_truth_bot**.

### Commands

| Command | Description |
|---|---|
| `/start` | Welcome message and usage instructions |
| `/help` | List all commands |
| `/sources` | Show sources and pipeline details |
| `/tips` | Tips for spotting fake news |
| `/reset` | Clear your conversation history |

### Example queries

```
Government giving RM5000 to all citizens
Is this viral WhatsApp message true?
Kerajaan umum bantuan RM1000 untuk semua rakyat
Did the Health Ministry confirm this outbreak?
```

---

## 🔍 Example Response

**User input:**
```
Government giving RM5000 assistance to all citizens.
```

**TruthBot response:**
```
🔍 Verdict: FALSE

📊 Confidence: 85%

📋 Explanation:
Based on Sebenarnya.my and Google Fact Check sources, no official
announcement matching this claim has been made. Government channels
deny the claim. Similar misinformation has been fact-checked and
debunked previously.

🔗 Sources:
https://www.sebenarnya.my/...
https://factcheck.afp.com/...
```

---

## 🎓 Academic Context

| | |
|---|---|
| **Project Title** | TruthBot: AI-Based Telegram Chatbot for Fake News Detection |
| **Programme** | Bachelor of Software Engineering |
| **Institution** | Faculty of Computer Science and Information Technology (FCSIT), Universiti Malaysia Sarawak (UNIMAS) |

---

## 👩‍💻 Author

**Zafiratul Sofea**  
Final Year Software Engineering Student, UNIMAS  
GitHub: [https://github.com/zfrtlsofea](https://github.com/zfrtlsofea)

---

## 📄 License

This project is developed for educational and research purposes.
