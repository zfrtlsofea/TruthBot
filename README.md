# TruthBot 🤖

**AI-Based Telegram Chatbot for Fake News Detection**

TruthBot is an intelligent, conversational AI system that detects and explains fake news, scams, and misleading information directly within Telegram. It combines **Retrieval-Augmented Generation (RAG)**, **LangChain orchestration**, and **hybrid fact-checking sources** to deliver accurate, evidence-backed verdicts in real-time.

---

## 🎯 Key Features

- **Real-Time Verification** – Instantly detect fake news, scams, and misleading claims
- **Hybrid Fact-Checking Pipeline** – Combines local knowledge base (ChromaDB) + live APIs (Sebenarnya.my, Google Fact Check)
- **LangChain-Powered RAG** – Retrieval-Augmented Generation using HuggingFace embeddings
- **Evidence-Based Responses** – Every verdict includes references and explanations
- **Multilingual Support** – English and Bahasa Melayu
- **Telegram Native** – Works directly in your existing messaging workflow
- **Scalable Architecture** – Built with FastAPI, async processing, and modular design

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRUTHBOT SYSTEM PIPELINE                     │
└─────────────────────────────────────────────────────────────────┘

User's Claim (via Telegram)
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    HYBRID EVIDENCE RETRIEVAL                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. LOCAL RAG (LangChain + ChromaDB)          [Weight: 0.2]     │
│     └─ ~5 most similar chunks from vector DB                    │
│        (multilingual embeddings, 20% similarity threshold)       │
│                                                                  │
│  2. LIVE SEBENARNYA.MY (Web Scraping)         [Weight: 0.3]     │
│     └─ Real-time articles from sebenarnya.my                    │
│        (Malaysia's official MCMC fact-check portal)             │
│                                                                  │
│  3. GOOGLE FACT CHECK API                     [Weight: 0.5]     │
│     └─ Authoritative verdicts from multiple global sources      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│              EVIDENCE SCORING & VERDICT COMPUTATION              │
├─────────────────────────────────────────────────────────────────┤
│  • Analyze ratings from each source                             │
│  • Compute weighted confidence score (-2.0 to +2.0)             │
│  • Map to verdict: TRUE / FALSE / MISLEADING / UNVERIFIED       │
│  • Calculate confidence percentage (0–100%)                      │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│            LLM EXPLANATION GENERATION (DeepSeek V3)              │
├─────────────────────────────────────────────────────────────────┤
│  • Temperature: 0.2 (consistent, factual)                       │
│  • Max tokens: 800                                              │
│  • Model: deepseek-ai/deepseek-v3.2 (via NVIDIA NIM)           │
│  • Format response with emoji indicators & citations            │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
Response to User (Telegram)
🔍 *Verdict:* [TRUE/FALSE/MISLEADING/UNVERIFIED]
📊 *Confidence:* [0–100%]
📋 *Explanation:* [Evidence-backed explanation]
🔗 *Sources:* [Linked references]
⚠️ *Tip:* [Practical prevention tip]
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Telegram account
- API Keys:
  - NVIDIA API Key (DeepSeek LLM access)
  - Telegram Bot Token
  - Google Fact Check API Key (optional but recommended)

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/zfrtlsofea/TruthBot.git
cd TruthBot

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the project root:

```env
NVIDIA_API_KEY=your_nvidia_api_key_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
GOOGLE_FACT_CHECK_API_KEY=your_google_api_key_here
NIM_MODEL=deepseek-ai/deepseek-v3.2
NIM_API_BASE=https://integrate.api.nvidia.com/v1
```

**Getting API Keys:**

- **NVIDIA API Key**: Register at [nvidia.com/ai](https://www.nvidia.com) and get NIM access
- **Telegram Bot Token**: Create bot via [@BotFather](https://t.me/botfather) on Telegram
- **Google Fact Check Key**: Get from [Google Cloud Console](https://console.cloud.google.com/)

### 3. Build Local Knowledge Base

TruthBot uses a ChromaDB vector database of Sebenarnya.my articles for fast, accurate retrieval:

```bash
# Step 1: Scrape Sebenarnya.my (builds sebenarnya_articles.json)
python scraper.py

# Step 2: Build vector database (creates chroma_db/ directory)
python build_vectordb.py

# Expected output:
# ✅ ~2000–5000 articles indexed
# ✅ ~400k–800k text chunks in vector store
# ✅ Multilingual embeddings (English + Bahasa Melayu)
```

### 4. Run the Chatbot

```bash
python chatbot_telegram.py
```

You should see:
```
INFO: Starting TruthBot — LangChain Hybrid RAG pipeline...
INFO: TruthBot is running. Press Ctrl+C to stop.
```

### 5. Test on Telegram

1. Find your bot on Telegram (use the bot token from BotFather)
2. Send any claim or news headline
3. TruthBot will respond with a verdict and evidence!

**Example:**
```
User: "Drinking salt water cures COVID-19"

TruthBot:
🔍 *Verdict:* FALSE
📊 *Confidence:* 98.5%

📋 *Explanation:* Salt water does not cure COVID-19. 
Official health sources and WHO guidelines confirm that 
COVID-19 requires medical treatment, not salt water. 
This claim has been debunked multiple times.

🔗 *Sources Checked:*
• https://sebenarnya.my/...
• https://factchecktools.googleapis.com/...

⚠️ *Tip:* Always verify health claims with official 
sources like WHO or your local health ministry.
```

---

## 📁 Project Structure

```
TruthBot/
├── README.md                      # This file
├── ARCHITECTURE.md                # System design & component details
├── CONTRIBUTING.md                # Contribution guidelines
├── LICENSE                        # MIT License
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment template
│
├── Core Pipeline
├── ├── scraper.py                 # Scrapes sebenarnya.my articles
├── ├── build_vectordb.py          # Builds ChromaDB vector store
├── └── chatbot_telegram.py         # Main Telegram bot with RAG pipeline
│
├── Documentation
├── ├── docs/
├── │   ├── API.md                 # API endpoint documentation
├── │   ├── PROMPTS.md             # LLM prompt engineering details
├── │   ├── EMBEDDING_MODEL.md     # Embedding model info
├── │   └── DEPLOYMENT.md          # Production deployment guide
│
├── Configuration
├── ├── .env.example               # Environment variables template
├── ├── .gitignore                 # Git ignore rules
├── └── .github/
│       └── workflows/
│           └── tests.yml          # CI/CD pipeline
│
└── Data (generated at runtime)
    ├── sebenarnya_articles.json   # Scraped articles dataset
    └── chroma_db/                 # Vector database (ChromaDB)
```

---

## 🔧 Core Components

### 1. **scraper.py** – Data Collector
- Fetches all articles from sebenarnya.my
- Extracts title, date, and full body text
- Skips already-scraped URLs (incremental updates)
- Outputs: `sebenarnya_articles.json` (~2000–5000 articles)

**Run weekly to stay updated:**
```bash
python scraper.py
```

### 2. **build_vectordb.py** – Vector Database Builder
- Converts articles → LangChain Documents
- Splits text using `RecursiveCharacterTextSplitter` (800 chars, 150-char overlap)
- Generates embeddings via HuggingFace (`paraphrase-multilingual-MiniLM-L12-v2`)
- Stores in ChromaDB with metadata (title, URL, date)

**Outputs:** `chroma_db/` directory with ~400k–800k searchable chunks

### 3. **chatbot_telegram.py** – Main Pipeline
Core verification flow:

| Step | Component | Details |
|------|-----------|---------|
| 1 | **Local RAG** | Query ChromaDB, retrieve top 5 similar chunks (20% threshold) |
| 2 | **Live Sebenarnya.my** | Web scrape latest articles matching the claim |
| 3 | **Google Fact Check** | Query authoritative fact-check database |
| 4 | **Evidence Scoring** | Weighted combination of all sources |
| 5 | **Verdict Computation** | Map score to TRUE/FALSE/MISLEADING/UNVERIFIED |
| 6 | **LLM Explanation** | Generate human-readable explanation with citations |
| 7 | **Telegram Response** | Send formatted response with emoji, confidence, sources, and tip |

---

## 🎛️ Configuration & Weights

### Fact-Check Source Weights

```python
WEIGHT_GOOGLE      = 0.5  # Authoritative, multiple sources
WEIGHT_SEBENARNYA  = 0.3  # Trusted local source, may have limited coverage
WEIGHT_LOCAL       = 0.2  # Local RAG, potentially outdated
```

### RAG Parameters

```python
SIMILARITY_THRESHOLD = 0.2      # Only use chunks ≥20% similar
MAX_LOCAL_CHUNKS     = 5        # Top 5 most relevant chunks
MAX_LIVE_ARTICLES    = 3        # Top 3 live Sebenarnya.my articles
CHUNK_SIZE           = 800      # Characters per chunk
CHUNK_OVERLAP        = 150      # Characters to preserve context
```

### LLM Settings

```python
MODEL              = "deepseek-ai/deepseek-v3.2"
TEMPERATURE        = 0.2       # Low = consistent, factual
MAX_TOKENS         = 800       # Response length limit
EMBEDDING_MODEL    = "paraphrase-multilingual-MiniLM-L12-v2"
```

---

## 🧪 Testing

### Manual Testing

1. **Start the bot:**
   ```bash
   python chatbot_telegram.py
   ```

2. **Send test claims:**
   - "Vaccines cause autism" → Should be FALSE
   - "The Earth is round" → Should be TRUE
   - "Coffee increases height" → Should be FALSE/MISLEADING
   - "Some random new claim from TikTok" → May be UNVERIFIED

3. **Verify responses:**
   - Verdict is one of: TRUE, FALSE, MISLEADING, UNVERIFIED
   - Confidence is between 0–100%
   - Explanation references actual sources
   - Tips are relevant to the claim type

### Unit Tests (Recommended)

Create `tests/test_chatbot.py`:
```bash
mkdir -p tests
# Tests will verify: RAG retrieval, scoring logic, LLM generation
```

---

## 📚 Documentation

For detailed information, see:

- **[ARCHITECTURE.md](./ARCHITECTURE.md)** – System design, data flow, component interactions

---

## 🌍 Multilingual Support

TruthBot supports **English** and **Bahasa Melayu** natively:

- **Embedding Model**: `paraphrase-multilingual-MiniLM-L12-v2` (supports 50+ languages)
- **LLM Inference**: Detects user language and responds in kind
- **Dataset**: Mix of English and Malay articles from Sebenarnya.my

---

## ⚙️ Performance Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| Response Time | < 10 seconds | Includes API calls + LLM generation |
| Accuracy | 70–90% | Varies by claim type & evidence availability |
| Confidence Range | 0–100% | Higher = more evidence found |
| Vector DB Size | 400k–800k chunks | Re-scrape weekly for freshness |
| Average Tokens per Response | 200–300 | Under 800-token limit |

---

## 🐛 Troubleshooting

### ChromaDB Not Loading
```
WARNING: ChromaDB not available
Run: python scraper.py → python build_vectordb.py
```

### Slow Responses
- Check internet connection (live API calls)
- Reduce `MAX_LOCAL_CHUNKS` or `MAX_LIVE_ARTICLES`
- Use `SIMILARITY_THRESHOLD = 0.3` for faster retrieval

### Empty Responses
- Verify API keys in `.env`
- Check NVIDIA NIM API status
- Ensure Telegram bot token is valid

### Low Accuracy
- Update knowledge base: `python scraper.py && python build_vectordb.py`
- Check claim language matches dataset language
- Adjust source weights in `chatbot_telegram.py`

---

## 📜 License

This project is built for academic research and responsible AI deployment.

---

## 🙏 Acknowledgments

- **Sebenarnya.my** – Malaysia's official fact-checking portal
- **MCMC** – Malaysian Communications and Multimedia Commission
- **Google Fact Check Tools** – Authoritative fact-checking database
- **LangChain** – RAG orchestration framework
- **NVIDIA NIM** – DeepSeek LLM inference
- **HuggingFace** – Multilingual embeddings

---
