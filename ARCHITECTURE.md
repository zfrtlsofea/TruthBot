# TruthBot Architecture

A comprehensive guide to TruthBot's system design, component interactions, and technical decisions.

---

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [Component Architecture](#component-architecture)
3. [Data Flow Pipeline](#data-flow-pipeline)
4. [Technology Stack](#technology-stack)
5. [Hybrid RAG System](#hybrid-rag-system)
6. [Verdict Decision Logic](#verdict-decision-logic)
7. [Database & Storage](#database--storage)
8. [API Integration](#api-integration)
9. [Performance & Scalability](#performance--scalability)
10. [Error Handling & Resilience](#error-handling--resilience)
11. [Future Improvements](#future-improvements)

---

## High-Level Overview

TruthBot is a **Hybrid Retrieval-Augmented Generation (RAG) system** designed to verify claims in real-time by combining:

- **Local Knowledge Base** (ChromaDB vector store)
- **Live APIs** (Sebenarnya.my web scraping + Google Fact Check)
- **LLM Inference** (DeepSeek V3.2 via NVIDIA NIM)
- **Intelligent Scoring** (weighted evidence combination)

**Design Philosophy:**
> No single source of truth is complete. By combining local domain expertise (Sebenarnya.my), global authoritative sources (Google Fact Check), and fast semantic search (ChromaDB), TruthBot achieves both speed and accuracy.

---

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          TRUTHBOT SYSTEM                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  TELEGRAM INTERFACE (telegram-bot library)                  │  │
│  │  ├─ /start, /help, /sources, /tips, /reset commands        │  │
│  │  └─ Handle message input, format responses                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
│           │                                                         │
│           ▼                                                         │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  MESSAGE HANDLER (async)                                   │  │
│  │  ├─ Receive user claim                                      │  │
│  │  ├─ Add to conversation history                            │  │
│  │  └─ Trigger verification pipeline                          │  │
│  └──────────────────────────────────────────────────────────────┘  │
│           │                                                         │
│           ▼                                                         │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  VERIFICATION PIPELINE (verify_claim function)              │  │
│  │                                                              │  │
│  │  ┌──────────────────────────────────────────────────────┐   │  │
│  │  │ 1. LOCAL RAG RETRIEVAL (LangChain + ChromaDB)        │   │  │
│  │  │    ├─ Embed user claim (HuggingFace)                │   │  │
│  │  │    ├─ Search vector DB (cosine similarity)          │   │  │
│  │  │    └─ Retrieve top 5 chunks (≥20% threshold)        │   │  │
│  │  └──────────────────────────────────────────────────────┘   │  │
│  │           │                                                  │  │
│  │           ▼                                                  │  │
│  │  ┌──────────────────────────────────────────────────────┐   │  │
│  │  │ 2. LIVE SEBENARNYA.MY RETRIEVAL (Web Scraping)      │   │  │
│  │  │    ├─ Query sebenarnya.my with claim                │   │  │
│  │  │    ├─ Fetch latest articles                         │   │  │
│  │  │    └─ Extract title, body, URL (top 3)             │   │  │
│  │  └──────────────────────────────────────────────────────┘   │  │
│  │           │                                                  │  │
│  │           ▼                                                  │  │
│  │  ┌──────────────────────────────────────────────────────┐   │  │
│  │  │ 3. GOOGLE FACT CHECK RETRIEVAL (API)                │   │  │
│  │  │    ├─ Query Google Fact Check Tools API             │   │  │
│  │  │    ├─ Extract verdicts & ratings                    │   │  │
│  │  │    └─ Return top 5 results                          │   │  │
│  │  └──────────────────────────────────────────────────────┘   │  │
│  │           │                                                  │  │
│  │           ▼                                                  │  │
│  │  ┌──────────────────────────────────────────────────────┐   │  │
│  │  │ 4. EVIDENCE SCORING (Weighted Combination)          │   │  │
│  │  │    ├─ Score from Google Fact Check (weight: 0.5)    │   │  │
│  │  │    ├─ Score from Sebenarnya.my (weight: 0.3)       │   │  │
│  │  │    ├─ Score from Local RAG (weight: 0.2)           │   │  │
│  │  │    └─ Combine → Final score (-2.0 to +2.0)         │   │  │
│  │  └──────────────────────────────────────────────────────┘   │  │
│  │           │                                                  │  │
│  │           ▼                                                  │  │
│  │  ┌──────────────────────────────────────────────────────┐   │  │
│  │  │ 5. VERDICT COMPUTATION                              │   │  │
│  │  │    ├─ Map score to verdict                          │   │  │
│  │  │    ├─ Calculate confidence percentage               │   │  │
│  │  │    └─ Return: verdict + confidence                 │   │  │
│  │  └──────────────────────────────────────────────────────┘   │  │
│  │           │                                                  │  │
│  │           ▼                                                  │  │
│  │  ┌──────────────────────────────────────────────────────┐   │  │
│  │  │ 6. LLM EXPLANATION GENERATION (DeepSeek V3.2)       │   │  │
│  │  │    ├─ Assemble evidence context                     │   │  │
│  │  │    ├─ Prompt LLM with verdict (not changeable)     │   │  │
│  │  │    ├─ Generate human-readable explanation          │   │  │
│  │  │    └─ Format with emoji & citations                │   │  │
│  │  └──────────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│           │                                                         │
│           ▼                                                         │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  RESPONSE FORMATTER                                        │  │
│  │  └─ Return to Telegram with sources & tips               │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

EXTERNAL SERVICES (Right Column)
├─ ChromaDB (Local Vector Store)
├─ NVIDIA NIM (DeepSeek LLM)
├─ Sebenarnya.my (Web Scraping)
└─ Google Fact Check API (REST)
```

---

## Data Flow Pipeline

### Request Flow (User → Response)

```
User (Telegram)
    │
    │ sends claim: "Vaccine causes autism"
    │
    ▼
MessageHandler (async)
    │
    │ update.message.text = "Vaccine causes autism"
    │
    ▼
verify_claim(claim) function
    │
    ├─► retrieve_sebenarnya_live(claim)
    │   └─ GET https://sebenarnya.my/?s=...
    │   └─ Returns: list of articles with body text
    │
    ├─► retrieve_google_factcheck(claim)
    │   └─ GET https://factchecktools.googleapis.com/v1alpha1/claims:search
    │   └─ Returns: [{"claim_text": "...", "rating": "false"}]
    │
    ├─► qa_chain.invoke({"query": claim}) [if ChromaDB loaded]
    │   └─ Embed claim → Search ChromaDB → Retrieve + Rank
    │   └─ Returns: {"result": "...", "source_documents": [...]}
    │
    ├─► compute_evidence_score(local, google, live)
    │   ├─ if google rating contains "false": score -= 1.0
    │   ├─ if sebenarnya articles found: score += 0.3
    │   └─ if local RAG has "FALSE": score -= 0.2
    │
    ├─► compute_verdict_and_confidence(score)
    │   ├─ if score >= 1.0: verdict = "TRUE"
    │   ├─ elif score <= -1.0: verdict = "FALSE"
    │   └─ elif -1.0 < score < 1.0: verdict = "MISLEADING"
    │
    ├─► llm.invoke(final_prompt) [DeepSeek V3.2]
    │   └─ Returns: explanation text
    │
    └─► Format response with emoji + sources + tip
        └─ Return to handle_message()

    ▼
Telegram sends to user
    │
    │ 🔍 *Verdict:* FALSE
    │ 📊 *Confidence:* 98.5%
    │ 📋 *Explanation:* ...
    │ 🔗 *Sources:* ...
    │ ⚠️ *Tip:* ...
    │
    ▼
User receives response
```

### Data Initialization Flow (Setup → Ready)

```
1. python scraper.py
   ├─ GET https://sebenarnya.my/page/1/
   ├─ GET https://sebenarnya.my/page/2/
   ├─ ... (up to 200 pages)
   └─ Output: sebenarnya_articles.json
      ├─ 2,000–5,000 articles
      ├─ Each with: {url, title, date, content}
      └─ File size: ~100–300 MB

2. python build_vectordb.py
   ├─ Load sebenarnya_articles.json
   ├─ Convert to LangChain Documents
   ├─ Split with RecursiveCharacterTextSplitter
   │  ├─ chunk_size: 800 chars
   │  ├─ chunk_overlap: 150 chars
   │  └─ Total chunks: 400k–800k
   ├─ Generate embeddings (paraphrase-multilingual-MiniLM-L12-v2)
   ├─ Store in ChromaDB
   └─ Output: chroma_db/
      ├─ embeddings.parquet
      ├─ documents.parquet
      └─ metadata.json

3. python chatbot_telegram.py
   ├─ Load ChromaDB
   ├─ Initialize LangChain components
   │  ├─ HuggingFaceEmbeddings
   │  ├─ Chroma vector store
   │  ├─ RetrievalQA chain
   │  └─ ChatOpenAI (NVIDIA NIM)
   ├─ Initialize Telegram bot
   └─ Start polling for messages
```

---

## Technology Stack

### Core Framework

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Chatbot** | python-telegram-bot | 20.0+ | Telegram message handling |
| **Backend** | FastAPI | N/A (not in scope for Phase 1) | Planned for production |
| **LLM Orchestration** | LangChain | 0.2.0+ | RAG pipeline coordination |
| **Vector Store** | ChromaDB | 0.5.0+ | Semantic search & storage |

### AI/ML Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language Model** | DeepSeek V3.2 (NVIDIA NIM) | Explanation generation |
| **Embeddings** | HuggingFace (paraphrase-multilingual-MiniLM-L12-v2) | Text vectorization |
| **Text Splitting** | LangChain RecursiveCharacterTextSplitter | Intelligent chunking |
| **RAG Chain** | LangChain RetrievalQA | Local knowledge retrieval |

### External APIs

| Service | Purpose | API Type | Rate Limit |
|---------|---------|----------|-----------|
| **Sebenarnya.my** | Malaysian fact-check articles | Web scraping (HTML) | No official limit |
| **Google Fact Check Tools** | Authoritative verdicts | REST API | 100 req/min |
| **NVIDIA NIM** | DeepSeek LLM inference | REST (OpenAI-compatible) | Per API key |

### Data Storage

| System | Purpose | Storage |
|--------|---------|---------|
| **ChromaDB** | Vector embeddings + metadata | Local filesystem (`chroma_db/`) |
| **JSON** | Scraped articles | `sebenarnya_articles.json` |
| **.env** | Secrets & config | Local file (git-ignored) |
| **Telegram Messages** | User conversation history | In-memory dict (cleared on reset) |

---

## Hybrid RAG System

### Why Hybrid?

No single fact-checking source is comprehensive:

- **Local RAG** (ChromaDB): ✅ Fast, available offline | ❌ Limited to Sebenarnya.my, may be stale
- **Sebenarnya.my API**: ✅ Official, trusted, Malaysian context | ❌ Slow, may have limited coverage
- **Google Fact Check**: ✅ Global, authoritative, diverse sources | ❌ Slow API, may not cover all claims

**TruthBot's Solution:** Query all three in parallel, weight by trustworthiness & recency.

### Source Weights

```python
WEIGHT_GOOGLE      = 0.5   # Most authoritative
WEIGHT_SEBENARNYA  = 0.3   # Official local source
WEIGHT_LOCAL       = 0.2   # Fast but potentially stale
```

**Rationale:**
- Google Fact Check pulls from multiple verified sources worldwide
- Sebenarnya.my is Malaysia's official MCMC portal, trusted locally
- Local RAG is fast but shouldn't dominate (outdated info risk)

### Scoring Algorithm

```python
def compute_evidence_score(local_answer, google_results, live_articles):
    score = 0.0
    
    # Google Fact Check scoring
    for item in google_results:
        rating = item.get("rating", "").lower()
        if "false" in rating:
            score -= WEIGHT_GOOGLE * 2      # Strong FALSE signal
        elif "true" in rating:
            score += WEIGHT_GOOGLE * 2      # Strong TRUE signal
        elif "misleading" in rating:
            score -= WEIGHT_GOOGLE * 1      # Weak FALSE signal
    
    # Sebenarnya.my presence (existence = some verification happened)
    if live_articles:
        score += WEIGHT_SEBENARNYA * 1
    
    # Local RAG scoring
    if local_answer:
        if "FALSE" in local_answer:
            score -= WEIGHT_LOCAL * 1
        elif "TRUE" in local_answer:
            score += WEIGHT_LOCAL * 1
    
    return score  # Range: -2.0 to +2.0
```

### Verdict Mapping

```python
def compute_verdict_and_confidence(score):
    # Confidence: how confident are we in the verdict
    confidence = min((abs(score) / 2) * 100, 100)
    
    # Verdict: based on score magnitude & direction
    if score >= 1.0:
        verdict = "TRUE"
    elif score <= -1.0:
        verdict = "FALSE"
    elif -1.0 < score < 1.0:
        verdict = "MISLEADING"
    else:
        verdict = "UNVERIFIED"
    
    return verdict, round(confidence, 2)
```

**Examples:**

| Evidence | Score | Verdict | Confidence |
|----------|-------|---------|------------|
| Google: FALSE + Sebenarnya exists | -1.5 | FALSE | 75% |
| Google: TRUE + Google: TRUE | +2.0 | TRUE | 100% |
| Google: MISLEADING only | -0.5 | MISLEADING | 25% |
| No evidence found | 0.0 | UNVERIFIED | 0% |

---

## Verdict Decision Logic

### Four Possible Verdicts

1. **TRUE** – Evidence clearly confirms the claim
2. **FALSE** – Evidence clearly contradicts or debunks the claim
3. **MISLEADING** – Claim is partially true, out of context, or exaggerated
4. **UNVERIFIED** – Insufficient or no evidence found

### Decision Tree

```
User claim
    │
    ├─ Evidence from Google Fact Check?
    │  ├─ YES
    │  │  ├─ Rating = "false/debunked/incorrect" → score -= 2.0
    │  │  ├─ Rating = "true/verified/correct" → score += 2.0
    │  │  └─ Rating = "misleading/partially" → score -= 1.0
    │  │
    │  └─ NO
    │     └─ Continue to next sources
    │
    ├─ Evidence from Live Sebenarnya.my?
    │  ├─ YES → score += 1.0 (verification happened)
    │  └─ NO → Continue
    │
    ├─ Evidence from Local RAG (ChromaDB)?
    │  ├─ YES & verdict = FALSE → score -= 1.0
    │  ├─ YES & verdict = TRUE → score += 1.0
    │  └─ NO → score unchanged
    │
    ▼
    Final Score → Verdict
    ├─ score >= 1.0 → TRUE (high confidence)
    ├─ score <= -1.0 → FALSE (high confidence)
    ├─ -1.0 < score < 1.0 → MISLEADING (mixed signals)
    └─ score ≈ 0.0 → UNVERIFIED (no strong signals)
```

---

## Database & Storage

### ChromaDB Vector Store

**Purpose:** Store embeddings of Sebenarnya.my articles for fast semantic search.

**Schema:**
```python
Document {
    id: str                    # Unique document ID
    embedding: float[384]      # 384-dim vector (MiniLM output)
    metadata: {
        "title": str,          # Article title
        "url": str,            # Source URL
        "date": str            # Publication date
    }
    content: str               # Article text (800 chars per chunk)
}
```

**Size Estimates:**
- Articles: 2,000–5,000
- Chunks per article: 3–8 (avg 5)
- **Total chunks: 400k–800k**
- Embedding size: 384 dimensions × 4 bytes = 1,536 bytes per embedding
- **Total disk: ~600 GB–1.2 TB** (with metadata & indexes)

**Location:** `./chroma_db/`

### Sebenarnya Articles JSON

**File:** `sebenarnya_articles.json`

**Schema:**
```json
[
  {
    "url": "https://sebenarnya.my/article-123/",
    "title": "Artikel Mengenai Vaksin COVID-19",
    "date": "2023-06-15T10:30:00Z",
    "content": "Isi lengkap artikel..."
  }
]
```

**Size:** ~100–300 MB (2000–5000 articles × ~50 KB each)

### Conversation History

**Storage:** In-memory dictionary (per user session)

```python
user_conversations: dict = {
    123456: [           # user_id
        {"role": "user", "content": "Is this fake?"},
        {"role": "assistant", "content": "🔍 Verdict: ..."},
        ...
    ]
}
```

**Cleanup:** Auto-limit to last 20 messages per user to prevent memory overflow.

---

## API Integration

### Google Fact Check Tools API

**Endpoint:** `https://factchecktools.googleapis.com/v1alpha1/claims:search`

**Request:**
```python
GET https://factchecktools.googleapis.com/v1alpha1/claims:search?query=<claim>&key=<API_KEY>
```

**Response:**
```json
{
  "claims": [
    {
      "text": "Vaccines cause autism",
      "claimReview": [
        {
          "title": "Vaccines Do Not Cause Autism",
          "textualRating": "False",
          "publisher": { "name": "Snopes" },
          "url": "https://snopes.com/..."
        }
      ]
    }
  ]
}
```

**Parsed by:**
```python
def retrieve_google_factcheck(claim: str) -> list:
    results = []
    for item in data.get("claims", [])[:5]:
        reviews = item.get("claimReview", [])
        if reviews:
            results.append({
                "claim_text": item.get("text", ""),
                "rating": reviews[0].get("textualRating", "Unknown"),
                "source_name": reviews[0].get("publisher", {}).get("name", ""),
                "url": reviews[0].get("url", "")
            })
    return results
```

### Sebenarnya.my Web Scraping

**Approach:** Direct HTML scraping (no official API)

**1. Search for articles:**
```python
GET https://sebenarnya.my/?s=<query>
```

**2. Extract links:**
```python
soup.select("h2.entry-title a, h1.entry-title a, .post-title a")
```

**3. Fetch article details:**
```python
GET https://sebenarnya.my/article-title/
```

**4. Extract content:**
```python
# Title
soup.select_one("h1.entry-title, h2.entry-title, .post-title")

# Body
soup.select_one(".entry-content, .post-content, article")

# Date
soup.select_one("time.entry-date, .post-date, time")
```

### NVIDIA NIM API (DeepSeek)

**Endpoint:** `https://integrate.api.nvidia.com/v1`

**LangChain Wrapper:**
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-ai/deepseek-v3.2",
    openai_api_key=NVIDIA_API_KEY,
    openai_api_base="https://integrate.api.nvidia.com/v1",
    temperature=0.2,
    max_tokens=800
)

response = llm.invoke(prompt)  # Returns: str
```

**Why DeepSeek V3.2?**
- Multilingual (supports Malay + English)
- Fast inference via NVIDIA NIM
- Low hallucination risk at temp=0.2
- Good instruction-following

---

## Performance & Scalability

### Latency Breakdown

Typical request flow (with all sources available):

| Step | Time | Details |
|------|------|---------|
| Telegram receives message | < 1s | Network + polling |
| Local ChromaDB embedding | 0.5–1s | Sentence-transformer |
| Local ChromaDB search | 0.1–0.5s | Cosine similarity |
| Sebenarnya.my scrape | 2–5s | Network + HTML parsing |
| Google Fact Check API | 1–3s | Network + JSON parsing |
| LLM generation | 2–4s | DeepSeek inference via NIM |
| Response formatting | 0.1s | String manipulation |
| **Total** | **~7–15s** | Typical response time |

**Optimization strategies:**
- Parallel API calls (concurrent.futures)
- Cache embeddings in ChromaDB
- Set `similarity_threshold = 0.3` for faster retrieval
- Reduce `MAX_LIVE_ARTICLES` or `MAX_LOCAL_CHUNKS`

### Scalability Considerations

#### Vertical Scaling (Single Server)
- **CPU**: 4+ cores (parallel scraping + embedding)
- **RAM**: 8–16 GB (ChromaDB in memory for fast search)
- **Storage**: 1–2 TB (vector DB + article cache)
- **Network**: 100+ Mbps (live API calls)

#### Horizontal Scaling (Multiple Servers)
1. **Shared ChromaDB**: Mount vector DB on distributed filesystem
2. **Load Balancer**: Route requests to multiple chatbot instances
3. **Message Queue**: Use Redis/RabbitMQ for async verification
4. **Caching**: Redis to cache recent verdicts

**Pseudo-architecture:**
```
User (Telegram)
    │
    ▼
Load Balancer
    │
    ├─ Bot Instance 1 ─────┐
    ├─ Bot Instance 2 ─────┼─ Shared ChromaDB (NFS)
    ├─ Bot Instance 3 ─────┤
    └─ Bot Instance N ─────┤
                            │
                   Cache (Redis) + Queue (RabbitMQ)
```

---

## Error Handling & Resilience

### Failure Scenarios & Recovery

| Scenario | Impact | Recovery |
|----------|--------|----------|
| ChromaDB offline | Local RAG unavailable | Continue with live APIs |
| Google API rate limit | Missing authoritative source | Use Sebenarnya.my + local RAG |
| Sebenarnya.my down | Missing live articles | Use local RAG + Google |
| DeepSeek API error | Can't generate explanation | Fallback to basic template response |
| Telegram connection lost | Message delivery fails | Telegram client retries automatically |

### Error Handling Code

```python
# ChromaDB may not be loaded
try:
    vectorstore = Chroma(...)
    retriever = vectorstore.as_retriever(...)
except Exception as e:
    logger.warning(f"ChromaDB not available: {e}")
    retriever = None

# Live APIs may fail
try:
    articles = retrieve_sebenarnya_live(claim)
except Exception as e:
    logger.error(f"Sebenarnya.my error: {e}")
    articles = []

try:
    results = retrieve_google_factcheck(claim)
except Exception as e:
    logger.error(f"Google Fact Check error: {e}")
    results = []

# LLM may fail
try:
    response = llm.invoke(final_prompt_text)
except Exception as e:
    logger.error(f"LLM error: {e}")
    return fallback_response()
```

### Fallback Strategies

**If no sources found:**
```
🔍 *Verdict:* UNVERIFIED
📊 *Confidence:* 0%

📋 *Explanation:* No matching information was found in 
available databases. This claim could not be verified.

⚠️ *Tip:* When you can't verify a claim, don't share it. 
Check sebenarnya.my or official sources directly.
```

---

## Security Considerations

### API Key Management
- Store in `.env` file (git-ignored)
- Rotate keys quarterly
- Use separate keys for dev/prod
- Monitor API usage for anomalies

### Input Validation
- Limit claim length to 500 chars
- Sanitize user input before scraping
- Rate-limit per user (prevent abuse)

### Data Privacy
- Don't store user claims on disk
- Clear conversation history on reset
- GDPR-compliant (no personal data logging)
- Comply with Telegram Bot API terms

---

## References

- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [NVIDIA NIM API](https://docs.nvidia.com/)
- [Google Fact Check Tools](https://toolbox.google.com/factcheck/explorer)
- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)

---
