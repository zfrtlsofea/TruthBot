# 🛡️ TruthBot: AI-Powered Fake News Detection Telegram Chatbot

## 📖 Overview

TruthBot is an AI-powered Telegram chatbot developed to help users verify the authenticity of news articles and combat misinformation. The system leverages Retrieval-Augmented Generation (RAG), vector embeddings, semantic search, and Large Language Models (LLMs) to provide evidence-based responses using data collected from Malaysia's official fact-checking portal, Sebenarnya.my.

Users can submit news claims, headlines, or article snippets through Telegram, and TruthBot retrieves relevant fact-checking articles before generating a concise explanation of whether the information is true, false, misleading, or partially true.

This project was developed as a Final Year Project (FYP) for the Bachelor of Software Engineering programme.

---

## 🎯 Objectives

* Combat the spread of misinformation and fake news.
* Provide fast and accessible fact-checking services through Telegram.
* Utilize AI and Natural Language Processing (NLP) techniques for semantic search.
* Improve public awareness and digital literacy regarding online information.

---

## 🏗️ System Architecture

```text
┌─────────────────┐
│ Telegram User   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ TruthBot        │
│ Telegram Bot    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Query Embedding │
│ (Sentence       │
│ Transformer)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ChromaDB Vector │
│ Database        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Relevant Fact   │
│ Check Articles  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LLM Response    │
│ Generation      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ User Receives   │
│ Explanation     │
└─────────────────┘
```

---

## 🚀 Features

### ✅ AI-Powered Fact Checking

Analyzes user-submitted claims using semantic similarity search and retrieval-augmented generation.

### ✅ Telegram Integration

Accessible directly through Telegram without requiring additional applications.

### ✅ Vector Search

Uses embeddings and ChromaDB to identify relevant fact-checking articles.

### ✅ Context-Aware Responses

Provides explanations based on retrieved evidence instead of simple keyword matching.

### ✅ Automated Dataset Collection

Scrapes verified fact-check articles from Sebenarnya.my for knowledge base construction.

### ✅ Scalable Knowledge Base

Supports continuous addition of newly published fact-checking articles.

---

## 📂 Project Structure

```text
TruthBot/
│
├── chatbot_telegram.py      # Main Telegram chatbot
├── scraper.py               # Dataset collection from Sebenarnya.my
├── build_vectordb.py        # Generate embeddings and vector database
├── requirements.txt         # Python dependencies
│
├── data/
│   ├── articles.json
│   └── cleaned_articles.json
│
├── chroma_db/
│   ├── chroma.sqlite3
│   └── index files
│
├── README.md
└── .gitignore
```

---

## 📊 Dataset

### Source

The dataset was collected from:

**Sebenarnya.my**

Malaysia's official fact-checking platform managed by the Malaysian Communications and Multimedia Commission (MCMC).

Official Website:

[https://www.sebenarnya.my](https://www.sebenarnya.my)

---

### Dataset Contents

The dataset contains fact-checking articles including:

* News headlines
* Claims being investigated
* Fact-check verdicts
* Detailed explanations
* Publication dates
* Categories and tags
* Source URLs

Example:

```json
{
  "title": "Claim Regarding Government Financial Aid",
  "content": "Detailed fact-check explanation...",
  "url": "https://www.sebenarnya.my/...",
  "date": "2025-01-15"
}
```

---

### Dataset Size

| Component            | Approximate Size |
| -------------------- | ---------------- |
| Raw Scraped Articles | Several MB       |
| Processed Dataset    | Several MB       |
| Embedding Vectors    | ~1.2 GB          |
| ChromaDB Database    | ~1.2 GB          |

The majority of storage usage comes from vector embeddings generated for semantic search and retrieval.

---

## 🤖 AI Technologies Used

### Sentence Transformers

Used to convert text into vector embeddings.

Model:

```text
all-MiniLM-L6-v2
```

---

### ChromaDB

Vector database used for storing and retrieving embeddings.

Features:

* Fast similarity search
* Persistent storage
* Lightweight deployment

---

### Retrieval-Augmented Generation (RAG)

TruthBot follows a RAG architecture:

1. User submits a claim.
2. Query is converted into embeddings.
3. Similar fact-check articles are retrieved.
4. Retrieved context is passed to the language model.
5. Final response is generated.

---

## 🛠️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/zfrtlsofea/TruthBot.git

cd TruthBot
```

---

### 2. Create Virtual Environment

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

Linux/macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration

Create a `.env` file:

```env
TELEGRAM_BOT_TOKEN=YOUR_TELEGRAM_BOT_TOKEN

OPENAI_API_KEY=YOUR_OPENAI_API_KEY
```

---

## 📥 Build Dataset

Run the scraper:

```bash
python scraper.py
```

This will:

* Crawl Sebenarnya.my
* Extract fact-checking articles
* Save the dataset locally

---

## 🧠 Generate Embeddings

Build the vector database:

```bash
python build_vectordb.py
```

This process:

* Loads scraped articles
* Generates embeddings
* Stores vectors inside ChromaDB

---

## ▶️ Run TruthBot

```bash
python chatbot_telegram.py
```

If successful:

```text
Bot is running...
```

---

## 📱 Using the Bot

Open Telegram:

```text
@zs_truth_bot
```

Example queries:

```text
Is this government aid announcement real?

Did the government officially announce this subsidy?

Is this viral WhatsApp message true?

Can you verify this news article?
```

---

## 🔍 Example Workflow

### User Query

```text
Government giving RM5000 assistance to all citizens.
```

### Retrieval

TruthBot searches the vector database for semantically similar fact-check articles.

### Response

```text
Based on information retrieved from Sebenarnya.my,
this claim has been identified as false.

Evidence:
- Official government sources deny the claim.
- No legitimate announcement exists.
- Similar misinformation was previously fact-checked.
```

---

## 📈 Future Improvements

* Multilingual and slang support
* Voice-based fact checking
* Real-time news verification
* Advanced misinformation detection models
* Continuous automated data updates

---

## 🎓 Academic Context

**Project Title**

TruthBot: AI-Based Telegram Chatbot for Fake News Detection

**Programme**

Bachelor of Software Engineering

**Institution**

Faculty of Computer Science and Information Technology (FCSIT)

Universiti Malaysia Sarawak (UNIMAS)

---

## 👩‍💻 Author

**Zafiratul Sofea**

Final Year Software Engineering Student

Universiti Malaysia Sarawak (UNIMAS)

GitHub: [https://github.com/zfrtlsofea]

---

## 📄 License

This project is developed for educational and research purposes.
