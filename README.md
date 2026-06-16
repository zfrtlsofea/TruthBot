# 🤖 TruthBot

> AI-Powered Telegram Chatbot for Fake News Detection and Verification

TruthBot is an AI-based Telegram chatbot designed to help users verify suspicious news, viral claims, misinformation, and online scams directly within Telegram conversations. By combining Retrieval-Augmented Generation (RAG), verified fact-checking sources, Natural Language Processing (NLP), and Large Language Models (LLMs), TruthBot provides evidence-based verdicts, confidence scores, explanations, and source citations.

---

## 📌 Overview

The rapid spread of misinformation through messaging applications such as Telegram has made information verification increasingly difficult. Traditional fact-checking websites require users to manually search for information, which can be time-consuming and inconvenient.

TruthBot addresses this challenge by integrating an automated fact-checking assistant directly into Telegram, allowing users to verify claims through simple conversational interactions.

### Objectives

- Design and develop an AI-powered Telegram chatbot for fake news detection.
- Provide evidence-based explanations using verified fact-checking sources.
- Support both English and Bahasa Melayu.
- Improve digital literacy and critical thinking among users.
- Evaluate system performance through accuracy and user acceptance testing.

---

## ✨ Features

### 🔍 Fake News Verification

Verify:

- News headlines
- Viral social media claims
- Forwarded messages
- Suspicious information

### 📚 Verified Fact-Check Knowledge Base

Uses trusted verification sources including:

- Sebenarnya.my
- Google Fact Check API
- Local fact-check datasets

### 📖 Evidence-Based Explanations

Provides detailed explanations supported by retrieved evidence.

### 🔗 Source Citation Display

Displays sources used during verification.

### 🌐 English & Bahasa Melayu Support

Supports multilingual interactions:

- English
- Bahasa Melayu

### 🛡️ Online Safety Tips

Provides educational tips on identifying misinformation and scams.

---

## 🏗️ System Architecture

```text
Telegram User
      │
      ▼
Telegram Platform
      │
      ▼
Telegram Bot API
      │
      ▼
Python Telegram Bot
      │
      ▼
Query Processing
      │
      ▼
LangChain Controller
      │
      ▼
Hybrid RAG Retrieval
 ├── ChromaDB
 ├── Live Sebenarnya.my Search
 └── Google Fact Check API
      │
      ▼
Evidence Aggregator
      │
      ▼
NVIDIA NIM
      │
      ▼
OpenAI Analysis
      │
      ▼
Verdict Generator
      │
      ▼
Explanation Generator
      │
      ▼
Source Citation Generator
      │
      ▼
Telegram Response
```

---

## ⚙️ Technology Stack

### Frontend

- Telegram Bot Interface

### Backend

- Python

### AI & NLP

- OpenAI API
- NVIDIA NIM
- LangChain

### Vector Database

- ChromaDB

### Cloud Infrastructure

- Google Cloud Platform (GCP)

### External Sources

- Sebenarnya.my
- Google Fact Check API

---

## 🔄 Workflow

### 1. User Submission

User submits:

- News claim
- Headline
- Suspicious message
- Fact-check request

### 2. Query Processing

The claim is extracted and normalized.

### 3. Hybrid Retrieval

Evidence is gathered from:

- ChromaDB
- Sebenarnya.my
- Google Fact Check API

### 4. Evidence Aggregation

Results are combined into a unified evidence context.

### 5. AI Analysis

OpenAI and NVIDIA NIM analyze the collected evidence.

### 6. Verdict Generation

The system determines whether the claim is:

- ✅ TRUE
- ❌ FALSE
- ⚠️ UNVERIFIED

### 7. Response Formatting

The chatbot returns:

- Verdict
- Confidence Score
- Explanation
- Sources

---

## 📊 Evidence Scoring

TruthBot assigns scores based on verification results from multiple sources.

### Example Rules

| Evidence Source | Result | Score |
|---------------|---------|--------|
| Google Fact Check | TRUE | +0.6 |
| Google Fact Check | FALSE | -0.6 |
| Local RAG | TRUE | +0.3 |
| Local RAG | FALSE | -0.3 |
| Sebenarnya.my | TRUE | +0.3 |
| Sebenarnya.my | FALSE | -0.3 |

---

## 🏆 Verdict Classification

| Verdict | Condition |
|----------|------------|
| TRUE | Score ≥ 1.0 |
| FALSE | Score ≤ -1.0 |
| UNVERIFIED | Score < 0.3 |

---

## 📈 Confidence Calculation

```python
confidence = min(abs(score) * 50, 100)
```

### Examples

| Score | Confidence |
|---------|------------|
| 0.5 | 25% |
| 1.0 | 50% |
| 1.5 | 75% |
| 2.0 | 100% |

---

## 📂 Suggested Project Structure

```text
truthbot/
│
├── bot/
│   ├── handlers/
│   ├── commands/
│   └── telegram_bot.py
│
├── rag/
│   ├── retriever.py
│   ├── embeddings.py
│   └── vector_store.py
│
├── services/
│   ├── google_factcheck.py
│   ├── sebenarnya.py
│   └── evidence_aggregator.py
│
├── ai/
│   ├── verdict_generator.py
│   ├── explanation_generator.py
│   └── llm.py
│
├── data/
│   ├── datasets/
│   └── knowledge_base/
│
├── config/
│   └── settings.py
│
├── tests/
│
├── requirements.txt
│
├── .env
│
└── README.md
```

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/zfrtlsofea/truthbot.git

cd truthbot
```

### Create Virtual Environment

Linux/macOS:

```bash
python -m venv venv

source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file:

```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token

OPENAI_API_KEY=your_openai_api_key

GOOGLE_FACTCHECK_API_KEY=your_google_api_key

NVIDIA_API_KEY=your_nvidia_api_key
```

### Run the Bot

```bash
python main.py
```

---

## 💬 Telegram Commands

| Command | Description |
|----------|------------|
| `/start` | Welcome message |
| `/help` | Display available commands |
| `/sources` | Show verification sources |
| `/tips` | Learn how to identify fake news |
| `/reset` | Clear conversation history |

---

## 📱 Example Usage

### User Input

```text
COVID-19 vaccines cause infertility
```

### Bot Response

```text
❌ Verdict: FALSE

📊 Confidence: 95%

📝 Explanation:
Multiple scientific studies and verified fact-checking
organizations found no evidence that COVID-19
vaccines cause infertility.

🔗 Sources:
• BBC Fact Check
• WHO
• Google Fact Check
```

---

## 📊 Evaluation Results

### User Acceptance Testing (UAT)

- Respondents: 10
- Positive Responses: 92%
- Negative Responses: 8%

Users reported:

- Increased confidence through evidence citations.
- Willingness to recommend TruthBot.
- Belief that TruthBot can help reduce misinformation.

### Accuracy Testing

| Metric | Result |
|----------|---------|
| Total Queries | 30 |
| Verifiable Queries | 18 |
| Unverified Queries | 12 |
| True Positive | 7 |
| True Negative | 11 |
| False Positive | 0 |
| False Negative | 0 |
| Accuracy | 100% |
| Coverage Rate | 60% |

---

## 🎯 Achievements

- Developed a fully functional AI-powered Telegram chatbot.
- Implemented Hybrid RAG architecture.
- Integrated verified fact-checking sources.
- Built multilingual support.
- Successfully completed:
  - Unit Testing
  - Integration Testing
  - Reliability Testing
  - Performance Testing
  - User Acceptance Testing
  - Accuracy Testing

---

## ⚠️ Limitations

- Dependent on existing fact-check datasets.
- Cannot verify claims never previously fact-checked.
- Limited multilingual capabilities.
- Less effective for highly informal language.

---

## 🔮 Future Work

- Real-time dataset updates.
- Additional verification sources.
- Advanced NLP models.
- Fake news trend monitoring.
- Enhanced multilingual support.
- Scam detection improvements.

---

## 🎓 Academic Information

**Project Title:** TruthBot: AI-Based Telegram Chatbot for Fake News Detection

**Course:** TMF4935 Final Year Project 2

**Student:** Zafiratul Sofea Szuhady (86121)

**Supervisor:** Associate Professor Dr. Cheah Wai Shiang

**Institution:** Universiti Malaysia Sarawak (UNIMAS)

---

## 📄 License

This project was developed for academic and research purposes.

Feel free to use, modify, and extend the project with proper attribution.

---

## 🙏 Acknowledgements

- Universiti Malaysia Sarawak (UNIMAS)
- Faculty of Computer Science and Information Technology
- Sebenarnya.my
- Google Fact Check API
- OpenAI
- NVIDIA
- LangChain Community

---
