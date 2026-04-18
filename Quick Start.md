# TruthBot – Quick Start Guide

## 📋 Prerequisites

- Python 3.9+
- Git
- 3 API Keys (free):
  1. NVIDIA (for LLM)
  2. Telegram Bot Token
  3. Google Fact Check

---

## 🚀 5-Minute Setup

### Step 1: Get API Keys (5 min)

#### NVIDIA API Key
1. Go to https://www.nvidia.com/
2. Sign up → AI Foundation Models → Get NIM API Key
3. Copy key starting with `nvapi-...`

#### Telegram Bot Token
1. Open Telegram → Search `@BotFather`
2. Send `/newbot`
3. Follow instructions, copy token

#### Google Fact Check
1. Go to https://console.cloud.google.com/
2. Create new project
3. Enable "Google Fact Check Tools API"
4. Create API key

### Step 2: Clone & Install 

```bash
# Clone
git clone https://github.com/zfrtlsofea/TruthBot.git
cd TruthBot

# Virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install
pip install -r requirements.txt
```

### Step 3: Configure

```bash
# Copy template
cp .env.example .env

# Edit .env (add your API keys)
nano .env
```

Paste your keys:
```env
NVIDIA_API_KEY=nvapi-...
TELEGRAM_BOT_TOKEN=123456789:ABC...
GOOGLE_FACT_CHECK_API_KEY=AIza...
```

### Step 4: Build Knowledge Base

```bash
# Download articles from Sebenarnya.my
python scraper.py

# Build vector database
python build_vectordb.py
```

**Expected output:**
```
✅ Loaded 2345 articles
✅ Split into 450000 chunks
✅ Created vector database
```

### Step 5: Run Bot! 

```bash
python chatbot_telegram.py
```

You should see:
```
INFO: Starting TruthBot — LangChain Hybrid RAG pipeline...
INFO: TruthBot is running. Press Ctrl+C to stop.
```

---

## ✅ Test It

1. Open Telegram
2. Search for your bot (from BotFather token)
3. Send `/start`
4. Try: `"Vaccines cause autism"`
5. Bot should respond with **FALSE** verdict

**Success!** 🎉

---

## 📚 Next Steps

- Read [README.md](README.md) for full features
- Check [ARCHITECTURE.md](ARCHITECTURE.md) for how it works
- See [DEPLOYMENT.md](DEPLOYMENT.md) for production setup
- Join [GitHub Discussions](https://github.com/zfrtlsofea/TruthBot/discussions)

---

## 🆘 Troubleshooting

### "API key not found"
→ Check `.env` file exists and keys are correct

### "ChromaDB not available"
→ Run `python build_vectordb.py` again

### "Bot doesn't respond"
→ Check Telegram token is valid (test with @BotFather)

### "Slow responses"
→ Wait for initial vector DB build (~10 min)

---

## 📊 What Gets Downloaded

- **sebenarnya_articles.json** (~100 MB)
  - 2,000–5,000 Malaysian fact-checked articles

- **chroma_db/** (~600 MB)
  - 400k–800k searchable text chunks
  - Embeddings for fast semantic search

---

## 🔧 Common Commands

```bash
# Rebuild knowledge base
python scraper.py && python build_vectordb.py

# Clear conversation history
/reset (in Telegram)

# View available commands
/help (in Telegram)

# Stop bot
Ctrl+C

# Restart after changes
systemctl restart truthbot  # (if using systemd)
```

---

## 🎯 Quick Tips

1. **Keep API keys safe** – Never commit `.env`
2. **Update KB weekly** – Schedule `python scraper.py`
3. **Monitor logs** – Check for errors
4. **Test small claims first** – Before deploying widely

---

For details, see [README.md](README.md) and [ARCHITECTURE.md](ARCHITECTURE.md).