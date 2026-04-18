"""
TruthBot: AI-Based Telegram Chatbot for Fake News Detection
============================================================
HYBRID RAG PIPELINE — powered by LangChain + ChromaDB + NVIDIA NIM
"""

import os
import logging
import requests
import re
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA 
from langchain_core.documents import Document

from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, ContextTypes
)

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

NVIDIA_API_KEY        = os.getenv("NVIDIA_API_KEY")
TELEGRAM_BOT_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN")
GOOGLE_FACT_CHECK_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY", "")
NIM_MODEL             = os.getenv("NIM_MODEL", "deepseek-ai/deepseek-v3.1")
NIM_API_BASE          = os.getenv("NIM_API_BASE", "https://integrate.api.nvidia.com/v1")

CHROMA_DB_PATH        = "./chroma_db"
COLLECTION_NAME       = "sebenarnya_articles"
EMBEDDING_MODEL       = "paraphrase-multilingual-MiniLM-L12-v2"

WEIGHT_GOOGLE         = 0.5 # Google Fact Check is weighted most heavily due to its authoritative fact-checking verdicts from multiple sources
WEIGHT_SEBENARNYA     = 0.3 # Sebenarnya.my is weighted moderately — it's a trusted local source but may not have coverage of every claim, and some articles may be outdated
WEIGHT_LOCAL          = 0.2 # Local RAG is weighted less than live sources to avoid over-reliance on potentially outdated information in the vector store
SIMILARITY_THRESHOLD  = 0.2 # Only consider chunks with ≥20% similarity as relevant
MAX_LOCAL_CHUNKS      = 5 # Limit local RAG to top 5 most relevant chunks to maintain answer quality and relevance
MAX_LIVE_ARTICLES     = 1 # Limit live Sebenarnya.my retrieval to top 3 articles to ensure response speed and relevance

HEADERS = {"User-Agent": "TruthBot/2.0 (Academic Research, UNIMAS)"}

if not NVIDIA_API_KEY:
    logger.error("NVIDIA_API_KEY not set in .env")
    exit(1)

if not TELEGRAM_BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN not set in .env")
    exit(1)

logger.info("Initialising LangChain ChatOpenAI (NVIDIA NIM / DeepSeek)...")
llm = ChatOpenAI(
    model=NIM_MODEL,
    openai_api_key=NVIDIA_API_KEY,
    openai_api_base=NIM_API_BASE,
    temperature=0.2,
    max_tokens=800
)
logger.info("LangChain LLM ready.")

logger.info(f"Loading LangChain HuggingFaceEmbeddings: {EMBEDDING_MODEL} ...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
logger.info("LangChain Embeddings ready.")

vectorstore = None
retriever   = None
try:
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": SIMILARITY_THRESHOLD,
            "k": MAX_LOCAL_CHUNKS
        }
    )
    chunk_count = vectorstore._collection.count()
    logger.info(
        f"LangChain Chroma loaded — {chunk_count} chunks in vector store. "
        f"Similarity threshold: {int(SIMILARITY_THRESHOLD * 100)}%"
    )
except Exception as e:
    logger.warning(
        f"ChromaDB not available: {e}\n"
        f"Local RAG retrieval disabled. Run scraper.py -> build_vectordb.py to enable."
    )

RAG_PROMPT_TEMPLATE = """You are TruthBot, an AI-powered fake news and scam detection assistant on Telegram, built for Malaysia.
You support English and Bahasa Melayu. Respond in the same language the user used.

Your task is to verify the user's claim using ONLY the context provided below.
Apply these verdict rules strictly:
- TRUE       : Context clearly confirms the claim is accurate
- FALSE      : Context clearly contradicts or debunks the claim
- MISLEADING : Context shows the claim is partially true, out of context, or exaggerated
- UNVERIFIED : Context is insufficient or absent — do not guess

Never fabricate information. If unsure, always choose UNVERIFIED.

--- RETRIEVED CONTEXT FROM FACT-CHECKING SOURCES ---
{context}
----------------------------------------------------

User's claim: {question}

Respond using this exact format:

🔍 *Verdict:* [TRUE / FALSE / MISLEADING / UNVERIFIED]

📋 *Explanation:* [2–4 sentences referencing the retrieved context directly.]

⚠️ *Tip:* [One short practical tip relevant to this type of claim or scam.]"""

rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_PROMPT_TEMPLATE
)

qa_chain = None
if retriever is not None:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": rag_prompt}
    )
    logger.info("LangChain RetrievalQA chain ready.")
else:
    logger.warning("RetrievalQA chain not built — ChromaDB retriever unavailable.")

user_conversations: dict = {}


def retrieve_sebenarnya_live(claim: str) -> list:
    try:
        query = requests.utils.quote(claim[:150])
        logger.info(f"Fetching live from Sebenarnya.my for: {claim[:100]}")
        
        r = requests.get(
            f"https://sebenarnya.my/?s={query}",
            headers=HEADERS, timeout=10
        )
        
        if r.status_code == 404 or r.status_code >= 500:
            logger.warning(f"Sebenarnya.my returned status {r.status_code} — site may be down")
            return []
        
        soup = BeautifulSoup(r.text, "html.parser")
        links = []
        
        # Primary selectors
        for a in soup.select("h2.entry-title a, h1.entry-title a, .post-title a"):
            href = a.get("href", "")
            if href and "sebenarnya.my" in href:
                links.append(href)
        
        # Fallback selectors if primary ones don't work
        if not links:
            logger.debug("Primary selectors returned no results. Trying fallback selectors...")
            for a in soup.select("a[class*='title']"):
                href = a.get("href", "")
                if href and "sebenarnya.my" in href:
                    links.append(href)
        
        if not links:
            logger.debug("Still no results. Trying generic article selectors...")
            for a in soup.select("article a"):
                href = a.get("href", "")
                if href and "sebenarnya.my" in href:
                    links.append(href)
        
        links = list(set(links))[:MAX_LIVE_ARTICLES]
        logger.info(f"Sebenarnya.my search: Found {len(links)} article links")

        articles = []
        for url in links:
            try:
                logger.info(f"  → Fetching article: {url}")
                r = requests.get(url, headers=HEADERS, timeout=10)
                soup = BeautifulSoup(r.text, "html.parser")

                title_tag = soup.select_one("h1.entry-title, h2.entry-title, .post-title")
                title = title_tag.get_text(strip=True) if title_tag else "Untitled"

                body_tag = soup.select_one(".entry-content, .post-content, article")
                if not body_tag:
                    logger.warning(f"  → No article body found in {url}")
                    continue

                for tag in body_tag.select(
                    "script, style, nav, footer, .sharedaddy, .jp-relatedposts"
                ):
                    tag.decompose()

                body = body_tag.get_text(separator=" ", strip=True)
                if len(body) < 100:
                    logger.warning(f"  → Article too short ({len(body)} chars): {url}")
                    continue

                articles.append({"title": title, "url": url, "body": body[:4000]})
                logger.info(f"  ✓ Article extracted: {len(body)} chars")
            except requests.exceptions.Timeout:
                logger.warning(f"  ✗ Timeout fetching article: {url}")
                continue
            except Exception as e:
                logger.warning(f"  ✗ Error fetching article {url}: {e}")
                continue

        logger.info(f"Live Sebenarnya.my: extracted {len(articles)} full articles.")
        return articles

    except requests.exceptions.Timeout:
        logger.error("Sebenarnya.my live search timeout (10s) — site may be slow/down")
        return []
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Sebenarnya.my connection error: {e} — site may be down")
        return []
    except Exception as e:
        logger.error(f"Sebenarnya.my live retrieval error: {e}", exc_info=True)
        return []


def retrieve_google_factcheck(claim: str) -> list:
    if not GOOGLE_FACT_CHECK_KEY:
        logger.warning("Google Fact Check API key not set")
        return []
    try:
        logger.info(f"Querying Google Fact Check API for: {claim[:100]}")
        r = requests.get(
            "https://factchecktools.googleapis.com/v1alpha1/claims:search",
            params={"query": claim[:200], "key": GOOGLE_FACT_CHECK_KEY},
            timeout=3 # Short timeout since this is a secondary source and we don't want to delay the response if Google is slow
        )
        
        if r.status_code != 200:
            logger.warning(f"Google Fact Check returned status {r.status_code}: {r.text[:200]}")
            return []
        
        data = r.json()

        results = []
        for item in data.get("claims", [])[:5]:
            reviews = item.get("claimReview", [])
            if reviews:
                result_item = {
                    "claim_text":   item.get("text", ""),
                    "rating":       reviews[0].get("textualRating", "Unknown"),
                    "source_name":  reviews[0].get("publisher", {}).get("name", "Unknown"),
                    "url":          reviews[0].get("url", ""),
                    "review_title": reviews[0].get("title", "")
                }
                results.append(result_item)
                logger.info(f"  → Found: {result_item['rating']} ({result_item['source_name']})")

        logger.info(f"Google Fact Check: {len(results)} results found.")
        return results
    except requests.exceptions.Timeout:
        logger.error("Google Fact Check API timeout (10s)")
        return []
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Google Fact Check API connection error: {e}")
        return []
    except Exception as e:
        logger.error(f"Google Fact Check error: {e}", exc_info=True)
        return []


def compute_evidence_score(local_answer, google_results, live_articles):
    score = 0.0

    for item in google_results:
        rating = item.get("rating", "").lower()
        if "false" in rating:
            score -= WEIGHT_GOOGLE * 2
        elif "true" in rating:
            score += WEIGHT_GOOGLE * 2
        elif "misleading" in rating:
            score -= WEIGHT_GOOGLE * 1

    if live_articles:
        score += WEIGHT_SEBENARNYA * 1

    if local_answer:
        if "FALSE" in local_answer:
            score -= WEIGHT_LOCAL * 1
        elif "TRUE" in local_answer:
            score += WEIGHT_LOCAL * 1

    return score


def compute_verdict_and_confidence(score):
    confidence = min((abs(score) / 2) * 100, 100) # Scale to 0–100% based on max possible score of ±2

    if score >= 1.0:
        verdict = "TRUE"
    elif score <= -1.0:
        verdict = "FALSE"
    else:
        verdict = "MISLEADING"

    return verdict, round(confidence, 2)


def detect_language(text: str) -> str:
    """
    Detect whether the user's input is in English or Bahasa Melayu.
    
    Returns:
        "english" or "malay"
    """
    
    # Common Malay words and patterns
    malay_keywords = [
        'adalah', 'dengan', 'untuk', 'tidak', 'ada', 'telah', 'yang', 'dari',
        'ini', 'itu', 'ke', 'di', 'pada', 'oleh', 'jika', 'atau', 'dan',
        'tapi', 'karena', 'apa', 'siapa', 'mana', 'kapan', 'bagaimana',
        'sudah', 'akan', 'bisa', 'dapat', 'harus', 'perlu', 'boleh',
        'negara', 'malaysia', 'klaim', 'berita', 'palsu', 'bohong', 'nyata',
        'fakta', 'bukti', 'sumber', 'artikel', 'penjelasan', 'kesimpulan'
    ]
    
    # Common English words
    english_keywords = [
        'the', 'is', 'are', 'be', 'have', 'has', 'do', 'does', 'did',
        'will', 'would', 'should', 'could', 'can', 'may', 'must',
        'claim', 'news', 'fake', 'false', 'true', 'fact', 'evidence',
        'source', 'article', 'explanation', 'verdict', 'verify', 'check',
        'what', 'which', 'who', 'where', 'when', 'why', 'how', 'about',
        'for', 'from', 'to', 'in', 'on', 'at', 'by', 'with', 'and', 'or'
    ]
    
    # Convert to lowercase and split into words
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    # Count keyword matches
    malay_count = sum(1 for word in words if word in malay_keywords)
    english_count = sum(1 for word in words if word in english_keywords)
    
    logger.debug(f"Language detection: Malay={malay_count}, English={english_count}")
    
    # If no clear match, check for Malay-specific characters/patterns
    if malay_count == 0 and english_count == 0:
        # Check for common Malay patterns like 'ng' suffix, 'kah' suffix
        if re.search(r'\b\w+(ng|kah|lah|loh|kan|apa|apa\?)\b', text_lower):
            logger.debug("Language detection: Detected Malay based on pattern matching")
            return "malay"
        else:
            # Default to English if uncertain
            logger.debug("Language detection: No clear match, defaulting to English")
            return "english"
    
    # Return the language with more keyword matches
    if malay_count > english_count:
        logger.debug("Language detection: Detected Malay")
        return "malay"
    else:
        logger.debug("Language detection: Detected English")
        return "english"


def verify_claim(claim: str) -> dict:
    logger.info(f"Verifying: {claim[:80]}...")

    # ── Detect user's language ───────────────────────────────────────────────
    user_language = detect_language(claim)
    logger.info(f"Detected user language: {user_language}")

    # ── Step 1: LangChain RetrievalQA (local ChromaDB) ───────────────────────
    local_answer = ""
    local_source_docs = []
    local_source_urls = []

    if qa_chain is not None:
        try:
            result = qa_chain.invoke({"query": claim})
            local_answer = result.get("result", "")
            local_source_docs = result.get("source_documents", [])

            for doc in local_source_docs:
                url = doc.metadata.get("url", "")
                if url and url not in local_source_urls:
                    local_source_urls.append(url)

            logger.info(
                f"LangChain RetrievalQA: answer generated, "
                f"{len(local_source_docs)} source documents used."
            )
        except Exception as e:
            logger.error(f"LangChain RetrievalQA error: {e}", exc_info=True)

    if not local_answer.strip():
        logger.warning("LangChain RetrievalQA / Local RAG returned empty answer.")

    # ── Step 2: Retrieve live evidence from Sebenarnya.my and Google Fact Check API in parallel ──
    with ThreadPoolExecutor() as executor:
        future_sebenarnya = executor.submit(retrieve_sebenarnya_live, claim)
        future_google = executor.submit(retrieve_google_factcheck, claim)

        live_articles = future_sebenarnya.result()
        google_results = future_google.result()

    logger.info(
        f"Evidence: Local RAG answer length={len(local_answer)}, "
        f"Google results={len(google_results)}, Live articles={len(live_articles)}"
    )

    # ── Step 3: Compute score, verdict, confidence BEFORE building prompts ────
    score = compute_evidence_score(local_answer, google_results, live_articles)
    final_verdict, confidence = compute_verdict_and_confidence(score)

    # ── Step 4: Build supplementary live context ──────────────────────────────
    live_context_parts = []
    all_sources = list(local_source_urls)

    if live_articles:
        live_context_parts.append(
            "=== LIVE — Sebenarnya.my (latest articles, fetched in real time) ==="
        )
        for i, article in enumerate(live_articles):
            live_context_parts.append(
                f"[Live Article {i + 1}] \"{article['title']}\"\n"
                f"URL: {article['url']}\n"
                f"{article['body']}"
            )
            if article["url"] not in all_sources:
                all_sources.append(article["url"])

    if google_results:
        live_context_parts.append(
            "\n=== LIVE — Google Fact Check Tools API ==="
        )
        for item in google_results:
            live_context_parts.append(
                f"Claim: {item['claim_text']}\n"
                f"Verdict by {item['source_name']}: {item['rating']}\n"
                f"Review: {item['review_title']}\n"
                f"URL: {item['url']}"
            )
            if item["url"] and item["url"] not in all_sources:
                all_sources.append(item["url"])

    live_context = "\n\n".join(live_context_parts)

    # ── FALLBACK: ALL SOURCES EMPTY ──────────────────────────────────────────
    # If we have NO data from anywhere, return early with clear message
    if len(google_results) == 0 and len(live_articles) == 0:
        logger.warning("ALL data sources returned empty — returning fallback message")
        
        if user_language == "malay":
            fallback_answer = (
                "🔍 *Keputusan:* TIDAK DAPAT DISAHKAN\n\n"
                "📊 *Keyakinan:* 0%\n\n"
                "📋 *Penjelasan:*\n"
                "Saya tidak dapat menemukan informasi yang sesuai dalam sumber apa pun:\n"
                "• Pangkalan pengetahuan lokal (Sebenarnya.my)\n"
                "• Pencarian langsung Sebenarnya.my\n"
                "• Google Fact Check API\n\n"
                "Ini BUKAN bermakna klaim itu palsu — "
                "hanya bermakna saya tidak dapat mengesahkannya sekarang.\n\n"
                "⚠️ *Petua:* Apabila pengesahan tidak jelas, semak sebenarnya.my secara manual "
                "atau tanya sumber rasmi terus."
            )
        else:
            fallback_answer = (
                "🔍 *Verdict:* UNVERIFIED\n\n"
                "📊 *Confidence:* 0%\n\n"
                "📋 *Explanation:*\n"
                "I could not find matching information in any of my sources:\n"
                "• Local knowledge base (Sebenarnya.my)\n"
                "• Live Sebenarnya.my search\n"
                "• Google Fact Check API\n\n"
                "This does NOT mean the claim is false — "
                "it just means I cannot verify it right now.\n\n"
                "⚠️ *Tip:* When verification is unclear, check sebenarnya.my manually "
                "or ask official sources directly."
            )
        
        return {
            "answer": fallback_answer,
            "sources": [],
            "success": True,
            "score": 0,
            "verdict": "UNVERIFIED",
            "confidence": 0
        }

    # ── Step 5: Choose and build final prompt ─────────────────────────────────
    final_prompt_text = ""
    
    # Determine language instruction
    if user_language == "malay":
        language_instruction = "Respond ONLY in Bahasa Melayu. Do not use English."
    else:
        language_instruction = "Respond ONLY in English. Do not use Malay."
    
    if local_answer and live_context:
        # Both local RAG and live evidence available
        final_prompt_text = f"""You are TruthBot, a Malaysian AI fact-check explanation assistant.

LANGUAGE INSTRUCTION:
{language_instruction}

IMPORTANT:
- You are NOT allowed to change the verdict.
- You only explain the result.
- MUST respond in the specified language ONLY.

FINAL VERDICT: {final_verdict}
CONFIDENCE SCORE: {confidence}%

EVIDENCE SUMMARY:
- Local RAG: Found relevant articles
- Google Fact Check: {len(google_results)} results
- Sebenarnya.my articles: {len(live_articles)} found

User claim:
"{claim}"

TASK:
1. Explain why this verdict was assigned
2. Mention key evidence briefly
3. Do NOT change verdict
4. Keep answer short (2–4 sentences)
5. RESPOND ONLY IN THE LANGUAGE SPECIFIED ABOVE

FORMAT:

🔍 Verdict: {final_verdict}
📊 Confidence: {confidence}%

📋 Explanation:
[Your explanation here in {user_language.upper()} only]

⚠️ Tip: [One short practical tip in {user_language.upper()} only]"""

    elif live_context:
        # Only live evidence (no local RAG)
        final_prompt_text = f"""You are TruthBot, an AI-powered fake news and scam detection assistant for Malaysia.

LANGUAGE INSTRUCTION:
{language_instruction}

IMPORTANT:
- You are NOT allowed to change the verdict.
- You only explain the result.
- MUST respond in the specified language ONLY.

The local knowledge base had no relevant results for this claim.
Here is evidence retrieved live from fact-checking sources:

FINAL VERDICT: {final_verdict}
CONFIDENCE SCORE: {confidence}%

Evidence:
{live_context}

User claim:
"{claim}"

Explain the verdict in 2–4 sentences. RESPOND ONLY IN THE LANGUAGE SPECIFIED ABOVE.

FORMAT:

🔍 Verdict: {final_verdict}
📊 Confidence: {confidence}%

📋 Explanation:
[Your explanation here in {user_language.upper()} only]

🔗 Sources Checked:
{chr(10).join(f"• {s}" for s in all_sources) if all_sources else "• No direct sources found."}

⚠️ Tip: [One short practical tip in {user_language.upper()} only]"""

    elif local_answer:
        # Only local RAG (no live sources)
        final_prompt_text = f"""You are TruthBot, an AI-powered fake news and scam detection assistant for Malaysia.

LANGUAGE INSTRUCTION:
{language_instruction}

IMPORTANT:
- You are NOT allowed to change the verdict.
- You only explain the result.
- MUST respond in the specified language ONLY.

Based on local knowledge base only:

FINAL VERDICT: {final_verdict}
CONFIDENCE SCORE: {confidence}%

Local RAG answer:
{local_answer}

User claim:
"{claim}"

Explain the verdict in 2–4 sentences. RESPOND ONLY IN THE LANGUAGE SPECIFIED ABOVE.

FORMAT:

🔍 Verdict: {final_verdict}
📊 Confidence: {confidence}%

📋 Explanation:
[Your explanation here in {user_language.upper()} only]

🔗 Sources Checked:
{chr(10).join(f"• {s}" for s in all_sources) if all_sources else "• Local knowledge base only"}

⚠️ Tip: [One short practical tip in {user_language.upper()} only]"""

    else:
        # This should be caught by the FALLBACK check above, but just in case
        return {
            "answer": "⚠️ Sorry, I could not process your request. Please try again." if user_language == "english" else "⚠️ Maaf, saya tidak dapat memproses permintaan anda. Sila cuba lagi.",
            "sources": [],
            "success": False
        }

    # ── Step 6: Send final prompt to LangChain LLM ───────────────────────────
    try:
        logger.info("Sending prompt to LangChain LLM...")
        response = llm.invoke(final_prompt_text)
        answer = response.content.strip()
        logger.info(f"LLM response generated: {len(answer)} chars")
        
        return {
            "answer": answer,
            "sources": all_sources,
            "success": True,
            "score": score,
            "verdict": final_verdict,
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"LangChain LLM final generation error: {e}", exc_info=True)
        
        # Fallback: return a structured response even if LLM fails
        if user_language == "malay":
            error_answer = (
                "⚠️ *Ralat Sistem*\n\n"
                f"🔍 *Keputusan:* {final_verdict}\n"
                f"📊 *Keyakinan:* {confidence}%\n\n"
                "📋 *Penjelasan:*\n"
                "Saya menghadapi ralat semasa menjana penjelasan. "
                "Walau bagaimanapun, berdasarkan bukti yang tersedia, keputusan di atas telah dikira.\n\n"
                f"🔗 *Sumber Disemak:* {len(all_sources)} sumber ditemui\n\n"
                "⚠️ *Petua:* Cuba nyatakan semula soalan anda atau semak sebenarnya.my secara terus."
            )
        else:
            error_answer = (
                "⚠️ *System Error*\n\n"
                f"🔍 *Verdict:* {final_verdict}\n"
                f"📊 *Confidence:* {confidence}%\n\n"
                "📋 *Explanation:*\n"
                "I encountered an error while generating the explanation. "
                "However, based on available evidence, the verdict above was computed.\n\n"
                f"🔗 *Sources Checked:* {len(all_sources)} sources found\n\n"
                "⚠️ *Tip:* Try rephrasing your question or check sebenarnya.my directly."
            )
        
        return {
            "answer": error_answer,
            "sources": all_sources,
            "success": False,
            "score": score,
            "verdict": final_verdict,
            "confidence": confidence
        }


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to *TruthBot*!\n\n"
        "I help you verify news, detect fake information, and identify online scams — "
        "powered by LangChain, RAG, and live fact-checking sources.\n\n"
        "*How to use:*\n"
        "Simply send me any:\n"
        "• News headline or claim\n"
        "• Suspicious message you received\n"
        "• Text you want to fact-check\n\n"
        "I'll search my local knowledge base and live sources, "
        "then give you a verdict backed by evidence.\n\n"
        "🇲🇾 Supports *English* and *Bahasa Melayu*\n\n"
        "Type /help for more commands.",
        parse_mode="Markdown"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "*TruthBot Commands*\n\n"
        "/start — Welcome message\n"
        "/help — Show this help\n"
        "/sources — Hybrid System uses weighted scoring + confidence model\n"
        "/tips — How to spot fake news\n"
        "/reset — Clear your conversation history\n\n"
        "*How to verify:*\n"
        "Just type or paste any news, claim, or suspicious message "
        "and I'll check it for you.",
        parse_mode="Markdown"
    )


async def sources_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db_size = 0
    if vectorstore:
        try:
            db_size = vectorstore._collection.count()
        except Exception:
            pass

    await update.message.reply_text(
        "*Fact-Checking Sources Used by TruthBot*\n\n"
        "📦 *Local Dataset (LangChain + ChromaDB)*\n"
        f"   └ {db_size:,} chunks from Sebenarnya.my articles\n"
        f"   └ Similarity threshold: ≥{int(SIMILARITY_THRESHOLD * 100)}% — "
        f"only genuinely relevant chunks used\n"
        "   └ Searched via LangChain RetrievalQA chain\n\n"
        "🌐 *Live — Sebenarnya.my*\n"
        "   └ Malaysia's official MCMC fact-checking portal\n"
        "   └ Queried live to catch the latest articles\n\n"
        "🌐 *Live — Google Fact Check Tools API*\n"
        "   └ International fact-check database\n\n"
        "🤖 *DeepSeek V3.1 via NVIDIA NIM*\n"
        "   └ Accessed via LangChain ChatOpenAI wrapper\n\n"
        "_TruthBot uses Hybrid RAG — LangChain orchestrates the full pipeline._",
        parse_mode="Markdown"
    )


async def tips_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "*Tips to Spot Fake News 🔍*\n\n"
        "1️⃣ Check the source — is it a known credible news outlet?\n"
        "2️⃣ Look for emotional language — fake news often uses alarming words\n"
        "3️⃣ Search for the same story on multiple sites\n"
        "4️⃣ Check the date — old news is often recycled as new\n"
        "5️⃣ Verify images using Google Reverse Image Search\n"
        "6️⃣ Check Sebenarnya.my for Malaysian news verification\n\n"
        "When in doubt — don't share! Send it to me first 😊",
        parse_mode="Markdown"
    )


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_conversations.pop(user_id, None)
    await update.message.reply_text("✅ Your conversation history has been cleared.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_message = update.message.text.strip()

    if not user_message:
        return

    logger.info(f"User {user_id}: {user_message[:80]}")

    # ── Send typing indicator ──────────────────────────────────────
    try:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )
    except Exception as e:
        logger.warning(f"Failed to send typing indicator: {e}")

    # ── Track conversation ────────────────────────────────────────
    if user_id not in user_conversations:
        user_conversations[user_id] = []
    user_conversations[user_id].append({"role": "user", "content": user_message})

    # ── Verify claim (with error handling) ─────────────────────────
    try:
        result = verify_claim(user_message)
    except Exception as e:
        logger.error(f"verify_claim() exception: {e}", exc_info=True)
        result = {
            "answer": (
                "⚠️ *System Error*\n\n"
                "An error occurred while processing your request. "
                "Please try again in a moment."
            ),
            "sources": [],
            "success": False
        }

    # ── Add to conversation history ───────────────────────────────
    user_conversations[user_id].append(
        {"role": "assistant", "content": result["answer"]}
    )

    if len(user_conversations[user_id]) > 20:
        user_conversations[user_id] = user_conversations[user_id][-20:]

    # ── Send response (with fallback parsing modes) ───────────────
    response_text = result.get("answer", "No response generated")
    
    try:
        # Try with Markdown first (preferred)
        await update.message.reply_text(response_text, parse_mode="Markdown")
        logger.info(f"Response sent successfully (Markdown)")
    except Exception as md_error:
        logger.warning(f"Markdown parsing failed: {md_error}. Retrying without Markdown...")
        try:
            # Fallback: send without Markdown
            await update.message.reply_text(response_text, parse_mode=None)
            logger.info(f"Response sent successfully (plain text)")
        except Exception as plain_error:
            logger.error(f"Failed to send response in both modes: {plain_error}", exc_info=True)
            # Last resort: send a simple plain-text error
            try:
                await update.message.reply_text(
                    "Sorry, I encountered an issue sending the response. "
                    "The verification may have failed. Please try again."
                )
            except Exception as final_error:
                logger.critical(f"CRITICAL: Cannot send ANY response: {final_error}")


def main():
    logger.info("MAIN FUNCTION STARTED") 
    logger.info("Starting TruthBot — LangChain Hybrid RAG pipeline...")

    if qa_chain is None:
        logger.warning(
            "⚠️  LangChain RetrievalQA chain not available (ChromaDB not loaded). "
            "Run scraper.py → build_vectordb.py to enable local RAG. "
            "Live retrieval (Sebenarnya.my + Google) is still active."
        )

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("sources", sources_command))
    app.add_handler(CommandHandler("tips", tips_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("TruthBot is running. Press Ctrl+C to stop.")
    logger.info("About to start polling...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()