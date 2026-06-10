"""
TruthBot: AI-Based Telegram Chatbot for Fake News Detection
============================================================
HYBRID RAG PIPELINE — powered by LangChain + ChromaDB + NVIDIA NIM
"""

import os
import logging
from urllib import response
import requests
import re
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from openai import OpenAI
from openai import RateLimitError


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

user_conversations = {} # In-memory conversation history per user (user_id -> list of messages)

NVIDIA_API_KEY        = os.getenv("NVIDIA_API_KEY")
TELEGRAM_BOT_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN")
GOOGLE_FACT_CHECK_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY", "")
NIM_MODEL             = os.getenv("NIM_MODEL", "openai/gpt-oss-20b")
NIM_API_BASE          = os.getenv("NIM_API_BASE", "https://integrate.api.nvidia.com/v1")

CHROMA_DB_PATH        = "./chroma_db"
COLLECTION_NAME       = "sebenarnya_articles"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

WEIGHT_GOOGLE         = 0.3 # Google Fact Check is weighted heavily due to its authoritative fact-checking content, but we still want to balance it with local and live sources for a more comprehensive verification
WEIGHT_SEBENARNYA     = 0.4 # Sebenarnya.my is weighted moderately — it's a trusted local source but may not have coverage of every claim, and some articles may be outdated
WEIGHT_LOCAL          = 0.3 # Local RAG is weighted less than live sources to avoid over-reliance on potentially outdated information in the vector store
SIMILARITY_THRESHOLD  = 0.4 # Only consider chunks with ≥40% similarity as relevant
MAX_LOCAL_CHUNKS      = 2 # Limit local RAG to top 2 most relevant chunks to maintain answer quality and relevance
MAX_LIVE_ARTICLES     = 2 # Limit live Sebenarnya.my retrieval to top 2 articles to ensure response speed and relevance


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
    temperature=0.2,# Low temperature for more factual and deterministic responses in a fact-checking context
    max_tokens=300, # Limit max tokens to control response length and speed
    timeout=40 # Longer timeout for LLM calls since they may take more time when processing complex prompts with multiple evidence sources
)
logger.info("LangChain LLM ready.") # Log after successful LLM initialization to confirm that the bot is ready to process claims and generate responses

logger.info(f"Loading LangChain HuggingFaceEmbeddings: {EMBEDDING_MODEL} ...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
logger.info("LangChain Embeddings ready.")

vectorstore = None
retriever   = None
try: # Attempt to load ChromaDB vector store (local RAG) — if it fails, we log a warning and continue with live retrieval only
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 20 # Fetch more documents for MMR to rerank, but only return top 3 most relevant after reranking to ensure quality and relevance of local RAG evidence
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


#─────────────────────────────────────────────────────────────────────────────
#  RETRIEVAL FUNCTIONS
#─────────────────────────────────────────────────────────────────────────────
def retrieve_sebenarnya_live(claim: str) -> list: # Retrieve live articles from Sebenarnya.my based on a claim
    """
    Search Sebenarnya.my live and return only relevant articles.
    """

    try:
        logger.info(f"Fetching live from Sebenarnya.my for: {claim}") 

        # ---------------------------------------
        # Build cleaner search query
        # ---------------------------------------

        stopwords = { # Common Malay and English stopwords to filter out for better search relevance
            "ada", "adalah", "yang", "dan", "atau",
            "di", "ke", "dari", "untuk", "dengan",
            "the", "is", "are", "was", "were",
            "in", "on", "at", "to", "for"
        }

        keywords = re.findall(r"\w+", claim.lower())

        keywords = [
            w for w in keywords
            if len(w) > 2 and w not in stopwords
        ]

        search_text = " ".join(keywords[:5])

        logger.info(f"Search keywords: {search_text}")

        query = requests.utils.quote(search_text)

        search_url = f"https://sebenarnya.my/?s={query}"

        logger.info(f"Search URL: {search_url}")

        r = requests.get(
            search_url,
            headers=HEADERS,
            timeout=5
        )

        if r.status_code != 200:
            logger.warning(
                f"Sebenarnya search returned {r.status_code}"
            )
            return []

        soup = BeautifulSoup(r.text, "html.parser")

        links = []

        # ---------------------------------------
        # ONLY use actual search result titles
        # ---------------------------------------

        selectors = [
            "h2.entry-title a",
            "h1.entry-title a",
            ".entry-title a",
            ".post-title a"
        ]

        for selector in selectors:

            for a in soup.select(selector):

                href = a.get("href", "").strip()

                title = a.get_text(
                    " ",
                    strip=True
                ).lower()

                if not href:
                    continue

                if "sebenarnya.my" not in href:
                    continue

                relevance = 0

                for keyword in keywords:
                    if keyword in title:
                        relevance += 1

                links.append(
                    (
                        relevance,
                        href,
                        title
                    )
                )

        if not links: # If no links found from the search results, we log a warning and return an empty list to avoid unnecessary processing
            logger.warning(
                "No search results found."
            )
            return []

        # ---------------------------------------
        # Sort by relevance
        # ---------------------------------------

        links.sort(
            key=lambda x: x[0],
            reverse=True
        )
       
        logger.info("=== SEARCH RESULTS ===")

        for score, href, title in links[:10]:
         logger.info(
         f"[score={score}] {title} -> {href}"
    )

        logger.info("======================")


        MIN_RELEVANCE_SCORE = 2 # Only consider articles that have at least 2 keyword matches in the title as relevant to ensure we are selecting genuinely relevant articles for evidence, and log the relevance scores to monitor and adjust this threshold if necessary based on observed results.

        unique_links = []
        seen = set()

        for score, href, title in links:

            # Skip unrelated articles
            if score < MIN_RELEVANCE_SCORE:
                continue

            if href in seen:
                continue

            seen.add(href)
            unique_links.append(href)

        unique_links = unique_links[:MAX_LIVE_ARTICLES]

        logger.info(
            f"Selected {len(unique_links)} relevant article(s)"
        )

        # ---------------------------------------
        # Download articles and extract content from the top 2 most relevant links to ensure response speed and relevance. We log each step and any issues encountered to ensure transparency in the retrieval process.
        # ---------------------------------------

        articles = []

        for url in unique_links:

            try:

                logger.info(
                    f"Fetching article: {url}"
                )

                article_response = requests.get(
                    url,
                    headers=HEADERS,
                    timeout=5
                )

                article_soup = BeautifulSoup(
                    article_response.text,
                    "html.parser"
                )

                title_tag = article_soup.select_one(
                    "h1.entry-title"
                )

                title = (
                    title_tag.get_text(
                        strip=True
                    )
                    if title_tag
                    else "Untitled"
                )

                content_selectors = [
                    ".entry-content",
                    ".td-post-content",
                    ".post-content",
                    "article",
                    ".content"
                ]
                content_tag = None

                for selector in content_selectors:
                        content_tag = article_soup.select_one(selector)
                        if content_tag:
                            logger.info(
                                f"Found content using {selector}"
                            )
                            break

                if not content_tag:
                    logger.warning(
                        f"No content found: {url}"
                    )
                    continue

                for tag in content_tag.select(
                    "script, style, nav, footer"
                ):
                    tag.decompose()

                body = content_tag.get_text(
                    separator=" ",
                    strip=True
                )

                if len(body) < 200:
                    continue

                articles.append({
                    "title": title,
                    "url": url,
                    "body": body[:800] # Limit to first 800 chars to ensure relevance and speed
                })

                logger.info(
                    f"✓ Extracted article ({len(body)} chars)"
                )

            except Exception as e:
                logger.warning(
                    f"Error reading article {url}: {e}"
                )

        logger.info(
            f"Live retrieval completed. "
            f"Articles={len(articles)}"
        )

        return articles

    except Exception as e:
        logger.error(
            f"Live retrieval error: {e}",
            exc_info=True
        )
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
        logger.error("Google Fact Check API timeout (3s)")
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
        score += 0

    if local_answer: 
        if "FALSE" in local_answer:
            score -= WEIGHT_LOCAL * 1
        elif "TRUE" in local_answer:
            score += WEIGHT_LOCAL * 1

    return score


def compute_verdict_and_confidence(score):

    confidence = min(
        abs(score) * 50,
        100
    )

    if abs(score) < 0.3:
        verdict = "UNVERIFIED"

    elif score >= 1:
        verdict = "TRUE"

    elif score <= -1:
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
        'tapi', 'kerana', 'apa', 'siapa', 'mana','bagaimana',
        'sudah', 'akan', 'dapat', 'harus', 'perlu', 'boleh',
        'negara', 'klaim', 'berita', 'palsu', 'bohong', 'nyata',
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
    logger.info(f"Verifying claim: {claim[:100]}")

    user_language = detect_language(claim)

    # =====================================================
    # STEP 1: Retrieve Chroma documents ONLY (NO LLM CALL)
    # =====================================================

    local_context_parts = []
    local_source_urls = []

    try:
        if vectorstore:

            results = vectorstore.similarity_search_with_score(
                claim,
                k=MAX_LOCAL_CHUNKS # We fetch more documents for MMR to rerank, but we will only use the top 3 most relevant chunks after reranking to ensure quality and relevance of local RAG evidence
            )

            logger.info("=== CHROMA RESULTS ===")

            docs = []
            MAX_DISTANCE = 0.45 # Only consider chunks with similarity score ≤0.45 as relevant evidence for the local RAG context, and log the distance scores to monitor the relevance of retrieved chunks and adjust this threshold if necessary based on observed results.

            for doc, score in results:

                logger.info(
                    f"DISTANCE={score:.4f} | {doc.metadata.get('title')}"
                )

                if score <= MAX_DISTANCE:
                    docs.append(doc) # Only consider documents within the similarity threshold as relevant evidence for the local RAG context, and log the distance scores to monitor the relevance of retrieved chunks and adjust the threshold if necessary based on observed results. We also limit the number of local chunks used in the final prompt to maintain answer quality and relevance.

            logger.info("======================")

            for doc in docs[:MAX_LOCAL_CHUNKS]: # Limit to most relevant chunks to maintain answer quality and relevance

                title = doc.metadata.get("title", "Unknown")
                url = doc.metadata.get("url", "")

                excerpt = doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content

                local_context_parts.append(
                    f"""
Title: {title}
URL: {url}

{excerpt}
"""
                )

                if url and url not in local_source_urls:
                    local_source_urls.append(url)

        logger.info(
            f"Retrieved {len(local_context_parts)} local chunks"
        )

    except Exception as e:
        logger.error(
            f"Local retrieval failed: {e}",
            exc_info=True
        )

    # =====================================================
    # STEP 2: Retrieve live sources in parallel (Google Fact Check + Sebenarnya.my)
    # =====================================================

    with ThreadPoolExecutor(max_workers=2) as executor:

        future_google = executor.submit(
            retrieve_google_factcheck,
            claim
        )

        future_live = executor.submit(
            retrieve_sebenarnya_live,
            claim
        )

        google_results = future_google.result()
        live_articles = future_live.result()

    logger.info(
        f"Google={len(google_results)} | "
        f"Live={len(live_articles)}"
    )

    # =====================================================
    # STEP 3: Build Google context for prompt (only top 3 results, prioritising authoritative fact-checks)
    # =====================================================

    # =====================================
    # Filter Google results by claim match
    # =====================================

    filtered_google = []

    claim_words = set(claim.lower().split())

    for g in google_results:

        google_claim = g.get("claim_text", "").lower()

        matches = sum(
            1 for word in claim_words
            if len(word) > 3 and word in google_claim
        )

        if matches >= 2:
            filtered_google.append(g)

    logger.info(
        f"Google results after filtering: "
        f"{len(filtered_google)}"
    )

    google_results = filtered_google

    google_context = ""

    for item in google_results[:3]: # Limit to top 3 Google Fact Check results to ensure relevance and speed, and to prioritise the most authoritative fact-checks while avoiding overwhelming the prompt with too much information from this secondary source

        google_context += f"""
Claim: {item.get('claim_text', '')}

Rating: {item.get('rating', '')}

Publisher: {item.get('source_name', '')}

Review Title:
{item.get('review_title', '')}

URL:
{item.get('url', '')}

----------------------------------
"""

    # =====================================================
    # STEP 4: Build Live context for prompt (only top 2 most relevant articles to ensure response speed and relevance)
    # =====================================================

    live_context = ""

    live_urls = []

    for article in live_articles[:2]:

        live_context += f"""
Title:
{article['title']}

URL:
{article['url']}

Excerpt:
{article['body'][:600] + "..." if len(article['body']) > 600 else article['body']}

----------------------------------
"""

        live_urls.append(article["url"])

    # =====================================================
    # STEP 5: Collect all source URLs for final prompt and response (local RAG + Google + Live)
    # =====================================================

    # Only keep the most relevant sources

    all_sources = []

    # 1. Live articles first (highest priority)
    for url in local_source_urls[:1]:
        if url and url not in all_sources:
            all_sources.append(url)

    # 2. Google Fact Check results
    for g in google_results[:1]:
        url = g.get("url", "")
        if url and url not in all_sources:
            all_sources.append(url)

    # 3. Local Chroma sources
    for url in live_urls[:2]:
        if url and url not in all_sources:
            all_sources.append(url)

    # =====================================================
    # STEP 6: Fallback if no evidence found
    # =====================================================

    if (
        not local_context_parts
        and not google_results
        and not live_articles
    ):

        return {
            "success": True,
            "answer": (
                "🔍 Verdict: UNVERIFIED\n\n"
                "📊 Confidence: 0%\n\n"
                "📋 Explanation:\n"
                "No relevant evidence was found in "
                "ChromaDB, Google Fact Check, "
                "or live Sebenarnya.my sources.\n\n"
                "🔗 Sources:\n"
                "None"
            ),
            "sources": []
        }

    # =====================================================
    # STEP 7: Build ONE final prompt with ALL evidence (local RAG + Google + Live) and send to LLM for final verification answer
    # =====================================================

    final_prompt = f"""
        You are TruthBot, a Malaysian fact-checking assistant.

        Reply in the SAME language as the claim.

        Use ONLY the evidence provided.
        Do NOT use outside knowledge.
        Do NOT invent facts.

        Verdict:
        TRUE = evidence supports claim
        FALSE = evidence contradicts claim
        MISLEADING = partially true but missing context
        UNVERIFIED = insufficient evidence

        Confidence:
        95-100 = very strong evidence
        80-94 = strong evidence
        60-79 = moderate evidence
        40-59 = weak/conflicting evidence
        0-39 = little or no evidence

        Evidence:
        Claim: {claim}

        Local:
        {chr(10).join(local_context_parts)}

        Google:
        {google_context}

        Live:
        {live_context}

        Rules:
        - Prefer newer sources if conflict exists.
        - Mention only key evidence.
        - Maximum 3 short explanation sentences.
        - Lower confidence when evidence is weak.

        Output exactly:
        🔍 Verdict: [TRUE/FALSE/MISLEADING/UNVERIFIED]

        📊 Confidence: [0-100%]

        📋 Explanation:
        [short explanation]

        💡 Tip:
        [verification tip]

"""

    # =====================================================
    # STEP 8: SINGLE LLM CALL with all evidence and final prompt (with error handling for rate limits and other LLM issues)
    # =====================================================
    
    try:

        logger.info(f"Prompt length = {len(final_prompt)} chars") # Log the final prompt length to monitor for potential issues with prompt size and to ensure that the prompt is being constructed correctly with all evidence included.

        response = llm.invoke(final_prompt)

        logger.info(f"Usage: {response.usage_metadata}")

        raw = response.content

        if isinstance(raw, list):
            answer = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in raw
            ).strip()
        else:
            answer = raw.strip()

            # ==========================================
            # Append REAL source URLs manually
            # ==========================================

            if all_sources:

                answer += "\n\n🔗 Sources:\n"

                for url in all_sources[:3]: # Limit to top 3 sources in the final answer to maintain relevance and avoid overwhelming the user with too many links, while ensuring we provide the most important sources that back the verification result.
                    answer += f"• {url}\n"

        logger.info(
            "Single LLM response generated successfully." # Log after successful LLM response to confirm that the bot is able to generate answers based on the provided evidence and prompt, and to help identify any issues in the LLM call or response processing.
        )

        return {
            "success": True,
            "answer": answer,
            "sources": all_sources
        }

    except RateLimitError as e: # NVIDIA NIM rate limit exceeded

        logger.error(
            f"Rate limit error: {e}",
            exc_info=True
        )

        return { # Even if the LLM call fails due to rate limits, we still return the sources we found for transparency and to provide some value to the user, along with a clear message about the issue.  
            "success": False,
            "answer": (
                "⚠️ TruthBot is currently experiencing high traffic.\n\n"
                "Please try again in a few minutes."
            ),
            "sources": all_sources
        }

    except Exception as e:

        logger.error(
            f"LLM error: {e}",
            exc_info=True
        )

        return {
            "success": False,
            "answer": (
                "⚠️ System Error\n\n"
                "Unable to generate verification result." # LLM call failed, but we still return sources for transparency
            ),
            "sources": all_sources
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
        "   └ Searched directly via Chroma retriever\n\n"
        "🌐 *Live — Sebenarnya.my*\n"
        "   └ Malaysia's official MCMC fact-checking portal\n"
        "   └ Queried live to catch the latest articles\n\n"
        "🌐 *Live — Google Fact Check Tools API*\n"
        "   └ International fact-check database\n\n"
        "🤖 *OpenAI GPT via NVIDIA NIM*\n"
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


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE): # Clear conversation history for the user  
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
    if user_id not in user_conversations: # Start a new conversation thread for this user
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

    if len(user_conversations[user_id]) > 20: # Keep only the last 20 messages to limit memory usage
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

    if retriever is None:
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
    logger.info("About to start polling...") # Log before polling to confirm the bot has started successfully and is ready to receive messages
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()