"""
build_vectordb.py — TruthBot Vector Database Builder (LangChain Version)
=========================================================================
Uses LangChain components to build the ChromaDB vector database:

  LangChain components used here:
  ┌─────────────────────────────────────────────────────┐
  │  Document          — LangChain's standard text unit │
  │  RecursiveCharacter│
  │  TextSplitter      — LangChain's smart text chunker │
  │  HuggingFaceEmbed- │
  │  dings             — LangChain's embedding wrapper  │
  │  Chroma            — LangChain's ChromaDB wrapper   │
  └─────────────────────────────────────────────────────┘

Run AFTER scraper.py, and BEFORE truthbot_FIXED.py (chatbot_telegram.py).
Re-run whenever the dataset is updated.

Installation:
    pip install -U langchain langchain-core langchain-community langchain-text-splitters \
    langchain-huggingface langchain-chroma chromadb sentence-transformers

Usage:
    python build_vectordb.py

Output:
    ./chroma_db/
"""

import json
import logging
import os
import shutil

# ── LangChain imports ──────────────────────────────────────────────
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
# ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

ARTICLES_FILE   = "sebenarnya_articles.json"
CHROMA_DB_PATH  = "./chroma_db"
COLLECTION_NAME = "sebenarnya_articles"

# Use a smaller, faster model for embedding. "all-MiniLM-L6-v2" is a good balance of speed and quality.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# LangChain RecursiveCharacterTextSplitter settings
# RecursiveCharacterTextSplitter is smarter than plain chunking —
# it tries to split on paragraphs first, then sentences, then words,
# so chunks are always meaningful and never cut mid-sentence.
CHUNK_SIZE    = 800   # characters per chunk
CHUNK_OVERLAP = 150   # overlap to preserve context at chunk boundaries


def build_database():
    """
    Build ChromaDB vector database from scraped articles.
    
    Steps:
    1. Load articles from JSON
    2. Convert to LangChain Documents
    3. Split documents into chunks
    4. Generate embeddings
    5. Store in ChromaDB
    """
    
    # ── Load dataset ───────────────────────────────────────────────
    if not os.path.exists(ARTICLES_FILE):
        logger.error(
            f"'{ARTICLES_FILE}' not found. Run scraper.py first."
        )
        return

    with open(ARTICLES_FILE, "r", encoding="utf-8") as f:
        articles = json.load(f)

    if not articles:
        logger.error("Dataset is empty. Run scraper.py first.")
        return

    logger.info(f"Loaded {len(articles)} articles from '{ARTICLES_FILE}'")

    # ── Step 1: Convert articles to LangChain Documents ───────────
    # LangChain uses a standard Document object with:
    #   .page_content — the text
    #   .metadata     — dict of extra info (url, title, date)
    # This is the standard unit that all LangChain components understand.
    logger.info("Converting articles to LangChain Documents...")
    raw_documents = []
    for article in articles:
        title   = article.get("title",   "")
        content = article.get("content", "")
        if not content:
            logger.debug(f"Skipping article with no content: {title}")
            continue

        # Prepend title so every chunk carries article context
        full_text = f"Tajuk / Title: {title}\n\n{content}"

        raw_documents.append(
            Document(
                page_content=full_text,
                metadata={
                    "title": title,
                    "url":   article.get("url",  ""),
                    "date":  article.get("date", "")
                }
            )
        )

    logger.info(f"Created {len(raw_documents)} LangChain Documents.")

    if not raw_documents:
        logger.error("No valid documents to process. Check your JSON file.")
        return

    # ── Step 2: Split with LangChain's RecursiveCharacterTextSplitter
    # This is LangChain's recommended text splitter.
    # It splits recursively: tries "\n\n" first (paragraphs),
    # then "\n" (lines), then " " (words) — always keeping chunks meaningful.
    logger.info("Splitting documents with LangChain RecursiveCharacterTextSplitter...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]  # Priority order for splitting
    )
    split_documents = text_splitter.split_documents(raw_documents)
    logger.info(f"Split into {len(split_documents)} chunks.")

    if not split_documents:
        logger.error("Failed to split documents. Check your content.")
        return

    # ── Step 3: Load LangChain embedding model ────────────────────
    # HuggingFaceEmbeddings is LangChain's wrapper around sentence-transformers.
    # It converts text into vectors that ChromaDB can store and search.
    logger.info(f"Loading LangChain HuggingFaceEmbeddings: {EMBEDDING_MODEL} ...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}  # Normalise for cosine similarity
        )
        logger.info("Embeddings model ready.")
    except Exception as e:
        logger.error(f"Failed to load embeddings model: {e}", exc_info=True)
        return

    # ── Step 4: Backup existing ChromaDB (if any) ─────────────────
    if os.path.exists(CHROMA_DB_PATH):
        backup_path = CHROMA_DB_PATH + ".backup"
        logger.info(f"Backing up existing database to '{backup_path}'...")
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
        shutil.copytree(CHROMA_DB_PATH, backup_path)
        shutil.rmtree(CHROMA_DB_PATH)  # Remove original for fresh build
        logger.info("Backup complete. Creating fresh database...")

    # ── Step 5: Create and persist ChromaDB ───────────────────────
    # LangChain's Chroma wrapper handles all ChromaDB operations.
    # persist_directory ensures data is saved to disk.
    logger.info(f"Creating ChromaDB at '{CHROMA_DB_PATH}'...")
    try:
        vectorstore = Chroma.from_documents(
            documents=split_documents,
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH,
            collection_name=COLLECTION_NAME,
            collection_metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        logger.info("✅ ChromaDB created successfully!")
        
        # Verify the database
        chunk_count = vectorstore._collection.count()
        logger.info(f"✅ Database contains {chunk_count} chunks ready for retrieval.")
        
    except Exception as e:
        logger.error(f"Failed to create ChromaDB: {e}", exc_info=True)
        return

    # ── Step 6: Test the database ──────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("Testing database with sample queries...")
    logger.info("="*60)
    
    try:
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": 0.2,
                "k": 3
            }
        )
        
        # Test queries in both languages
        test_queries = [
            "fakta palsu COVID",  # Malay
            "false claims about COVID",  # English
        ]
        
        for query in test_queries:
            logger.info(f"\nTest query: '{query}'")
            results = retriever.invoke(query)
            logger.info(f"  Found {len(results)} results:")
            for i, doc in enumerate(results, 1):
                title = doc.metadata.get("title", "Unknown")
                logger.info(f"    {i}. {title[:60]}...")
                
    except Exception as e:
        logger.warning(f"Error during testing: {e}")

    logger.info("\n" + "="*60)
    logger.info("✅ DATABASE BUILD COMPLETE!")
    logger.info("="*60)
    logger.info(f"📦 Location: {CHROMA_DB_PATH}")
    logger.info(f"📊 Total chunks: {chunk_count}")
    logger.info(f"🔍 Ready for retrieval in truthbot_FIXED.py")
    logger.info("="*60)


if __name__ == "__main__":
    logger.info("Starting ChromaDB build process...")
    build_database()
    logger.info("Build process finished.")