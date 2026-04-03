# Minimal Python RAG Demo (Streamlit)

A practical end-to-end RAG demo for technical PDFs, with explicit **RAG vs LLM-only** comparison and strict anti-hallucination handling.

Sample document for testing: **C-MAPSS User's Guide**.  
Default PDF path: `data/document.pdf`

## What This Demo Shows

- Ingestion of a local technical PDF
- Structure-aware chunking with metadata
- Hybrid retrieval: BM25 + semantic embeddings + RRF fusion + reranking
- Grounded answer generation with Gemini
- Retrieval-only fallback when Gemini is unavailable
- Side-by-side optional comparison:
  - `RAG Mode (grounded answers)`
  - `LLM-only Mode (may hallucinate)`

## Quick Start

```bash
# 1. Setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. Run
streamlit run app.py

# 3. Use
# - Upload or point to data/document.pdf
# - Click "Index Document" (one-time, 20-45 seconds)
# - Enter your question
# - See both RAG and LLM-only answers
```

## User Guide

### Workflow

The app has a clear two-step workflow:

1. **Step 1️⃣: Index Document** (one-time setup)
   - Select PDF source: file path or upload
   - Click "Index Document" button
   - App extracts text → chunks → builds semantic index
   - Progress shown: Extracting... → Chunking... → Building embeddings...
   - Takes 20-45 seconds for a typical 100-page PDF (one-time only)
   - Status shows: "✓ Document indexed: C-MAPSS (847 chunks, indexed at 2026-04-02 10:30:45)"

2. **Step 2️⃣: Ask Questions** (instant, after indexing)
   - Enter your question in the text field
   - Press Enter OR click "Query" button
   - Results appear in <1 second (retrieves from cached index)
   - Ask as many questions as you want without re-indexing

### Modes Explained

#### RAG Mode (Grounded Answers)
- **What happens**: Question + retrieved document evidence → Gemini generates answer
- **Reliability**: High. Answers are grounded in the document.
- **Speed**: <1 second per query (fast)
- **Display**: Shows answer + Retrieved Chunks (rank, score, page, section)
- **When to use**: Real document QA, reliable answers, learning how RAG works

#### LLM-only Mode (Ungrounded, Educational)
- **What happens**: Question only (no document context) → Gemini generates answer
- **Reliability**: Low. Can hallucinate with confidence.
- **Speed**: <1 second per query
- **Display**: Shows answer only (no chunks, since no retrieval used)
- **Status**: Badge says "LLM-only Mode (may hallucinate)"
- **When to use**: Learning about hallucinations, understanding model biases, comparing behaviors

### Example Queries

Click any button in "Example Queries" section to auto-fill the question field:

- "What assumptions does this simulation model make?"
- "How are engine degradation effects represented?"
- "Which section discusses model outputs and limitations?"
- "What is the closed-loop analysis section about?"

### Understanding Retrieved Chunks

After a RAG query, you see 5 retrieved chunks in ranked order:

```
Rank 1 | Score: 0.8791 | Confidence: 0.544
Page 8 | Section 1.0 | Chunk #16
[Text snippet from the document...]
```

- **Rank**: Order by relevance (1 = most relevant)
- **Score**: Raw retrieval score (higher = more similar to question)
- **Confidence**: Normalized confidence (0.0–1.0) whether this chunk answers the question
- **Page**: Which page of the document
- **Section**: Which numbered section (if detected)
- **Text snippet**: First 320 characters of the chunk

### Caching & Performance

The `.cache/` folder stores indexed data to speed up repeated use:

**First time with a PDF (cold start):**
- 20-45 seconds for a 100-page PDF
- Computes embeddings using `sentence-transformers/all-mpnet-base-v2`
- Saves chunks, FAISS index, embeddings to `.cache/`

**Same PDF, new session (warm start):**
- 5 seconds
- Loads chunks from JSON, embeddings from disk, rebuilds FAISS in memory

**Subsequent queries in same session:**
- <1 second
- Everything already in memory

**Key insight**: Identical PDFs (same content, different filenames) are detected by SHA256 hash and reuse cache automatically.

## LLM-Only Mode Guide

### Purpose

LLM-only mode exists to **educate** and **demonstrate** the hallucination problem:

- Compare RAG (grounded) vs LLM-only (ungrounded) side-by-side
- See how RAG reduces uncertainty and fabrication
- Understand when pure LLMs struggle with domain-specific facts

### When LLM-only is Available

- ✓ **Enabled**: You have a valid Gemini API key (env var, secrets, or sidebar)
- ✗ **Disabled**: No API key configured (RAG retrieval-only mode is available instead)

### How to Use

1. Configure a valid Gemini API key
2. In sidebar, select "Mode" → choose "LLM-only"
3. Ask a question
4. See the ungrounded answer (may be wrong or hallucinatory)

### Side-by-Side Comparison

Enable "Show optional RAG vs LLM-only comparison" checkbox to see both answers side-by-side:

**Left column (RAG):**
- Answer grounded in retrieved document chunks
- Shows where it found the answer
- Conservative: says "I cannot find this information in the document" if no evidence

**Right column (LLM-only):**
- Answer from the model's training, no document reference
- May sound confident but be inaccurate
- No Retrieved Chunks shown (nothing to retrieve from)

### Example: RAG vs LLM-only Contrast

**Question**: "What is the exact fuel flow rate in the closed-loop system?"

**RAG answer** (grounded):
> I cannot find this information in the document.

**LLM-only answer** (ungrounded):
> The closed-loop system typically operates at 500 lb/hr fuel flow rate.
> (Sounds confident, but may not be true; no document evidence provided)

### Limitations

- LLM-only has no document constraints; answers are from model training data
- May confidently state facts that are wrong or outdated
- Useful for comparison and learning, not reliable for actual QA

### Recommendation

- **Use RAG Mode** for real document QA and reliable answers
- **Use LLM-only Mode** for learning how RAG helps, understanding model behavior
- **Use Comparison** to see the difference visually

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## PDF Input Modes

- **File path**: Point to a PDF on disk (default: `data/document.pdf`)
- **Upload PDF**: Upload a file directly in the app

Default path is `data/document.pdf`.

## Gemini Configuration

API key lookup priority:

1. Environment variable: `GEMINI_API_KEY`
2. Streamlit secrets: `GEMINI_API_KEY`
3. Temporary sidebar input (session-only, secure)

Security rules:

- Keys are not written to disk
- Keys are not logged
- Manual key is session-only (expires when you close browser)

Provider status shown in UI:

- `working`
- `invalid key`
- `quota issue`
- `network error`
- `fallback mode active`

## No API Key Behavior

The app remains usable without an API key:

- Retrieval-only remains fully available
- LLM-only mode is disabled
- UI shows fallback mode status
- Perfect for exploring the RAG pipeline without LLM generation

## Troubleshooting

### "First query is slow!"

**Normal.** First time indexing a PDF takes 20-45 seconds because the app is:
1. Extracting text from all pages
2. Chunking into ~1300 char pieces
3. Computing embeddings for all chunks (~400-1000 embeddings)
4. Building FAISS semantic index
5. Saving to `.cache/` for reuse

**Solution**: Wait for progress bar to complete. Indexed PDFs are fast on reuse.

### "LLM-only mode is disabled"

**Cause**: No valid Gemini API key configured.

**Solution**:
- Set `GEMINI_API_KEY` environment variable
- Or add to Streamlit secrets
- Or enter in sidebar "Temporary API key" field    **Note**: LLM-only mode is optional. RAG retrieval-only mode works without any API key!

### "No retrieved chunks found"

**Cause**: Query didn't match document content well, or document index is corrupted.

**Solution**:
- Try rephrasing your question (use different keywords)
- Ask about sections you know exist in the document
- Check that Retrieved Chunks show up with reasonable scores
- Consider uploading a different document

### ".cache/ folder is growing large"

**Normal.** Cache grows with more PDFs:
- `chunks_*.json` → extracted text
- `metadata_*.json` → chunk metadata
- `faiss_*.index` → semantic search index
- `embeddings_*.npy` → embedding vectors

**Safe to delete?** Yes! The `.cache/` folder is safe to delete. Next run will recompute everything. No data loss.

**Size estimate**: ~50-100 MB per typical 100-page PDF.

### "Same document in new session—will it re-index?"

**No.** If you close and reopen the app with the same document:
- Source ID (SHA256 hash) is identical
- `.cache/` files already exist
- App loads chunks from JSON (~200 ms)
- Embeddings loaded from disk (~1 s)
- No re-embedding needed

**Instant warm start**: ~5 seconds vs. 30-45 seconds cold start.

### "Can I delete `.cache/`?"

**Yes, safely.** `.cache/` is entirely computer-generated. Delete anytime:
- Re-indexing will happen on next run
- No permanent data loss
- Useful if cache gets corrupted or you run low on disk space

## Project Structure

- `app.py` — Main Streamlit app
- `src/doc_ingestion.py` — PDF parsing, chunking, caching
- `src/hybrid_retriever.py` — BM25 + semantic retrieval + reranking
- `src/response_generator.py` — Gemini integration, anti-hallucination prompts
- `src/ui_styles.py` — Streamlit styling
- `data/` — Where to put your PDF
- `.cache/` — Persistent index storage (auto-created)

## The RAG Pipeline Explained

1. **Document Ingestion**: Extract text from PDF pages
2. **Chunking**: Split into ~1300 character chunks with ~180 char overlap (preserves context)
3. **Embedding**: Convert each chunk to 768-dim vector using `all-mpnet-base-v2`
4. **Indexing**: Build FAISS index for fast semantic search
5. **Retrieval**: On query, retrieve top 8 semantic + top 8 BM25 hits, fuse with RRF
6. **Reranking**: Rerank top candidates using `ms-marco-MiniLM-L-6-v2` cross-encoder
7. **Generation**: Pass top 5 chunks + query to Gemini with strict grounding prompt

## Hallucinations And Why RAG Helps

**Pure LLM mode** may answer with no document grounding:
- Model uses training data (which may be outdated, wrong, or confidently wrong)
- No way to verify answer against actual document
- Useful for general knowledge, bad for domain-specific facts

**RAG mode** retrieves evidence first:
- Question → retrieve matching chunks → pass to LLM with context
- LLM grounds answer in actual document content
- Reduces hallucinations, improves traceability
- Shows retrieved evidence so user can verify

When evidence is weak or missing, RAG mode returns:

```
I cannot find this information in the document.
```

## Caching

The `.cache/` folder stores:

- `chunks_{source_id}.json` — extracted and chunked text
- `metadata_{source_id}.json` — chunk metadata (page, section, token count)
- `faiss_{source_id}.index` — semantic search index
- `embeddings_{source_id}.npy` — embedding vectors
- `uploads/{source_id}.pdf` — uploaded PDFs (if uploading)

All text cache files use UTF-8 encoding for Windows compatibility.

Source ID is SHA256 hash of PDF bytes (first 16 chars), ensuring identical PDFs reuse cache.

## Limitations

- Diagram-heavy pages may be partially represented as text only (diagrams skipped)
- Scanned PDFs need OCR for best results
- Very large PDFs (500+ pages) may take 2+ minutes for first indexing
- Sensitive PDFs: make sure you trust the Gemini API with your data

## Requirements

See `requirements.txt` for details. Main dependencies:

- `streamlit` — Web UI framework
- `pdfplumber` — PDF text extraction
- `pypdf` — PDF fallback extraction
- `sentence-transformers` — Embedding model
- `faiss-cpu` — Semantic search index
- `rank-bm25` — Lexical search
- `google-generativeai` — Gemini API

