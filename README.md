# home_cc
Home central command.

_Books folder

- remove unneccesary:
    - symbols
    - characters
    - uniform format
    - proper case
    - add or remove spaces
    
     book information
    
    - indexable
    - tagged
    - metadata
    - get author and summary from web?

- Features:
    - [ ]  convert books to audio
    - [ ]  file converter
    - [ ]  integrate ArchiveMind?

- UI/UX
    - searches
    - summaries
    - chatbot
    - AI\ML learning
    - Graphs
    - Charts
    - Dashboard
    - Knowledge graphs
    
    compress originals 
    
    edit with proxies?
    
    edits applied then re-compress
    
    algortihm for ‘fav’ files ( maybe compression is not applied, then after a time period of statisical decline in usage the file gets compressed?)
    
    seperate dashboads and passwords for users
    
    Admin Dashboard
    
    Community Dashboard
    
    genealogy music documents books pictures videos storage sharing
    
- Algorithms
- database
    - naming conventions

Python

Data cleaning

App for learning DATA engineering, databasing, dashboards, home automations, media and entertainment, weather reports (from IoT devices, devs, plus api weather data), weather, calendar, TODO’s, appointments, outlooks, document and file server, cloud storage, containers?, user prefs, user privacy, personal assistant, nutrition system, health metrics , IoT, finance reports, investment calculators and simulations, calculators, open in another app (open photo in photoshop), research assistant, web-scrape, dd

- **Open Questions:**
    - Rpi as a server and host?
    - NAS
    - best performance
    - localizations
    - access through web
    - future users access how?
    - admin has main app on MacBook
    - React? - Streamlit?

## PostgreSQL + pgvector RAG Platform

### 1. Start PostgreSQL + pgvector Container
```bash
docker compose up -d
```

### 2. Initialize Database Schema & Vector Extensions
```bash
python pg_vector_db.py --init
```

### 3. Ingest Catalog Records & Document Chunks
```bash
# Full ingestion using local FastEmbed (BAAI/bge-small-en-v1.5)
python ingest_vector_db.py

# Ingest catalog metadata only (without physical document chunking)
python ingest_vector_db.py --catalog-only

# Ingest using Ollama or OpenAI embeddings
python ingest_vector_db.py --provider ollama --model nomic-embed-text
python ingest_vector_db.py --provider openai --model text-embedding-3-small
```

### 4. Hybrid Search & RAG Retrieval
```bash
# Hybrid Search (Dense HNSW + Sparse tsvector fused with Reciprocal Rank Fusion)
python rag_search.py --query "How do convolutional neural networks work?" --mode hybrid --top-k 5

# Dense Vector Similarity Search
python rag_search.py --query "Bipolar transistors amplifier circuits" --mode vector --top-k 5

# Keyword tsvector Full-Text Search
python rag_search.py --query "Raspberry Pi GPIO sensors" --mode keyword

# Hierarchical Macro-to-Micro Book Search
python rag_search.py --query "Deep learning transformers" --mode hierarchical

# Generate Citations and LLM Context Block
python rag_search.py --query "Transformer self-attention mechanism" --context

# Synthesize Complete Prompt for LLM Generation
python rag_search.py --query "Explain backpropagation in deep neural networks" --synthesize
```
