# KUST AI Assistant

> A Retrieval-Augmented Generation (RAG) AI assistant for **Kohat University of Science & Technology (KUST)**, built with FastAPI, Pinecone, OpenAI GPT-4o-mini, and Next.js 14.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI (Python 3.11) |
| Vector DB | Pinecone |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | text-embedding-3-small (1536-dim) |
| Voice | OpenAI Whisper |
| PDF Parsing | PyMuPDF (fitz) |
| Auth | JWT via python-jose + passlib/bcrypt |
| Frontend | Next.js 14 (App Router) + Tailwind CSS |
| Containers | Docker + Docker Compose |

---

## Project Structure

```
kust-rag/
├── backend/
│   ├── ingest/         # PDF loading, chunking, embedding
│   ├── retrieval/      # Pinecone search, query rewriting
│   ├── auth/           # JWT creation, FastAPI dependencies
│   ├── routes/         # chat, admin, transcribe endpoints
│   ├── core/           # settings (pydantic), language utils
│   ├── main.py         # FastAPI app factory
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app/            # Next.js App Router pages
│   ├── components/     # ChatInput, MessageBubble, CitationBadge
│   ├── hooks/          # useChat, useVoice
│   ├── next.config.js
│   └── Dockerfile
├── pdfs/               # ← drop KUST PDF documents here
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Quick Start

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in OPENAI_API_KEY, JWT_SECRET
```

### 2. Add PDFs

Drop KUST PDF documents into the `pdfs/` directory.

### 3. Start all services

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

### 4. Ingest PDFs

After all services are healthy, trigger ingestion from the Admin dashboard at `http://localhost:3000/admin` or via the API:

```bash
curl -X POST http://localhost:8000/admin/ingest \
  -H "Authorization: Bearer <your-jwt-token>"
```

---

## Development

### Backend (without Docker)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

### Frontend (without Docker)

```bash
cd frontend
npm install
npm run dev
```

---

## Roadmap

- [x] Project scaffolding
- [ ] Day 1 — `pdf_loader.py`: PyMuPDF-based PDF parsing
- [ ] Day 2 — `chunker.py` + `embedder.py`: chunking & Pinecone upsert
- [ ] Day 3 — `retriever.py` + `chat.py`: RAG query pipeline
- [ ] Day 4 — Auth, admin panel, voice transcription
- [ ] Day 5 — Frontend polish, streaming UI, citations

---

## Environment Variables

See `.env.example` for all required and optional variables.

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key |
| `PINECONE_API_KEY` | Pinecone API key |
| `PINECONE_INDEX` | Pinecone index name |
| `JWT_SECRET` | HMAC secret for JWT signing |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Token lifetime (default 60) |
| `DEBUG` | Enable debug logging |

---

## License

MIT — feel free to adapt for your institution.
