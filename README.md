# 📚 DS Notes RAG Chatbot

A production-style **Retrieval Augmented Generation (RAG)** chatbot that answers questions based on Data Science lecture notes. Built with LangChain, ChromaDB, Groq (Llama 3.3 70B), and served via FastAPI — fully containerized with Docker.

---

## 🚀 What This Project Does

Instead of relying on general AI knowledge, this chatbot:

1. **Reads** your Data Science notes (`.docx` file)
2. **Chunks** the content into smaller pieces
3. **Embeds** each chunk into vectors using `all-MiniLM-L6-v2`
4. **Stores** them in a ChromaDB vector database
5. **Retrieves** the most relevant chunks when you ask a question
6. **Generates** an answer using Llama 3.3 70B via Groq — based on YOUR notes

```
User Question
      ↓
RouterAgent (RAG or General?)
      ↓
ChromaDB (find top-4 relevant chunks)
      ↓
Prompt Builder (question + context)
      ↓
Llama 3.3 70B via Groq
      ↓
Answer based on your notes ✅
```

---

## 🗂️ Project Structure

```
ds_rag_chatbot/
│
├── api/                        # FastAPI layer
│   ├── main.py                 # App startup, CORS, lifespan
│   └── routes/                 # API route handlers
│       └── chat.py             # /chat endpoint
│
├── ingestion/                  # Data processing pipeline
│   ├── document_loader.py      # Reads .docx file
│   ├── chunker.py              # Splits text into chunks
│   └── embedder.py             # Converts chunks to vectors
│
├── retrieval/                  # Vector search layer
│   └── vectorstore.py          # ChromaDB load, store, retrieve
│
├── generation/                 # LLM answer generation
│   ├── llm_client.py           # Groq LLM + RAG chain
│   └── prompt_builder.py       # Prompt template builder
│
├── agents/                     # Decision making
│   └── router_agent.py         # Routes to RAG or general LLM
│
├── tests/                      # All test files
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_generation.py
│
├── chroma_db/                  # Persisted vector store (auto-generated)
├── data/                       # Your raw .docx notes go here
├── chat.html                   # Simple browser-based chat UI
├── ingest.py                   # Run this once to build the vector store
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container build instructions
├── docker-compose.yml          # Container orchestration
├── .env                        # Environment variables (never commit!)
└── README.md                   # You are here
```

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| LLM | Llama 3.3 70B via Groq | Free tier, extremely fast |
| Embeddings | all-MiniLM-L6-v2 | Lightweight, accurate, free |
| Vector DB | ChromaDB | Simple, local, no setup needed |
| RAG Framework | LangChain | Industry standard for RAG pipelines |
| API Framework | FastAPI | Async, auto docs, fast |
| Containerization | Docker + Docker Compose | Consistent across all environments |
| Document Parsing | docx2txt | Reads .docx notes |

---

## ⚙️ Setup & Running Locally

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed
- A free [Groq API Key](https://console.groq.com/)

### Step 1 — Clone the Repository
```bash
git clone https://github.com/your-username/ds_rag_chatbot.git
cd ds_rag_chatbot
```

### Step 2 — Add Your Notes
Place your Data Science notes file inside the `data/` folder:
```
data/
└── your_notes.docx
```

### Step 3 — Create the `.env` File
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```
> ⚠️ Never commit this file to GitHub. It is already in `.gitignore`.

### Step 4 — Build the Vector Store (First Time Only)
```bash
docker-compose run --rm ds-rag-api python ingest.py
```
This reads your notes, creates embeddings, and saves them to `chroma_db/`.

### Step 5 — Start the Application
```bash
docker-compose up
```

The API will be live at:
```
http://localhost:8000
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/chat` | Send a question, get an answer |
| `GET` | `/health` | Check if the service is running |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/redoc` | ReDoc API documentation |

### Example Request
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is a pandas DataFrame?"}'
```

### Example Response
```json
{
  "answer": "A DataFrame is a 2-dimensional labeled data structure in pandas...",
  "source": "rag",
  "chunks_used": 4
}
```

---

## 💬 Using the Chat UI

Open `chat.html` directly in your browser:
```
Just double-click chat.html
```
Or visit `http://localhost:8000/docs` for the full Swagger interface.

---

## 🧪 Running Tests

```bash
# Run all tests
docker-compose run --rm ds-rag-api pytest tests/ -v

# Run specific test file
docker-compose run --rm ds-rag-api pytest tests/test_ingestion.py -v
```

---

## 🔄 How RAG Works (Simple Explanation)

```
Step 1 — INGESTION (done once)
  Your Notes (.docx)
        ↓
  Split into chunks (~500 words each)
        ↓
  Each chunk → converted to vector (384 numbers)
        ↓
  All vectors stored in ChromaDB
  (108 vectors currently in index)

Step 2 — RETRIEVAL + GENERATION (every query)
  User asks: "Explain pandas merge"
        ↓
  Question → converted to vector
        ↓
  ChromaDB finds top 4 most similar chunks
        ↓
  Chunks + Question → sent to Llama 3.3 70B
        ↓
  Answer grounded in YOUR notes
```

---

## 🏗️ Architecture Diagram

```
                    ┌─────────────┐
                    │  chat.html  │
                    │  (Browser)  │
                    └──────┬──────┘
                           │ HTTP POST /chat
                    ┌──────▼──────┐
                    │   FastAPI   │
                    │  (main.py)  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │RouterAgent  │
                    │(RAG/General)│
                    └──────┬──────┘
               ┌───────────┴───────────┐
               │                       │
        ┌──────▼──────┐         ┌──────▼──────┐
        │  ChromaDB   │         │  Groq API   │
        │(Vector Search│        │(Llama 3.3)  │
        └──────┬──────┘         └──────┬──────┘
               │                       │
               └───────────┬───────────┘
                    ┌──────▼──────┐
                    │   Answer    │
                    └─────────────┘
```

---

## 📦 Docker Details

```bash
# Build image
docker-compose build

# Start container
docker-compose up

# Start in background
docker-compose up -d

# Stop container
docker-compose down

# View logs
docker-compose logs -f

# Rebuild from scratch
docker-compose up --build
```

The Docker image (~8.82 GB) is large due to PyTorch and CUDA libraries required by `sentence-transformers`.

---

## 🔮 Future Improvements

- [ ] Add user authentication (JWT)
- [ ] Use CPU-only PyTorch to reduce image size to ~2 GB
- [ ] Add conversation memory (multi-turn chat)
- [ ] Deploy to Railway / Render / AWS
- [ ] Add CI/CD pipeline with GitHub Actions
- [ ] Replace local ChromaDB with Pinecone (cloud vector DB)
- [ ] Add source citations in answers
- [ ] Add support for PDF notes

---

## 📁 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | Your Groq API key from console.groq.com |
| `HF_TOKEN` | ❌ Optional | HuggingFace token (removes rate limit warnings) |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Aniket**
- Built as a portfolio project demonstrating RAG, LangChain, FastAPI, and Docker
- Covers real-world MLOps patterns used in production AI systems

---

> 💡 **Interview Tip:** Be ready to explain every layer — why RAG over fine-tuning, why ChromaDB, why FastAPI over Flask, why Docker. Each decision was intentional.
