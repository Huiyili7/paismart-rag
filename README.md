# PaiSmart — Enterprise RAG Knowledge Base

An enterprise-grade **Retrieval-Augmented Generation (RAG)** knowledge base. Users upload documents, ask questions in natural language, and get answers **grounded in their own corpus with inline citations**. Designed for multi-tenant use with organization-level data isolation.

Beyond the application itself, this repo ships a **RAG evaluation & quality harness** (`eval/`) and a CI pipeline that continuously measure retrieval quality (Recall / MRR / nDCG), latency, and hallucination rate — the system is treated as something to be *measured*, not just built.

---

## Highlights

- **Hybrid retrieval** — vector KNN recall (window `topK×30`) + BM25 rescoring (`operator=AND`), with graceful degradation to text-only search when embedding is unavailable.
- **Semantic chunking** — streaming parent/child chunking that avoids OOM on large files; child splitting cascades paragraph → sentence (zh/en punctuation) → HanLP word segmentation → character fallback.
- **Grounded generation / anti-hallucination** — system prompt enforces citations, refuses ("暂无相关信息") when no supporting context is retrieved, and includes a prompt-injection guard; low temperature (0.3).
- **Function-calling query router** — an LLM tool-call decides whether a query needs knowledge-base retrieval or is general chat, avoiding needless retrieval and grounding noise.
- **Multi-tenant isolation** — `userId` / `public` / `orgTag` filters are pushed *into* the Elasticsearch query, so permission is enforced at retrieval time.
- **Real-time UX** — WebSocket + SSE streaming responses; conversation memory in Redis (last 20 turns, 7-day TTL).
- **Quality engineering** — a Python evaluation harness with regression gates wired into GitHub Actions CI (shift-left).

---

## Architecture

```
Upload (chunked/resumable) → MinIO → Kafka → async parse (Tika)
   → semantic chunking → embedding (DashScope text-embedding-v4, 2048-d)
   → indexed into Elasticsearch (vector + BM25)

Query → Function-calling router → Hybrid search (KNN + BM25, permission-filtered)
   → context assembly (numbered, cited) → LLM (DeepSeek, streamed) → client
```

The backend follows a clean layered design (controller → service → repository → entity) with JWT-secured REST APIs and transactional, multi-source service logic.

### Backend (`src/main/java/com/yizhaoqi/smartpai/`)
```
client/      external API clients (DeepSeek LLM, embedding)
config/      security, JWT, Elasticsearch, Kafka, Redis, WebSocket config
consumer/    Kafka consumer for async file processing
controller/  REST endpoints
service/     business logic (search, chunking, routing, chat, conversation)
model/ entity/ repository/   domain + persistence
handler/     WebSocket chat handler
```

### Frontend (`frontend/`)
Vue 3 + TypeScript, Vite, Naive UI, Pinia, Vue Router.

### Evaluation (`eval/`)
See [`eval/README.md`](eval/README.md). Computes Recall@k, MRR, nDCG@k, latency p50/p95/p99, hallucination rate and refusal accuracy over a gold set; enforces regression gates that fail CI on quality regressions.

---

## Tech Stack

**Backend:** Spring Boot 3.4 (Java 17) · Elasticsearch 8.10 · Apache Kafka · Redis · MySQL 8 · MinIO · Apache Tika · HanLP · Spring Security + JWT · WebFlux / WebSocket
**Frontend:** Vue 3 · TypeScript · Vite · Naive UI · Pinia
**AI:** DeepSeek (chat) · DashScope `text-embedding-v4` (embeddings)
**Quality/CI:** Python evaluation harness · GitHub Actions

---

## Getting Started

### Prerequisites
Java 17, Maven 3.8+, Node 18+ / pnpm (frontend, optional), Python 3.11+ (eval), Docker (for backing services).

### 1. Start backing services
```bash
cd docs && docker compose up -d   # MySQL, Redis, Elasticsearch, Kafka, MinIO
```

### 2. Configure API keys (read from environment — never commit secrets)
```bash
export DEEPSEEK_API_KEY=sk-...
export DASHSCOPE_API_KEY=sk-...
```

### 3. Run the backend
```bash
mvn spring-boot:run        # serves on :8081
```

### 4. Run the frontend (optional)
```bash
cd frontend && pnpm install && pnpm dev
```

### 5. Evaluate retrieval & answer quality
```bash
cd eval && pip install -r requirements.txt
python scripts/seed_corpus.py                 # seed the sample education corpus
python src/evaluate.py --mode live --generate --judge
```

---

## Testing & CI

- **Unit tests:** `mvn test` (Mockito / plain JUnit; integration tests tagged separately as they require live ES/MySQL/Kafka/Redis).
- **RAG evaluation:** metric unit tests + an offline, deterministic retrieval/quality evaluation with regression gates.
- **CI** (`.github/workflows/ci.yml`): compiles the project, runs infra-free unit tests, and runs the RAG evaluation gates on every push/PR.

---

## License

See [LICENSE](LICENSE).
