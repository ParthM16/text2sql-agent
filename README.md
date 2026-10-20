# 🤖 Text-to-SQL AI Agent: Talk to Your Database in Plain English

<table>
<tr>
<td width="70%">

## 📑 Table of Contents

* [🎯 What We Built](#-what-we-built)
* [👥 Who Is This For?](#-who-is-this-for)
* [⏱️ Whose Time Does It Save?](#%EF%B8%8F-whose-time-does-it-save)
* [🏆 What We Achieved](#-what-we-achieved)
* [🚧 The Challenges & How We Overcame Them](#-the-challenges--how-we-overcame-them)
* [🛠️ The Tech Stack (Under the Hood)](#%EF%B8%8F-the-tech-stack-under-the-hood)
* [📐 Architecture](#-architecture)
* [🚀 How to Run Locally](#-how-to-run-locally)
* [📁 Project Structure](#-project-structure)

## 🎯 What We Built
In most businesses, data is locked behind a wall called **SQL**. When business users (marketing, sales, executives) have a question, they have to file a ticket with the data team and wait-sometimes for hours or days-just to get a simple answer.

This project tears down that wall. 

I built an **AI Agent** that allows anyone to ask questions about their business in plain English. The agent understands the intent, writes the database query, executes it, and returns a formatted human-readable answer with charts-in under few seconds.

</td>
<td width="30%" align="center">

<!-- 
INSTRUCTIONS TO ADD YOUR VERTICAL VIDEO:
1. Open this README.md on GitHub.com and click the "Edit" (pencil) icon.
2. Drag and drop your .mp4 video file right here into this space. 
3. GitHub will automatically upload it and generate a video link that looks like this:
   https://github.com/user-attachments/assets/xxxx-xxxx-xxxx
4. It will render perfectly in a vertical format on the right side!
-->

**Demo Video**<br>


https://github.com/user-attachments/assets/f4b3aecb-36d7-445f-a58f-7d0cd8131dca

</td>
</tr>
</table>

## 👥 Who Is This For?

This app is built for **non-technical business professionals** who need data answers but don't know SQL:

| User | Example Question They'd Ask |
|---|---|
| 📊 **Business Analysts** | *"What were the top 5 products by revenue last quarter?"* |
| 💼 **Sales Managers** | *"Compare this month's sales to last month by region"* |
| 👔 **Executives / C-Suite** | *"Show me the overall profit trend for 2024"* |
| 📣 **Marketing Teams** | *"Which customer segment has the highest repeat purchase rate?"* |
| 🏭 **Operations Leads** | *"What's the average order fulfillment time by warehouse?"* |

## ⏱️ Whose Time Does It Save?

| Without This Agent | With This Agent |
|---|---|
| Business user writes a ticket → waits for data team → data engineer writes SQL → debugs → sends results back (**~30-60 min**) | Business user types a question → gets an answer with charts (**~5-15 sec**) |
| Data engineers spend **40%** of their day answering ad-hoc report requests | Data engineers focus on **real engineering** - pipelines, architecture, optimization |
| Executives wait until the **next weekly report** to get a number | Executives get **instant, self-service** access to any metric, anytime |

## 🏆 What We Achieved
* **Time Saved**: Reduced the time it takes to get a data answer from **~45 minutes** (manual SQL writing and debugging) to **few seconds**.
* **Zero Technical Barrier**: Users don't need to know what a database is, let alone how to write code.
* **Self-Improving Memory**: The agent actually learns from its mistakes, storing successful corrections in its permanent memory so it never makes the same mistake twice.

---

## 🚧 The Challenges & How We Overcame Them
Building an AI demo is easy. Building a reliable product is hard. Here are the real-world challenges this project solved:

### 1. The "Hallucination" Problem (Making Things Up)
* **Challenge**: AI models are people-pleasers. If you ask for a column that doesn't exist, they will confidently invent it, causing the database to crash.
* **Solution**: Implemented an **"Echo Test" and Schema Anchor**. Before generating code, the agent filters the database structure and ensures it only ever uses tables and columns that actually exist.

### 2. The Currency Collision Problem
* **Challenge**: If a database contains sales in USD, EUR, and INR, a basic AI will blindly add them together (e.g., $100 + 1000 INR = 1100 Nonsense).
* **Solution**: Built a **Universal Unit Intelligence (UUI)** layer. The agent scans the database before answering, detects conflicting currencies, fetches live exchange rates from the web, and converts everything into a single currency mathematically *before* giving you the answer.

### 3. The "AI Forgets" Problem
* **Challenge**: If an AI makes a mistake, you can tell it to fix it. But tomorrow, it will make the exact same mistake again.
* **Solution**: Built an **Automated Learning Loop**. When the AI successfully fixes a broken query, a background thread saves that specific fix into a local memory bank (Vector Database). The next time a similar question is asked, it pulls from its past experience. 

### 4. Database Dialect Wars
* **Challenge**: SQL isn't universal. A command that works in SQL Server might fail completely in SQLite.
* **Solution**: Implemented **Dialect-Aware Prompts**. The agent detects what kind of database you connected to and adjusts its grammar automatically.

### 5. Security & Prompt Injection
* **Challenge**: Malicious users could trick the AI into executing destructive SQL (DROP TABLE) or bypass system instructions.
* **Solution**: Built a **10-layer pre-LLM security guardrail** - including spell correction for evasion typos, machine-language encoding detection (binary/hex/base64), natural language validation, and regex-based DML/DDL blocking - all running *before* any AI model is called.

---

## 🛠️ The Tech Stack (Under the Hood)
For the technical folks, here is what powers the engine:
- **Centralized Configuration (`src/config.py`)**: Single Source of Truth (SSOT) managing all model IDs, API keys, vector store paths, pricing tables, and thresholds.
- **Multi-Model LLM Router (`src/llm/model_router.py`)**:
  - **Cloud AI**: Google Gemini (`gemini-3.8-flash` / `gemini-3.5-flash-lite`) via Google GenAI SDK.
  - **Local AI**: Quantized local LLMs via Ollama (`ollama/qwen2.5-coder:3b` / `llama3.2`).
  - **Resilience**: Bidirectional automatic fallback and live token/cost tracking.
- **Dual Vector Store Backend (`src/retrieval/store_factory.py`)**:
  - **In-Memory (Fast L2)**: FAISS `IndexFlatL2` for sub-millisecond retrieval.
  - **Persistent Local Disk**: ChromaDB persistent collections with HNSW indexing.
  - **RAM Optimization**: Shared `SentenceTransformer('all-MiniLM-L6-v2')` singleton preventing duplicate memory allocation.
  - **Dual-Write Sync**: Automated learning loops synchronize corrections across active stores.
- **Enterprise Schema RAG (3-Tier Funnel)**:
  - **Tier 1 (Table Discovery)**: Lean table catalog + vector table routing (scales to 50+ tables).
  - **Tier 2 (Schema Linking)**: Semantic column pruning isolating top 10–18 columns + PK/FK/Date anchors (scales to 1,000+ columns).
  - **Tier 3 (SQL Generation)**: Relation-aware prompts enforcing known primary/foreign key JOIN conditions.
- **Evaluation & Benchmarking Suite (`src/evaluation/benchmark.py`)**:
  - Automated benchmark measuring syntax validity, execution accuracy, self-healing rates, per-node latency, and cost trade-offs.
  - Built-in rate-limit guard (15 RPM compliance) and automatic markdown/CSV export (`BENCHMARK_REPORT.md`).
- **Frontend & UI (`app.py` & Streamlit)**:
  - Runtime dropdown selectors for LLM Provider (Gemini vs. Ollama) and Vector Store (FAISS vs. ChromaDB).
  - Live telemetry card showing per-node latency breakdown, token usage, and live multi-model cost comparisons ($0.00 Ollama vs. Gemini vs. GPT-4o-mini vs. Claude 3.5 Haiku).
- **Security**: 10-layer pre-LLM input guardrail (regex, langdetect, difflib, machine-encoded sequence blocks).

## 📐 Architecture
The agent processes each query through an enterprise-grade agentic pipeline:

```
User Question
      │
      ▼
Input Guardrails (Injection & Encoding Blocks)
      │
      ▼
Intent Detection (NEW_QUERY / CHART_REQUEST / FOLLOW_UP / EXPLORATORY)
      │
      ▼
[Tier 1] Table Catalog & Semantic Routing (Filters 50+ tables → 2-4 candidate tables)
      │
      ▼
[Tier 2] Schema Linking & Column Pruning (Filters 1,000+ cols → 10-18 high-signal cols + PK/FK)
      │
      ▼
Universal Unit Intelligence (UUI) (Live Currency & Unit Normalization)
      │
      ▼
Dual Vector Retrieval (Few-Shot Examples + Categorical Typo Correction)
      │
      ▼
Relation-Aware SQL Generation (Gemini Cloud ↔ Local Ollama with Fallback)
      │
      ▼
Database Execution (SQL Server / SQLite)
      │
      ├── (If Error) ──► Self-Healing SQL Correction Loop (3x Retry) ──► Automated Learning Store
      │
      ▼
Natural Language Answer & Interactive Visualization
```

## 🚀 How to Run Locally
1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `.\venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and configure your settings:
   - `GEMINI_API_KEY` (Free tier from Google AI Studio)
   - `OLLAMA_MODEL="qwen2.5-coder:3b"` (Optional: for local zero-cost offline execution)
   - `VECTOR_BACKEND="faiss"` or `"chroma"`
6. Run the app: `streamlit run app.py`
7. Run the benchmark suite: `python -m src.evaluation.benchmark`

## 📁 Project Structure
```
text2sql_agent/
├── main.py                     # Entry point (st.navigation)
├── app.py                      # Main Chat Agent UI with Runtime Model/Vector Selectors
├── pages/
│   └── 1_📊_Dashboard.py      # Telemetry & Prompt Analytics Dashboard
├── src/
│   ├── config.py               # Central Single Source of Truth (SSOT) configuration
│   ├── agent/
│   │   ├── workflow.py         # Multi-node orchestrator with 3-Tier Enterprise Funnel
│   │   ├── nodes.py            # Prompt engineering templates & node logic
│   │   └── exceptions.py       # Custom error hierarchy
│   ├── db/
│   │   └── helper.py           # SQL Server + SQLite abstraction & deep schema introspection
│   ├── llm/
│   │   ├── model_router.py     # Gemini & Ollama multi-model router with fallback
│   │   └── ollama_client.py    # Local Ollama client adapter
│   ├── retrieval/
│   │   ├── store_factory.py    # Unified factory for FAISS & ChromaDB
│   │   ├── vector_store.py     # In-memory FAISS L2 vector store & schema index
│   │   ├── chroma_store.py     # Persistent ChromaDB collections & schema index
│   │   └── embeddings.py       # Shared SentenceTransformer singleton
│   ├── evaluation/
│   │   └── benchmark.py        # Automated benchmarking suite with rate-limit guards
│   └── utils/
│       ├── logger.py           # Daily rotating structured file logs
│       ├── metrics.py          # Session + persistent telemetry
│       ├── web_service.py      # Live exchange rate & unit conversion service
│       └── disk_cache.py       # Cache management
├── data/
│   ├── few_shot_examples.json  # Seed examples for few-shot RAG
│   ├── schema_knowledge.json   # Persistent enriched schema catalog
│   └── eval_set.json           # Gold-standard benchmark evaluation set
├── .streamlit/config.toml      # Streamlit server config
├── .env.example                # Safe credential template
├── requirements.txt
├── CHANGELOG.md
└── README.md
```

---
*Built as a showcase of bridging the gap between cutting-edge LLMs and practical Data Engineering.*
