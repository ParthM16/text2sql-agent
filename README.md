# 🤖 Text-to-SQL AI Agent: Talk to Your Database in Plain English

## 🎯 What We Built
In most businesses, data is locked behind a wall called **SQL**. When business users (marketing, sales, executives) have a question, they have to file a ticket with the data team and wait-sometimes for hours or days-just to get a simple answer.

This project tears down that wall. 

I built an **AI Agent** that allows anyone to ask questions about their business in plain English. The agent understands the intent, writes the database query, executes it, and returns a formatted human-readable answer with charts-in under few seconds.

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
- **Language Model**: Google Gemini 2.5 Flash (via Vertex AI / GenAI SDK)
- **RAG Memory**: FAISS (Facebook AI Similarity Search) & HuggingFace Embeddings
- **Backend/Logic**: Pure Python, SQLAlchemy, synchronous pipeline
- **Frontend**: Streamlit + Altair for interactive charting
- **Security**: 10-layer pre-LLM input guardrail (regex, langdetect, difflib)

## 📐 Architecture
The agent processes each query through a **12-node sequential pipeline**:

```
User Question → Input Guardrail → Intent Detection → Table Selection → Shadow Discovery (UUI)
→ Few-Shot Retrieval → Attribute Matching → SQL Generation → SQL Execution
→ Auto-Correction Loop (3x) → Natural Language Answer → Chart Builder
```

## 🚀 How to Run Locally
1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `.\venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and add your Google AI credentials
6. Run the app: `streamlit run main.py`

## 📁 Project Structure
```
text2sql_agent/
├── main.py                  # Entry point (st.navigation)
├── app.py                   # Chat Agent UI
├── pages/
│   └── 1_📊_Dashboard.py   # Prompt Analytics Dashboard
├── src/
│   ├── agent/
│   │   ├── workflow.py      # 12-node orchestrator
│   │   ├── nodes.py         # LLM prompt templates
│   │   └── exceptions.py    # Custom error hierarchy
│   ├── db/
│   │   └── helper.py        # SQL Server + SQLite abstraction
│   ├── retrieval/
│   │   └── vector_store.py  # FAISS + embeddings
│   └── utils/
│       ├── logger.py        # Daily rotating file logs
│       ├── metrics.py       # Session + persistent metrics
│       ├── web_service.py   # Live unit conversion
│       └── disk_cache.py    # CSV cache GC
├── scripts/
│   ├── run_evals.py         # Offline evaluation framework
│   └── download_models.py   # Pre-download ML weights
├── data/
│   ├── few_shot_examples.json  # Seed examples for RAG
│   └── eval_set.json           # Gold-standard test set
├── .streamlit/config.toml   # Streamlit server config
├── .env.example             # Safe credential template
├── requirements.txt
├── CHANGELOG.md
└── README.md
```

---
*Built as a showcase of bridging the gap between cutting-edge LLMs and practical Data Engineering.*
