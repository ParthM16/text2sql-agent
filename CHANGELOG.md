# CHANGELOG: Text-to-SQL Agent Evolution

All notable changes to the Retail Insights Agent.

## [April 29, 2026] - Synchronous Execution & Pipeline Cleanup

### Changed
- **Synchronous "Block + Spinner" Model**: Completely removed background threading, polling fragments, and interrupt logic from `app.py`. The agent now runs as a clean, sequential pipeline under `st.spinner()`, blocking UI input until the query completes. This eliminates all concurrency overhead and maximizes execution speed.
- **Stop Check Removal**: Stripped all `is_stopped()`, `print_stop_check()`, and `stop_event` guard logic from every node in `workflow.py`. The agent pipeline no longer checks for cancellation signals between nodes, resulting in cleaner terminal logs and faster throughput.
- **Stable Session State**: Moved `st.session_state.messages` initialization to the top of `app.py` to prevent `KeyError` crashes during page navigation (e.g., switching between Chat Agent and Dashboard).
- **Entry Point Migration**: The app now uses `main.py` → `st.navigation()` as the single entry point (not `app.py` directly), enabling multi-page navigation with shared `st.set_page_config`.

### Removed
- `is_stopped()` helper function from `workflow.py`
- `print_stop_check()` helper function from `workflow.py`
- All `stop_event` conditional returns (`"⏹️ Query stopped by user."`) from every pipeline node
- Background thread polling and `st.fragment`-based status updater from `app.py`

---

## [April 28, 2026] - Iron-Clad Metadata Shield & Documentation

### Added
- **Hard Metadata Bypass (Negative-First)**: UUI now checks for system tables (`sys.`, `information_schema`, `count(*)`) BEFORE checking keywords. Metadata queries skip UUI entirely.
- **Regex Word-Boundary Protection**: Currency exception check uses `\bINR\b` regex to prevent SQL keyword `IN` from being mistaken as currency code `INR`.
- **Node-Level Fast Bypass**: `unit_discovery()` in `nodes.py` now pre-screens results for metadata labels and returns instantly without LLM call.
- **UUI Bypass Logging**: Agent now logs `[Node: UUI Bypass] System/Metadata query detected` in Thought Process for transparency.
- **DIAGRAM_PROMPTS.md — Prompt 7**: New dedicated prompt for UUI Integrity Layer visualization.
- **FUNCTION_REFERENCE.md**: Complete inventory of all 40+ functions across 10 source files with Mermaid diagram generation prompt.

### Changed
- **Prompt 6 Updated**: End-to-End Flow now includes Layer 5 (UUI) with Shadow Discovery, Scanner, and Double-Pass steps. Layers re-indexed (0-8).
- **Prompt 1 Updated**: Hero diagram now shows 10 nodes (added Unit Recovery + UUI Correction).
- **Dashboard**: Added `🧠 UUI Intelligence (sec)` column. Renamed "Thought Process" to "🧠 Post-Trace".

### Fixed
- **ERR-005**: `SUM(p.rows)` in `sys.tables` queries falsely triggered UUI due to keyword `SUM` matching.
- **ERR-006**: Pasted SQL with whitespace/newlines bypassed metadata detection (fixed with `sql_min` whitespace stripping).
- **ERR-007**: Word `IN` in SQL WHERE clauses partially matched currency code `INR` in exception list.

---

## [April 17, 2026] - The Data Integrity Standard

### Added
- **Universal Unit Intelligence (UUI)**: Comprehensive industry-agnostic unit collision detection.
- **Shadow Discovery**: Out-of-band unit detection logic to prevent false-labeling in SQL results.
- **Deterministic Python Scanner**: Hard-coded guardrail to catch unit mismatches independently of LLM variability.
- **Unit Relevance Filter**: Context-awareness logic to bypass UUI for metadata queries (e.g., Record Counts).
- **Double-Pass Normalization**: Automatic SQL rewriting to perform record-level math for 100% accuracy.
- **Source Attribution**: Transparent tracking of conversion factors (e.g., "Source: DuckDuckGo").
- **UUI Performance Dashboard**: Real-time tracking of unit intelligence processing time.
- **Web Conversion Service**: `src/utils/web_service.py` for live unit factor fetching.

### Changed
- **Pure Aggregation Rule**: SQL generator now forbidden from using `MAX(currency)` or unit labels in summaries.
- **High-Impact UI**: Restored ⚠️ [!CAUTION] icons and headers for integrity alerts.
- **Mobile Styling**: Implemented theme-aware CSS to fix dark-mode contrast issues on phones.

---

## [April 15, 2026] - Performance & Polish

### Added
- **Late-Join Optimization**: Integrated CTE-first aggregation patterns to handle high-volume transaction data.
- **Silent Startup**: Suppressed technical vector-store loading messages for a cleaner UX.
- **Domain Standardized Terminology**: Aligned all reporting to "Margin Contribution" and "Profit Margin" business rules.
- **Code-First Rule**: SQL generation now always calculates profit from raw components, never from pre-calculated columns.
- **Proactive Context Rule**: SQL generator automatically includes currency/unit columns in SELECT statements.

---

## [April 14, 2026] - Agentic Architecture & Visualization

### Added
- **Altair Clustered Bar Charts**: Side-by-side comparison using `xOffset` for multi-metric visualization.
- **Dynamic Time Aggregation**: Auto/Weekly/Monthly/Quarterly/Yearly resampling with smart date detection.
- **Implicit Chart Detection**: Agent auto-triggers chart config when query contains visual keywords.
- **ARCHITECTURE.md**: Full system architecture documentation with Mermaid dependency graphs.
- **DIAGRAM_PROMPTS.md**: 7 reusable prompts for generating professional architecture diagrams.

---

## [April 13, 2026] - Foundation

### Added
- **Python 3.12 Upgrade**: Re-established virtual environment with modern language features.
- **Schema Anchor Check**: Hallucination protection for DB columns (The Echo Test).
- **Knowledge Engine**: Persistent schema cache with description preservation (`data/schema_knowledge.json`).
- **SQL Safety Guard**: DDL/DML keyword blocker in `helper.py`.
- **Exception Hierarchy**: `AgentError` → `LLMError`, `DatabaseError`, `ValidationError`, `SecurityError`.
- **Persistent Metrics**: JSON-based query history with atomic writes.
- **Session Metrics**: In-memory per-session tracking (tokens, timing, errors).
- **File Upload Security**: Filename sanitization, extension whitelist, size limits.
- **Disk Cache GC**: Background garbage collector for stale CSV cache files.
- **Daily Rotating Logs**: Structured file logging via `src/utils/logger.py`.
- **Offline Model Fallback**: Pre-emptive HuggingFace connectivity check with instant offline switch.

---
*Last updated: April 29, 2026*
