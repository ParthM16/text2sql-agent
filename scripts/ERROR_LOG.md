# DATA INTEGRITY ERROR LOG

Historical tracking of technical challenges, regressions, and architectural solutions encountered during the development of the Text-to-SQL Unit Intelligence layer.

## [ERR-001] The Apples + Oranges Summation Error
- **Symptom**: Agent was aggregating mixed currencies (INR + YEN) and then converting the TOTAL.
- **Root Cause**: Math was performed on raw database totals without field-level normalization.
- **Solution**: Implemented **Double-Pass Normalization**. The agent now rewrites its SQL to include `CASE` statements for record-level conversion inside the database.

## [ERR-002] False Labeling via `MAX(currency)`
- **Symptom**: Aggregated results showed "YEN" for totals that actually included "INR" values.
- **Root Cause**: SQL generation logic was using `MAX(currency)` as a shortcut to get a unit label for a group.
- **Solution**: Implemented **Shadow Unit Discovery**. All unit/currency columns were removed from aggregated SQL queries. Units are now identified via silent out-of-band discovery queries.

## [ERR-003] UUI False Positives on Metadata
- **Symptom**: "How many records?" queries triggered currency integrity alerts and conversion loops.
- **Root Cause**: The Deterministic Unit Scanner triggered on *any* table involvement, regardless of whether the user asked for a numerical metric.
- **Solution**: Added a **Unit Relevance Filter**. The UUI loop now only fires if the query/SQL contains unit-sensitive keywords (Amount, Price, Weight, etc.). Metadata counts are now hard-exempt.

## [ERR-004] Mobile Dark-Mode Visibility
- **Symptom**: Assistant response text was white-on-white on mobile phones in dark mode.
- **Root Cause**: Static light backgrounds in CSS were clashing with Streamlit's automatic theme text-color overrides.
- **Solution**: Implemented **Theme-Aware CSS**. Forced high-contrast text (#1f1f1f) for all light-background containers and used `var(--text-color)` for adaptive main text.

## [ERR-005] SUM(p.rows) False-Positive Unit Trigger
- **Date**: April 18, 2026
- **Symptom**: Query `SELECT t.name, SUM(p.rows) FROM sys.tables` triggered the full UUI pipeline (21+ seconds) despite being a simple record count.
- **Root Cause**: The keyword `SUM` in `SUM(p.rows)` matched the unit-relevance keyword list, causing the agent to treat a metadata query as a financial aggregation.
- **Solution**: Implemented **Hard Metadata Bypass (Negative-First)**. The agent now checks for system table indicators (`sys.`, `information_schema`, `count(*)`, `tablename`, `rowcount`) BEFORE checking for unit-sensitive keywords. Metadata queries are completely exempt from UUI processing.

## [ERR-006] Whitespace-Blind Detection Gap
- **Date**: April 18, 2026
- **Symptom**: Pasting formatted SQL (with newlines and extra spaces like `SUM ( p.rows )`) bypassed the metadata shield because exact string matching failed on whitespace variations.
- **Root Cause**: The metadata keyword check used simple `in` operator on the raw SQL string, which failed when SQL had extra whitespace between function names and arguments.
- **Solution**: Added `sql_min = "".join(sql_clean.split())` to create a whitespace-stripped version for robust matching. Later simplified to broader keyword detection on `sql_clean` directly.

## [ERR-007] IN/INR Word-Boundary Collision
- **Date**: April 18, 2026
- **Symptom**: Even after metadata detection, the UUI override exception was being triggered because the SQL keyword `IN` (e.g., `WHERE name IN (...)`) partially matched the currency code `INR` in the exception list.
- **Root Cause**: The exception check used simple `kw in user_query.lower()`, which matched `IN` as a substring of `INR`.
- **Solution**: Upgraded to **Regex word-boundary matching** (`re.search(rf'\b{kw}\b', ...)`) ensuring that only standalone currency words (like `INR`, `YEN`) trigger the override, not SQL keywords.

---
*Last updated: April 23, 2026*
