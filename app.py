"""
Retail Insights Agent — Streamlit Entry Point
===============================================
Run with:  streamlit run app.py
"""
import sys
import os
import threading

# ── Production Environment Variables ─────────────────────────
os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), "models")

from src.utils.metrics import PersistentMetrics
import streamlit as st
import pandas as pd
from src.db.helper import DBHelper
from src.retrieval.vector_store import VectorStore
from src.agent.workflow import TextToSQLAgent
from src.utils.disk_cache import clean_old_cache, save_to_cache
from datetime import datetime
import time
import streamlit.components.v1 as components

# ── Background Disk Cache GC ─────────────────────────────────
try:
    clean_old_cache(max_age_hours=1)
except Exception:
    pass

# ── Page Config ──────────────────────────────────────────────
# Moved to main.py for st.navigation compatibility.

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Base App Styling */
    .stApp { color: var(--text-color); }
    
    .main-header {
        font-family: 'Inter', sans-serif;
        color: #0071ce !important;
        text-align: center;
        padding: 15px 20px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }

    /* Customize Streamlit's main block container padding */
    div[data-testid="stMainBlockContainer"] {
        padding-top: 40px !important;
        padding-bottom: 0px !important;
    }

    /* Universal Unit Intelligence (UUI) Alert Styles */
    .stAlert {
        border-left: 5px solid #ff4b4b !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
    }
    blockquote.stMarkdownContainer {
        border-left-color: #ff4b4b !important;
        background: #fffafa !important;
        color: #1f1f1f !important;
        padding: 10px 15px !important;
        border-radius: 5px !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Caching Heavy Resources ──────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_db_helper(): return DBHelper()

@st.cache_resource(show_spinner=False)
def get_vector_store(): return VectorStore()

# ── Session State Initialization ─────────────────────────────
if "db_helper" not in st.session_state:
    st.session_state.db_helper = get_db_helper()
if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vector_store()
if "agent" not in st.session_state:
    st.session_state.agent = TextToSQLAgent(st.session_state.db_helper, st.session_state.vector_store)
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help with your retail data today?"}]

SUPPORTED_LANGUAGES = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de",
    "Chinese": "zh-cn", "Japanese": "ja", "Korean": "ko", "Arabic": "ar",
    "Hindi": "hi", "Gujarati": "gu", "Portuguese": "pt", "Russian": "ru", "Italian": "it"
}

col1, col2 = st.columns([8, 2])
with col1:
    st.markdown("<h1 class='main-header'><span class='header-anchor'></span>🛒 Retail Insights Agent</h1>", unsafe_allow_html=True)
with col2:
    selected_lang_name = st.selectbox("🌐 Language", list(SUPPORTED_LANGUAGES.keys()), index=0)
    if "app_language" not in st.session_state or st.session_state.app_language != SUPPORTED_LANGUAGES[selected_lang_name]:
        st.session_state.app_language = SUPPORTED_LANGUAGES[selected_lang_name]

# ── Data Source Badge ────────────────────────────────────────
db_mode = st.session_state.db_helper.mode
if db_mode == "sqlite" and st.session_state.db_helper.sqlite_engine:
    st.info("🟡 **Active file:** CSV/Excel uploaded (in-memory SQLite)")
elif db_mode == "sqlserver":
    st.success(f"🟢 **Connected:** SQL Server — {os.getenv('DB_DATABASE_NAME', 'RetailDB')}")

# ── Sidebar ──────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

with st.sidebar:
    st.header("1. Upload Data (Optional)")
    uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
    if uploaded_file:
        save_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(save_path, "wb") as f: f.write(uploaded_file.getbuffer())
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(save_path)
            csv_path = save_path.replace(".xlsx", ".csv")
            df.to_csv(csv_path, index=False)
            save_path = csv_path
        table_name = st.session_state.db_helper.load_csv_to_memory(save_path)
        st.success(f"Loaded: `{table_name}`")
    else:
        # ── Reset to SQL Server when CSV is removed ──────────
        if st.session_state.db_helper.mode == "sqlite":
            st.session_state.db_helper.mode = "sqlserver"
            st.session_state.db_helper.sqlite_engine = None
            st.info("📡 Switched back to SQL Server mode.")
    
    st.markdown("---")
    st.header("2. Knowledge Engine")
    if st.button("🔄 Sync Database Knowledge", width="stretch"):
        with st.spinner("Synchronizing..."):
            ok, err = st.session_state.db_helper.sync_knowledge_base()
            if ok: 
                # Flush the UUI memory cache so new currencies are detected instantly
                st.session_state.agent._unit_cache = {}
                st.success("Agent Knowledge Updated & Unit Cache Cleared!")
            else: st.error(f"Sync failed: {err}")

    st.markdown("---")
    st.markdown("**📊 Session Stats**")
    metrics = st.session_state.agent.metrics
    c1, c2 = st.columns(2)
    c1.metric("Queries", metrics.total_queries)
    c2.metric("✅ Success", metrics.successful_queries)
    c3, c4 = st.columns(2)
    c3.metric("🔑 Tokens", f"{metrics.tokens_total:,}")
    avg_t = metrics.get_avg_response_time()
    c4.metric("⏱️ Avg", f"{avg_t:.1f}s" if avg_t else "—")
    
    st.markdown("---")
    st.markdown("**🕘 Recent Queries**")
    history = PersistentMetrics.get_query_history()[-3:]
    if history:
        for i, q in enumerate(reversed(history)):
            label = q.get("user_query", "")[:40] + "..."
            if st.button(f"🔍 {label}", key=f"hist_{i}", width="stretch"):
                st.session_state.rerun_query = q.get("user_query")
                st.rerun()
        if st.button("🗑️ Clear History", width="stretch"):
            PersistentMetrics.clear_query_history()
            st.rerun()

# ── Visual Rendering Engine ──────────────────────────────────
@st.fragment
def render_visuals(msg_id, chart_config, chart_df, head_df, dl_path, logs):
    """Renders visualizations in an isolated fragment."""
    if chart_config and chart_df is not None and not chart_df.empty:
        try:
            chart_type = chart_config.get("chart_type", "bar")
            display_df = chart_df.copy()
            
            # Smart Date Detection & Aggregation (Threshold > 10)
            if len(display_df) > 10:
                import pandas as pd
                temp_index = pd.to_datetime(display_df.index, errors='coerce')
                if temp_index.isnull().all():
                    for col in display_df.columns:
                        col_dt = pd.to_datetime(display_df[col], errors='coerce')
                        if col_dt.notnull().sum() > len(display_df) * 0.8:
                            display_df.index = col_dt
                            break
                else:
                    display_df.index = temp_index

                if display_df.index.dtype.kind in 'M' or pd.api.types.is_datetime64_any_dtype(display_df.index):
                    display_df = display_df[display_df.index.notnull()]
                    for col in display_df.columns:
                        display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                    
                    agg_opt = st.radio("📈 Time Aggregation:", ["Auto", "Raw", "Weekly", "Monthly", "Quarterly", "Yearly"], horizontal=True, key=f"rad_{msg_id}_{dl_path}")

                    
                    days = (display_df.index.max() - display_df.index.min()).days
                    rule = None
                    if agg_opt == "Yearly": rule = "YE"
                    elif agg_opt == "Quarterly": rule = "QE"
                    elif agg_opt == "Monthly": rule = "ME"
                    elif agg_opt == "Weekly": rule = "W"
                    elif agg_opt == "Auto":
                        if days > 730: rule = "QE"
                        elif days > 180: rule = "ME"
                        elif days > 30: rule = "W"
                    
                    if rule:
                        display_df = display_df.resample(rule).sum(numeric_only=True)
                        if "YE" in rule: display_df.index = display_df.index.to_period('Y').astype(str)
                        elif "QE" in rule: display_df.index = display_df.index.to_period('Q').astype(str)
                        elif "ME" in rule: display_df.index = display_df.index.to_period('M').astype(str)
                        elif "W" in rule: display_df.index = display_df.index.strftime('%Y-%m-%d')
                    elif len(display_df) > 1000:
                        display_df = display_df.sample(n=1000).sort_index()
            
            # Chart rendering
            if chart_type == "line": st.line_chart(display_df)
            elif chart_type == "area": st.area_chart(display_df)
            else:
                # ── Clustered Bar Chart (Side-by-Side Comparison) ────
                if len(display_df.columns) > 1:
                    import altair as alt
                    plot_df = display_df.reset_index()
                    x_col = plot_df.columns[0]
                    y_cols = list(display_df.columns)
                    
                    # Melt to long-format for Altair clustering
                    melted = plot_df.melt(id_vars=[x_col], value_vars=y_cols, var_name='Metric', value_name='Value')
                    
                    # Create a responsive clustered bar chart
                    chart = alt.Chart(melted).mark_bar().encode(
                        x=alt.X(f'{x_col}:N', title=None, axis=alt.Axis(labelAngle=0 if len(plot_df) < 15 else -45)),
                        y=alt.Y('Value:Q', title=None, axis=alt.Axis(grid=True, format='.2s')),
                        color=alt.Color('Metric:N', legend=alt.Legend(orient="top", title=None)),
                        xOffset='Metric:N'
                    ).properties(height=400)
                    
                    st.altair_chart(chart, width='stretch')
                else:
                    st.bar_chart(display_df)
        except Exception as e:
            st.warning(f"Chart Render Error: {e}")

    if head_df is not None and not head_df.empty:
        with st.expander("📊 Data Preview", expanded=False):
            if dl_path: st.warning("Showing first 50 rows.")
            st.dataframe(head_df, width="stretch")
        if dl_path:
            with open(dl_path, "rb") as f:
                st.download_button("💾 Download Full CSV", data=f, file_name="results.csv", mime="text/csv", key=f"dl_{msg_id}_{dl_path}")

    if logs:
        with st.expander("🧭 Agent Thought Process", expanded=False):
            for l in logs: st.text(l)

# ── Main Chat Interface ──────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help with your retail data today?"}]

# ── Create a scrolling container for Chat so the Header stays fixed ──
chat_container = st.container(height=570, border=False)

with chat_container:
    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            render_visuals(i, m.get("chart_config"), m.get("chart_df"), m.get("head_df"), m.get("dl_path"), m.get("logs"))


# ── Chat Input ──
prompt = st.session_state.pop("rerun_query", None) or st.chat_input("Ask a question...", max_chars=500)

if prompt:
    # 1. Add user message to state and render it immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Run agent synchronously with a spinner
        with st.chat_message("assistant"):
            with st.spinner("🧠 Working..."):
                agent_ref = st.session_state.agent
                app_lang  = st.session_state.get("app_language", "en")
                try:
                    # Synchronous call - zero overhead
                    ans, logs, cfg, df = agent_ref.run(prompt, app_language=app_lang)
                except Exception as e:
                    ans, logs, cfg, df = f"Agent error: {e}", [], None, None

                # 3. Process the results for visualization
                c_df, h_df, d_pt = None, None, None
                if df is not None and not df.empty:
                    h_df = df.head(50)
                    if len(df) > 50: d_pt = save_to_cache(df)
                    if cfg:
                        xc, yc, cc = cfg["x"], cfg["y"], cfg.get("color")
                        if isinstance(yc, str): yc = [yc]
                        if cc and len(yc) == 1:
                            c_df = df.pivot_table(index=xc, columns=cc, values=yc[0], aggfunc="sum").fillna(0)
                        else:
                            c_df = df.set_index(xc)[yc]

                # 4. Render the assistant response and visuals
                st.markdown(ans)
                render_visuals(len(st.session_state.messages), cfg, c_df, h_df, d_pt, logs)

                # 5. Save to message history
                st.session_state.messages.append({
                    "role": "assistant", "content": ans, "logs": logs,
                    "chart_config": cfg, "chart_df": c_df, "head_df": h_df, "dl_path": d_pt
                })
    
    # 6. Final rerun to sync state
    st.rerun()




