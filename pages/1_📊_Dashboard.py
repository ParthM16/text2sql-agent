import streamlit as st
import pandas as pd
import altair as alt

from src.utils.metrics import PersistentMetrics

st.set_page_config(page_title="Agent Dashboard", layout="wide")

st.title("📊 Text2SQL Agent — Dashboard")

history = PersistentMetrics.get_query_history()
if not history:
    st.info("No query history yet. Run some queries to populate the dashboard.")
    st.stop()

df = pd.DataFrame(history)

# Filters
with st.sidebar.expander("Filters"):
    status_opts = ["ALL", "SUCCESS", "FAIL"]
    status = st.selectbox("Status", status_opts, index=0)
    min_date = pd.to_datetime(df['timestamp']).min()
    max_date = pd.to_datetime(df['timestamp']).max()
    date_range = st.date_input("Date range", [min_date.date(), max_date.date()])

    if status != "ALL":
        if status == "SUCCESS":
            df = df[df['success'] == True]
        else:
            df = df[df['success'] == False]
    if date_range and len(date_range) == 2:
        start, end = date_range
        df['ts'] = pd.to_datetime(df['timestamp']).dt.date
        df = df[(df['ts'] >= start) & (df['ts'] <= end)]

# Top-level metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Queries", len(df))
col2.metric("Success Rate", f"{(df.success.mean() * 100 if 'success' in df.columns and len(df)>0 else 0):.1f}%")
col3.metric("Avg Tokens", f"{(df.tokens_used.map(lambda x: x.get('total_tokens') if isinstance(x, dict) else x).mean() if 'tokens_used' in df.columns and len(df)>0 else 0):.1f}")
col4.metric("Avg Response (sec)", f"{(df.response_time_sec.map(lambda x: x or 0).mean() if 'response_time_sec' in df.columns and len(df)>0 else 0):.2f}")

st.subheader("Query History")

# 🛠️ Reorder and Refine columns for a professional view
display_df = df.copy()
# Hide all helper/legacy columns
for col in ['response_time_ms', 'ts']:
    if col in display_df.columns:
        display_df = display_df.drop(columns=[col])

# Define EXACT user-requested column order
# [time, query, success, sql, intent, total time, uui time, thought process, tokens used]
display_df['🧠 UUI Intelligence (sec)'] = display_df['node_timings'].map(
    lambda x: round(x.get('09a_unit_validation_total', 0), 3) if isinstance(x, dict) else 0
)

preferred_order = ['timestamp', 'user_query', 'success', 'generated_sql', 'intent', 'response_time_sec', '🧠 UUI Intelligence (sec)', 'node_timings', 'tokens_used']
actual_cols = [c for c in preferred_order if c in display_df.columns]
remaining_cols = [c for c in display_df.columns if c not in preferred_order]

display_df = display_df[actual_cols + remaining_cols]
display_df = display_df.sort_values("timestamp", ascending=False).reset_index(drop=True)

# Rename for a cleaner UI
display_df = display_df.rename(columns={
    'timestamp': '🕒 Time',
    'user_query': '💬 Query',
    'success': '✅ Success',
    'generated_sql': '📜 SQL',
    'intent': '🎯 Intent',
    'response_time_sec': '⏱️ Total Time (sec)',
    'node_timings': '🧠 Post-Trace',
    'tokens_used': '🪙 Tokens Used'
})

st.dataframe(display_df)

# Per-node averages
st.subheader("Per-node Performance (averages)")
node_stats = {}
if 'node_timings' in df.columns:
    for row in df['node_timings']:
        if not isinstance(row, dict):
            continue
        for k, v in row.items():
            node_stats.setdefault(k, []).append(v)

if node_stats:
    stats = [{ 'node': k, 'avg_ms': sum(vs)/len(vs)*1000, 'calls': len(vs)} for k, vs in node_stats.items()]
    stats_df = pd.DataFrame(stats).sort_values('avg_ms', ascending=False)
    st.table(stats_df)
else:
    st.info("No per-node timing data available yet.")

# Error log
st.subheader("Error Log")
errors = []
if 'node_timings' in df.columns and 'success' in df.columns:
    # Build error rows from failed entries
    for r in history:
        if not r.get('success', True):
            errors.append({
                'timestamp': r.get('timestamp'),
                'user_query': r.get('user_query'),
                'intent': r.get('intent'),
                'node_timings': r.get('node_timings'),
            })
if errors:
    st.dataframe(pd.DataFrame(errors))
else:
    st.info("No errors recorded.")

# Token usage chart (if available)
if 'tokens_used' in df.columns:
    tokens = df.copy()
    tokens['ts'] = pd.to_datetime(tokens['timestamp'])
    tokens['total_tokens'] = tokens['tokens_used'].map(lambda t: t.get('total_tokens') if isinstance(t, dict) else t)
    tokens = tokens.dropna(subset=['total_tokens'])
    if not tokens.empty:
        st.subheader("Token Usage Over Time")
        chart = alt.Chart(tokens).mark_line(point=True).encode(
            x='ts:T',
            y='total_tokens:Q',
            tooltip=['timestamp', 'user_query', 'total_tokens']
        ).interactive()
        st.altair_chart(chart, width='stretch')

# Response time chart
if 'response_time_sec' in df.columns:
    st.subheader("Response Time (sec)")
    rt = df.copy()
    rt['ts'] = pd.to_datetime(rt['timestamp'])
    rt = rt.dropna(subset=['response_time_sec'])
    if not rt.empty:
        chart = alt.Chart(rt).mark_line().encode(
            x='ts:T',
            y='response_time_sec:Q',
            tooltip=['timestamp', 'user_query', 'response_time_sec']
        )
        st.altair_chart(chart, width='stretch')

# Export
st.download_button("Export History CSV", df.to_csv(index=False), file_name="query_history.csv")
