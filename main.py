import streamlit as st

# Set page config globally in the main entrypoint
st.set_page_config(
    page_title="Retail Data Agent",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define your pages and their sidebar appearances
agent_page = st.Page("app.py", title="Chat Agent", icon="💬", default=True)
dashboard_page = st.Page("pages/1_📊_Dashboard.py", title="Prompt Analytics", icon="📊")

# Initialize the navigation
pg = st.navigation([agent_page, dashboard_page])

# Run the selected page
pg.run()
