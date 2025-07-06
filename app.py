import streamlit as st
import cohere
import pandas as pd
import sqlite3
import tempfile
import re
from datetime import datetime

# Page config and styling
st.set_page_config(page_title="QueryGenie", layout="wide")

dark_css = """
<style>
body, .block-container {
    background-color: #121212;
    color: #e0e0e0;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #1f1f1f;
    color: #e0e0e0;
}
code, pre {
    background-color: #1e1e1e !important;
    color: #f8f8f2 !important;
    font-family: 'Source Code Pro', monospace !important;
    font-size: 13px !important;
    padding: 8px !important;
    border-radius: 8px !important;
    white-space: pre-wrap !important;
    word-break: break-word !important;
}
input[type="text"] {
    background-color: #222222 !important;
    color: #e0e0e0 !important;
}
</style>
"""
st.markdown(dark_css, unsafe_allow_html=True)

# Initialize Cohere client
COHERE_API_KEY = st.secrets["COHERE_API_KEY"]
co = cohere.Client(COHERE_API_KEY)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "db_path" not in st.session_state:
    st.session_state.db_path = None
if "upload_shown" not in st.session_state:
    st.session_state.upload_shown = False

# Sidebar: Upload + Schema + Download
with st.sidebar:
    st.header("📤 Upload SQLite DB")
    uploaded_db = st.file_uploader("Upload a .sqlite or .db file", type=["sqlite", "db"])

    if uploaded_db:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite")
        temp_file.write(uploaded_db.read())
        temp_file.close()
        st.session_state.db_path = temp_file.name

        if not st.session_state.upload_shown:
            st.session_state.chat_history.append({
                "role": "system",
                "content": f"📤 Database **'{uploaded_db.name}'** loaded! Size: {uploaded_db.size / 1e6:.1f} MB",
                "timestamp": datetime.now(),
                "type": "info"
            })
            st.session_state.upload_shown = True

    if st.session_state.db_path:
        conn = sqlite3.connect(st.session_state.db_path)
        cursor = conn.cursor()

        def get_schema():
            schema = {}
            foreign_keys = {}
            tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
            for (table_name,) in tables:
                columns = cursor.execute(f"PRAGMA table_info('{table_name}');").fetchall()
                schema[table_name] = [col[1] for col in columns]
                fks = cursor.execute(f"PRAGMA foreign_key_list('{table_name}');").fetchall()
                foreign_keys[table_name] = [(fk[3], fk[2]) for fk in fks]
            return schema, foreign_keys

        def format_schema(schema, fks):
            lines = []
            for table, cols in schema.items():
                lines.append(f"**{table}**: {', '.join(cols)}")
                if fks.get(table):
                    fk_lines = [f"{col} → {ref_table}" for col, ref_table in fks[table]]
                    lines.append(f"Foreign Keys: {', '.join(fk_lines)}")
                lines.append("")
            return "\n".join(lines)

        schema_dict, foreign_keys_dict = get_schema()
        st.header("📋 DB Schema & Foreign Keys")
        st.markdown(format_schema(schema_dict, foreign_keys_dict), unsafe_allow_html=True)

        st.markdown("---")
        if st.button("📥 Download Updated Database"):
            with open(st.session_state.db_path, "rb") as f:
                st.download_button(
                    label="Download SQLite DB",
                    data=f.read(),
                    file_name=uploaded_db.name,
                    mime="application/octet-stream"
                )
    else:
        st.info("Upload a SQLite file above to get started.")

# Utility functions
def clean_sql_output(response_text: str) -> str:
    cleaned = re.sub(r"```(?:sql)?\s*(.*?)```", r"\1", response_text, flags=re.DOTALL | re.IGNORECASE)
    match = re.search(r"(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|WITH|EXPLAIN).*?;", cleaned, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(0).strip()
    return cleaned.strip()

def generate_sql(prompt):
    with st.spinner("🤖 Generating SQL..."):
        response = co.chat(
            model="command-r-plus",
            message=prompt,
            temperature=0.3,
            max_tokens=150
        )
    return clean_sql_output(response.text)

def explain_sql(sql):
    with st.spinner("🤔 Explaining SQL..."):
        prompt = f"Explain this SQL query briefly in simple English:\n\nSQL:\n{sql}\n\nExplanation:"
        response = co.chat(
            model="command-r-plus",
            message=prompt,
            temperature=0.2,  # lower temp for concise focused answers
            max_tokens=80,    # limit tokens to keep explanation brief
            stop_sequences=["\n\n"]  # stop at paragraph breaks if possible
        )
    return response.text.strip()


# Input box and send button at top
st.title("QueryGenie 🪄")

query = st.text_input("💬 Type your query here", key="user_input_box", placeholder="Ask me to fetch or update your DB...")

send_pressed = st.button("Send", key="send_button")

if send_pressed:
    if not query.strip():
        st.error("Please enter your query.")
    elif st.session_state.db_path is None:
        st.error("Please upload a SQLite DB first.")
    else:
        st.session_state.chat_history.append({
            "role": "user",
            "content": query,
            "timestamp": datetime.now(),
            "type": "user"
        })

        # Build prompt for Cohere
        prompt = f"""You are a helpful assistant that converts natural language into valid SQLite SQL queries.

Here is the database schema with tables and columns (including foreign keys):
{format_schema(schema_dict, foreign_keys_dict)}

User request:
{query}

SQL:
"""

        sql_query = generate_sql(prompt)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": sql_query,
            "timestamp": datetime.now(),
            "type": "sql"
        })

        # Run SQL and show results with Streamlit table (not inside chat bubble)
        try:
            conn = sqlite3.connect(st.session_state.db_path)
            if sql_query.lower().startswith("select"):
                df = pd.read_sql_query(sql_query, conn)
                if df.empty:
                    result_text = "_Query returned no results._"
                else:
                    result_text = "Here is the result:"
            else:
                cursor = conn.cursor()
                cursor.execute(sql_query)
                conn.commit()
                result_text = "✅ SQL executed successfully!"
        except Exception as e:
            result_text = f"❌ SQL execution error:\n{e}"

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result_text,
            "timestamp": datetime.now(),
            "type": "result"
        })

        st.rerun()

# Simple chat renderer (no divs, just markdown and code blocks)
def render_chat_simple():
    for entry in st.session_state.chat_history:
        role = entry["role"]
        timestamp = entry["timestamp"].strftime('%H:%M:%S')
        content = entry["content"]
        if entry["type"] == "sql":
            st.code(content, language="sql")
        else:
            if role == "user":
                st.markdown(f"**You [{timestamp}]:** {content}")
            elif role == "assistant":
                st.markdown(f"**Assistant [{timestamp}]:** {content}")
            else:  # system or info
                st.markdown(f"*{content}*")
        st.markdown("---")

render_chat_simple()

# Display last query result table outside chat (if exists)
if st.session_state.chat_history:
    last = st.session_state.chat_history[-1]
    if last['type'] == "result" and "Here is the result:" in last['content']:
        try:
            # Find last SQL query from chat history
            last_sql = None
            for entry in reversed(st.session_state.chat_history):
                if entry["type"] == "sql":
                    last_sql = entry["content"].strip()
                    break
            if last_sql:
                conn = sqlite3.connect(st.session_state.db_path)
                df = pd.read_sql_query(last_sql, conn)
                st.table(df)
        except Exception:
            pass

# Explain last SQL button
last_sql = None
for entry in reversed(st.session_state.chat_history):
    if entry["type"] == "sql":
        last_sql = entry["content"].strip()
        break

if last_sql and st.button("💡 Explain last SQL query"):
    explanation = explain_sql(last_sql)
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": explanation,
        "timestamp": datetime.now(),
        "type": "explanation"
    })
    st.rerun()
