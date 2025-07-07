import streamlit as st
import cohere
import pandas as pd
import sqlite3
import tempfile
import re
import sqlparse
from datetime import datetime
import time

# Page setup
st.set_page_config(page_title="🪄 QueryGenie Chat (Dark Mode)", layout="wide")

# Custom CSS for modern dark theme & chat bubbles
custom_css = """
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
.chat-bubble {
    display: flex;
    flex-direction: column;
    margin: 16px 0;
    padding: 12px 20px;
    border: none;
    background-color: #1e1e1e;
    border-radius: 10px;
    color: #f5f5f5;
    max-width: 80%;
    animation: fadeIn 0.4s ease-in;
}
.chat-bubble.user {
    align-self: flex-end;
    background-color: #2a2a2a;
    text-align: left;
}
.chat-bubble.assistant {
    align-self: flex-start;
    background-color: #2d2d2d;
    text-align: left;
}
.chat-author {
    font-weight: 600;
    font-size: 13px;
    color: #bbb;
    display: flex;
    justify-content: space-between;
    margin-bottom: 6px;
    margin-top: -2px;
    padding-left: 4px;
    padding-right: 4px;
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# JS for toast sound
st.markdown("""
<script>
function playSound() {
    new Audio("https://actions.google.com/sounds/v1/cartoon/clang_and_wobble.ogg").play();
}
</script>
""", unsafe_allow_html=True)

# Initialize Cohere
COHERE_API_KEY = st.secrets["COHERE_API_KEY"]
co = cohere.Client(COHERE_API_KEY)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "db_path" not in st.session_state:
    st.session_state.db_path = None
if "upload_shown" not in st.session_state:
    st.session_state.upload_shown = False
if "conn" not in st.session_state:
    st.session_state.conn = None
if "cursor" not in st.session_state:
    st.session_state.cursor = None
if "last_query_df" not in st.session_state:
    st.session_state.last_query_df = None
if "last_sql_query" not in st.session_state:
    st.session_state.last_sql_query = None

# Sidebar for DB upload and schema display
with st.sidebar:
    st.header("🪄 QueryGenie")
    st.subheader("📤 Upload SQLite DB")
    uploaded_db = st.file_uploader("Upload a .sqlite or .db file", type=["sqlite", "db"])

    if uploaded_db:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite")
        temp_file.write(uploaded_db.read())
        temp_file.close()
        st.session_state.db_path = temp_file.name
        if st.session_state.conn:
            st.session_state.conn.close()
        st.session_state.conn = sqlite3.connect(st.session_state.db_path, check_same_thread=False)
        st.session_state.cursor = st.session_state.conn.cursor()

        # Reset upload toast flag for every new upload
        st.session_state.upload_shown = False

        if not st.session_state.upload_shown:
            st.toast("📤 Database uploaded successfully!", icon="✅")
            st.markdown("<script>playSound()</script>", unsafe_allow_html=True)
            st.session_state.upload_shown = True

    if st.session_state.db_path and st.session_state.conn:
        def get_schema():
            schema = {}
            foreign_keys = {}
            tables = st.session_state.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
            for (table_name,) in tables:
                columns = st.session_state.cursor.execute(f"PRAGMA table_info('{table_name}');").fetchall()
                schema[table_name] = [col[1] for col in columns]
                fks = st.session_state.cursor.execute(f"PRAGMA foreign_key_list('{table_name}');").fetchall()
                foreign_keys[table_name] = [(fk[3], fk[2]) for fk in fks]
            return schema, foreign_keys

        schema_dict, fk_dict = get_schema()
        def format_schema(schema, fks):
            return "\n".join([
                f"**{t}**: {', '.join(cols)}" + 
                (f"\nForeign Keys: {', '.join([f'{col} → {ref}' for col, ref in fks[t]])}" if fks[t] else "") 
                for t, cols in schema.items()
            ])

        st.header("📋 DB Schema")
        st.markdown(format_schema(schema_dict, fk_dict), unsafe_allow_html=True)
    else:
        st.info("Upload a SQLite file above to get started.")

# Utilities
def clean_sql_output(text):
    return re.sub(r"```(?:sql)?\s*(.*?)```", r"\1", text, flags=re.DOTALL | re.IGNORECASE).strip()

def generate_sql(prompt):
    with st.spinner("🤖 Generating SQL..."):
        resp = co.chat(model="command-r-plus", message=prompt, temperature=0.3, max_tokens=150)
    return clean_sql_output(resp.text)

def explain_sql(sql):
    prompt = f"""Explain briefly what this SQLite query does in simple terms:\n\n{sql}\n\nExplanation:"""
    resp = co.chat(model="command-r-plus", message=prompt, temperature=0.2, max_tokens=50)
    return resp.text.strip()

def render_chat():
    for entry in st.session_state.chat_history:
        ts = entry["timestamp"].strftime('%H:%M:%S')
        if entry["type"] in ["user", "assistant", "result"]:
            role = "You" if entry["type"] == "user" else "Assistant"
            bubble_class = "user" if entry["type"] == "user" else "assistant"
            content_html = entry["content"].replace("\n", "<br>").replace("```sql", "").replace("```", "")
            st.markdown(f'''<div class="chat-bubble {bubble_class}">
                <div class="chat-author"><span>{role}</span><span>{ts}</span></div>
                {content_html}
            </div>''', unsafe_allow_html=True)
        else:
            st.markdown(f"*{entry['content']}*")

def is_select_only(sql_text):
    parsed = sqlparse.parse(sql_text)
    for stmt in parsed:
        if stmt.get_type() != "SELECT":
            return False
    return True

# Main UI
st.title("🪄 QueryGenie")
query = st.text_area("💬 Ask your database...", height=100, key="chat_input")

if st.button("Send"):
    if not query.strip():
        st.error("Please enter your query.")
    elif st.session_state.db_path is None:
        st.error("Please upload a SQLite DB first.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": query, "timestamp": datetime.now(), "type": "user"})

        prompt = f"""You are an expert SQLite assistant. Generate only valid SQLite SQL based on user request and DB schema.\n\nUser Request:\n{query}\n\nSchema:\n{format_schema(schema_dict, fk_dict)}\n\nOnly SQL, no explanation:"""
        with st.spinner("💬 Assistant is typing..."):
            time.sleep(0.8)
            sql = generate_sql(prompt)

        st.session_state.chat_history.append({"role": "assistant", "content": sql, "timestamp": datetime.now(), "type": "assistant"})

        result = ""
        st.session_state.last_query_df = None

        try:
            if is_select_only(sql):
                df = pd.read_sql_query(sql, st.session_state.conn)
                if not df.empty:
                    result = "Here is the result:"
                    st.session_state.last_query_df = df
                else:
                    result = "_Query returned no results._"
            else:
                result = (
                    "⚠️ The generated query modifies the database "
                    "(INSERT/UPDATE/DELETE or DDL). Execution is not permitted.\n\n"
                    "You can review the SQL below:\n\n"
                    f"```sql\n{sql}\n```"
                )
        except Exception as e:
            result = f"❌ Error: {e}"

        st.session_state.chat_history.append({"role": "assistant", "content": result, "timestamp": datetime.now(), "type": "result"})
        st.session_state.last_sql_query = sql

# Render chat history
render_chat()

# Render last query dataframe below chat bubbles
if st.session_state.last_query_df is not None:
    st.table(st.session_state.last_query_df)

# Explain last SQL button
if st.session_state.last_sql_query:
    if st.button("💡 Explain Last SQL"):
        explanation = explain_sql(st.session_state.last_sql_query)
        st.session_state.chat_history.append({"role": "assistant", "content": explanation, "timestamp": datetime.now(), "type": "assistant"})
        st.rerun()
