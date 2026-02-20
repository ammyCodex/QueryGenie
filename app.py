import streamlit as st
import cohere
import pandas as pd
import sqlite3
import tempfile
import re
import sqlparse
from datetime import datetime
import time

# Import new modules
from config import COHERE_API_KEY, COHERE_MODEL, validate_config
from query_validator import QueryValidator
from safe_executor import SafeQueryExecutor
from logging_audit import audit_logger

# Validate configuration
try:
    validate_config()
except ValueError as e:
    st.error(f"Configuration error: {e}")
    st.stop()

# Page setup
st.set_page_config(page_title="🪄 QueryGenie", layout="wide")

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
try:
    co = cohere.Client(COHERE_API_KEY)
except Exception as e:
    st.error(f"Failed to initialize Cohere: {e}")
    st.stop()

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
if "pending_sql" not in st.session_state:
    st.session_state.pending_sql = None
if "improve_rounds" not in st.session_state:
    st.session_state.improve_rounds = 0

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
        resp = co.chat(model=COHERE_MODEL, message=prompt, temperature=0.3, max_tokens=150)
    return clean_sql_output(resp.text)

def explain_sql(sql):
    prompt = f"""Explain briefly what this SQLite query does in simple terms:\n\n{sql}\n\nExplanation:"""
    resp = co.chat(model=COHERE_MODEL, message=prompt, temperature=0.2, max_tokens=50)
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

        # Generate SQL but hold execution until human approval
        prompt = f"""You are an expert SQLite assistant. Generate only valid SQLite SQL based on user request and DB schema.\n\nUser Request:\n{query}\n\nSchema:\n{format_schema(schema_dict, fk_dict)}\n\nOnly SQL, no explanation:"""
        with st.spinner("💬 Assistant is typing..."):
            time.sleep(0.8)
            sql = generate_sql(prompt)

        # Log query generation
        audit_logger.log_query("PENDING", sql, "Generated by AI, awaiting user approval")

        # Save the generated SQL for human review
        st.session_state.pending_sql = sql
        st.session_state.improve_rounds = 0
        st.session_state.chat_history.append({"role": "assistant", "content": sql, "timestamp": datetime.now(), "type": "assistant"})

        st.session_state.last_query_df = None
        st.session_state.last_sql_query = sql

        # Present human-in-loop controls
        with st.expander("Review generated SQL (approve or improve)"):
            st.markdown(f"```sql\n{st.session_state.pending_sql}\n```")
            cols = st.columns([1,1,1])
            approve = cols[0].button("✅ Approve", key=f"approve_{len(st.session_state.chat_history)}")
            improve = cols[1].button("🛠 Improve", key=f"improve_{len(st.session_state.chat_history)}")
            edit = cols[2].button("✏️ Edit & Execute", key=f"edit_{len(st.session_state.chat_history)}")

            if improve:
                # Ask the model to refine the SQL using the original user intent and schema
                if st.session_state.improve_rounds >= 3:
                    st.warning("Maximum improve attempts reached.")
                    audit_logger.log_query("REJECTED", sql, "Max improve attempts exceeded")
                else:
                    improve_prompt = f"""Improve or fix the following SQLite query to better match the user request.\n\nUser Request:\n{query}\n\nCurrent SQL:\n{st.session_state.pending_sql}\n\nSchema:\n{format_schema(schema_dict, fk_dict)}\n\nReturn only the improved SQL, no explanation:"""
                    with st.spinner("🔁 Improving SQL..."):
                        new_sql = generate_sql(improve_prompt)
                    st.session_state.pending_sql = new_sql
                    st.session_state.improve_rounds += 1
                    st.session_state.chat_history.append({"role": "assistant", "content": new_sql, "timestamp": datetime.now(), "type": "assistant"})
                    audit_logger.log_query("PENDING", new_sql, f"Improved query (round {st.session_state.improve_rounds})")
                    st.experimental_rerun()

            if edit:
                edited_sql = st.text_area("Edit SQL before execution:", value=st.session_state.pending_sql, height=120, key="edited_sql_area")
                if st.button("Execute Edited SQL", key="exec_edited"):
                    st.session_state.pending_sql = edited_sql
                    approve = True

            if approve:
                sql_to_run = st.session_state.pending_sql
                
                # Log approval
                audit_logger.log_approval(sql_to_run, approved=True)
                
                st.session_state.chat_history.append({"role": "assistant", "content": "✅ Approved by user — executing SQL...", "timestamp": datetime.now(), "type": "assistant"})
                
                try:
                    # Create safe executor and validate/execute query
                    executor = SafeQueryExecutor(st.session_state.conn)
                    success, message, results = executor.validate_and_execute(
                        sql_to_run,
                        audit_log_func=audit_logger.log_query
                    )
                    
                    if success and results:
                        # Convert results to DataFrame
                        if results:
                            columns = list(results[0].keys()) if isinstance(results[0], dict) else [f"col_{i}" for i in range(len(results[0]))]
                            st.session_state.last_query_df = pd.DataFrame(results)
                        res = message
                    else:
                        res = message if message else "Query could not be executed"
                        
                except Exception as e:
                    res = f"❌ Error: {str(e)}"
                    audit_logger.log_query("ERROR", sql_to_run, str(e))

                st.session_state.chat_history.append({"role": "assistant", "content": res, "timestamp": datetime.now(), "type": "result"})

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
