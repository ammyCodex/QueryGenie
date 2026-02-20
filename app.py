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
from logging_audit import audit_logger

# Validate configuration
try:
    validate_config()
except ValueError as e:
    st.error(f"Configuration error: {e}")
    st.stop()

# Page setup
st.set_page_config(page_title="🪄 QueryGenie", layout="wide")

# Custom CSS
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
    margin: 12px 0;
    padding: 12px 16px;
    border: none;
    background-color: #1e1e1e;
    border-radius: 10px;
    color: #f5f5f5;
    animation: fadeIn 0.4s ease-in;
}
.chat-bubble.user {
    background-color: #2a5a8a;
    margin-left: 20%;
}
.chat-bubble.assistant {
    background-color: #2d3d4d;
    margin-right: 10%;
}
.chat-author {
    font-weight: 600;
    font-size: 12px;
    color: #aaa;
    margin-bottom: 4px;
}
.history-item {
    padding: 10px 12px;
    background-color: #1a1a1a;
    border-left: 3px solid #2a5a8a;
    margin: 6px 0;
    border-radius: 4px;
    font-size: 12px;
    cursor: pointer;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.history-item:hover {
    background-color: #222;
    border-left-color: #3a7aaa;
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
if "schema_dict" not in st.session_state:
    st.session_state.schema_dict = {}
if "fk_dict" not in st.session_state:
    st.session_state.fk_dict = {}
if "approval_mode" not in st.session_state:
    st.session_state.approval_mode = False
if "action_taken" not in st.session_state:
    st.session_state.action_taken = None

# Sidebar for DB upload
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
            
            # Check if DB is too big (more than 20 tables)
            if len(tables) > 20:
                return {"_error": "DB has too many tables"}, {}
            
            for (table_name,) in tables:
                columns = st.session_state.cursor.execute(f"PRAGMA table_info('{table_name}');").fetchall()
                schema[table_name] = [col[1] for col in columns]
                fks = st.session_state.cursor.execute(f"PRAGMA foreign_key_list('{table_name}');").fetchall()
                foreign_keys[table_name] = [(fk[3], fk[2]) for fk in fks]
            return schema, foreign_keys

        st.session_state.schema_dict, st.session_state.fk_dict = get_schema()
        
        if "_error" not in st.session_state.schema_dict:
            def format_schema(schema, fks):
                return "\n".join([
                    f"**{t}**: {', '.join(cols)}" + 
                    (f"\nForeign Keys: {', '.join([f'{col} → {ref}' for col, ref in fks[t]])}" if fks[t] else "") 
                    for t, cols in schema.items()
                ])

            st.header("📋 DB Schema")
            st.markdown(format_schema(st.session_state.schema_dict, st.session_state.fk_dict), unsafe_allow_html=True)
            
            # DB Records Preview
            st.header("📊 Records Preview")
            with st.expander("View Sample Data (Max 10 records per table)"):
                for table in list(st.session_state.schema_dict.keys())[:5]:
                    try:
                        df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 10", st.session_state.conn)
                        if len(df) > 0:
                            st.subheader(f"📋 {table} ({len(df)} records)")
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.info(f"{table}: No records")
                    except Exception as e:
                        st.warning(f"{table}: Error loading - {str(e)[:50]}")
                
                if len(st.session_state.schema_dict) > 5:
                    st.warning(f"⚠️ Only showing first 5 tables. Database has {len(st.session_state.schema_dict)} tables.")
        else:
            st.warning("🚨 DB is too big to be displayed here (>20 tables)")
    else:
        st.info("Upload a SQLite file above to get started.")

# Utilities
def clean_sql_output(text):
    return re.sub(r"```(?:sql)?\s*(.*?)```", r"\1", text, flags=re.DOTALL | re.IGNORECASE).strip()

def generate_sql(prompt):
    with st.spinner("🤖 Generating SQL..."):
        try:
            resp = co.chat(model=COHERE_MODEL, message=prompt, temperature=0.3, max_tokens=200)
            return clean_sql_output(resp.text)
        except Exception as e:
            st.error(f"Error generating SQL: {str(e)}")
            raise

def explain_sql(sql):
    prompt = f"""Explain briefly what this SQLite query does in simple terms:\n\n{sql}\n\nExplanation:"""
    resp = co.chat(model=COHERE_MODEL, message=prompt, temperature=0.2, max_tokens=50)
    return resp.text.strip()

def render_chat():
    for entry in st.session_state.chat_history:
        ts = entry["timestamp"].strftime('%H:%M:%S')
        if entry["type"] in ["user", "assistant", "result"]:
            role = "You" if entry["type"] == "user" else "🤖 Assistant"
            bubble_class = "user" if entry["type"] == "user" else "assistant"
            content_html = entry["content"].replace("\n", "<br>").replace("```sql", "<b>SQL:</b><br/>").replace("```", "")
            st.markdown(f'''<div class="chat-bubble {bubble_class}">
                <div class="chat-author">{role} • {ts}</div>
                {content_html}
            </div>''', unsafe_allow_html=True)
        else:
            st.markdown(f"*{entry['content']}*")

# Main Layout: Left content area + Right sidebar
col_main, col_history = st.columns([3, 1])

with col_main:
    st.title("💬 Chat with Your Database")
    
    query = st.text_area("Ask your database...", height=100, key="chat_input", placeholder="e.g., Show all users from 2024")

    if st.button("🚀 Send", use_container_width=True):
        if not query.strip():
            st.error("Please enter your query.")
        elif st.session_state.db_path is None:
            st.error("Please upload a SQLite DB first.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": query, "timestamp": datetime.now(), "type": "user"})

            # Generate SQL
            schema_str = "\n".join([
                f"**{t}**: {', '.join(cols)}" 
                for t, cols in st.session_state.schema_dict.items()
            ]) if st.session_state.schema_dict and "_error" not in st.session_state.schema_dict else "No tables available"
            
            prompt = f"""You are an expert SQLite assistant. Generate valid SQLite SQL based on user request and DB schema.

User Request:
{query}

Schema:
{schema_str}

Only SQL, no explanation:"""
            
            with st.spinner("💬 Generating SQL..."):
                time.sleep(0.5)
                sql = generate_sql(prompt)

            audit_logger.log_query("PENDING", sql, "Generated by AI, awaiting user approval")

            st.session_state.pending_sql = sql
            st.session_state.improve_rounds = 0
            st.session_state.chat_history.append({"role": "assistant", "content": sql, "timestamp": datetime.now(), "type": "assistant"})

            st.session_state.last_query_df = None
            st.session_state.last_sql_query = sql

            # Human-in-loop controls
            st.session_state.approval_mode = True
            st.markdown("### 📝 Review Generated SQL")
            st.code(st.session_state.pending_sql, language="sql")
            
            cols_action = st.columns(3, gap="medium")
            
            if cols_action[0].button("✅ Execute", use_container_width=True, key=f"exec_{datetime.now().timestamp()}"):
                st.session_state.action_taken = "execute"
                st.rerun()
            
            if cols_action[1].button("🔄 Improve", use_container_width=True, key=f"impr_{datetime.now().timestamp()}"):
                st.session_state.action_taken = "improve"
                st.rerun()
            
            if cols_action[2].button("✏️ Edit", use_container_width=True, key=f"edit_{datetime.now().timestamp()}"):
                st.session_state.action_taken = "edit"
                st.rerun()
            
            # Handle actions after button click
            if st.session_state.action_taken == "improve":
                if st.session_state.improve_rounds >= 3:
                    st.error("❌ Max improve attempts (3) reached.")
                    audit_logger.log_query("REJECTED", sql, "Max improve attempts exceeded")
                    st.session_state.action_taken = None
                else:
                    st.info(f"🔁 Improving query (attempt {st.session_state.improve_rounds + 1}/3)...")
                    improve_prompt = f"""Improve the SQLite query to better match the user's request:
{query}

Current SQL:
{st.session_state.pending_sql}

Return ONLY the improved SQL query, nothing else:"""
                    with st.spinner("✨ AI is improving your query..."):
                        new_sql = generate_sql(improve_prompt)
                    
                    st.session_state.pending_sql = new_sql
                    st.session_state.improve_rounds += 1
                    st.session_state.chat_history.append({"role": "assistant", "content": f"Improved SQL (v{st.session_state.improve_rounds}):\n{new_sql}", "timestamp": datetime.now(), "type": "assistant"})
                    audit_logger.log_query("PENDING", new_sql, f"Improved (round {st.session_state.improve_rounds})")
                    st.session_state.action_taken = None
                    st.rerun()
            
            elif st.session_state.action_taken == "edit":
                st.markdown("### ✏️ Edit SQL")
                edited_sql = st.text_area("Modify the SQL query:", value=st.session_state.pending_sql, height=120, key="edited_sql_area")
                
                if st.button("✅ Execute Edited SQL", use_container_width=True, key="exec_edited_now"):
                    st.session_state.pending_sql = edited_sql  # Save edited SQL
                    st.session_state.action_taken = "execute"
                    st.rerun()
            
            elif st.session_state.action_taken == "execute":
                sql_to_run = st.session_state.pending_sql
                audit_logger.log_approval(sql_to_run, approved=True)
                
                with st.spinner("⏳ Executing query..."):
                    try:
                        # Fetch ALL records from the query
                        df = pd.read_sql_query(sql_to_run, st.session_state.conn)
                        st.session_state.last_query_df = df
                        row_count = len(df)
                        res = f"✅ Success! Returned {row_count} rows."
                        st.success(res)
                        audit_logger.log_query("EXECUTED", sql_to_run, f"Success: {row_count} rows")
                        
                    except Exception as e:
                        res = f"❌ Query Error: {str(e)}"
                        st.error(res)
                        audit_logger.log_query("ERROR", sql_to_run, str(e))
                
                # Save result to history and switch out of approval mode so results render
                st.session_state.chat_history.append({"role": "assistant", "content": res, "timestamp": datetime.now(), "type": "result"})
                st.session_state.action_taken = None
                st.session_state.approval_mode = False
                # Rerun so the UI refreshes and shows the results/table
                st.rerun()

    # Render chat
    if not st.session_state.approval_mode:
        st.markdown("---")
        render_chat()
    
    # Display table results after chat (always show when DataFrame exists)
    if st.session_state.last_query_df is not None:
        st.markdown("---")
        st.subheader("📊 Query Results")
        if isinstance(st.session_state.last_query_df, pd.DataFrame) and len(st.session_state.last_query_df) > 0:
            st.table(st.session_state.last_query_df)
        else:
            st.info("No rows returned by the query.")
        
        # Explain button
        if st.session_state.last_sql_query:
            if st.button("💡 Explain Last SQL"):
                explanation = explain_sql(st.session_state.last_sql_query)
                st.session_state.chat_history.append({"role": "assistant", "content": explanation, "timestamp": datetime.now(), "type": "assistant"})
                st.rerun()

# Right sidebar: Chat History
with col_history:
    st.markdown("### 💬 Chat History")
    
    if len(st.session_state.chat_history) == 0:
        st.info("No conversations yet.\nAsk a question to get started!")
    else:
        # Show last 10 user questions as expandable dropdowns (most recent first)
        user_indices = [i for i, e in enumerate(st.session_state.chat_history) if e.get("type") == "user"]
        for idx in reversed(user_indices[-10:]):
            question_entry = st.session_state.chat_history[idx]
            question_text = question_entry["content"] if len(question_entry["content"]) <= 80 else question_entry["content"][:77] + "..."

            with st.expander(question_text):
                st.markdown(f"**💬 Question** ({question_entry['timestamp'].strftime('%H:%M:%S')})")
                st.markdown(question_entry["content"])

                # Find the next result (if any)
                found_answer = False
                for j in range(idx + 1, len(st.session_state.chat_history)):
                    subsequent_entry = st.session_state.chat_history[j]
                    if subsequent_entry.get("type") in ("result", "assistant"):
                        st.markdown(f"**✅ Result** ({subsequent_entry['timestamp'].strftime('%H:%M:%S')})")
                        st.markdown(subsequent_entry["content"])
                        found_answer = True
                        break

                if not found_answer:
                    st.info("Pending execution...")
    
    st.markdown("---")
    
    # Clear history button
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_query_df = None
        st.rerun()
