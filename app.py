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
from semantic_schema import build_faiss_from_schema, query_relevant_tables

# Validate configuration
try:
    validate_config()
    print("[STARTUP] Config validated successfully")
except ValueError as e:
    print(f"[ERROR] Config validation failed: {e}")
    st.error(f"Configuration error: {e}")
    st.stop()

# Page setup
st.set_page_config(page_title="🪄 QueryGenie", layout="wide")

# Custom CSS – UI only, no backend changes
custom_css = """
<style>
/* Base */
body, .block-container {
    background: linear-gradient(160deg, #0f0f12 0%, #1a1a22 50%, #12121a 100%);
    color: #e8e8ed;
    font-family: 'Segoe UI', 'SF Pro Text', system-ui, sans-serif;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #16161c 0%, #1c1c24 100%);
    color: #e0e0e0;
}
[data-testid="stSidebar"] .stMarkdown { color: #c8c8d0; }

/* Main title */
h1 {
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #e8e8ed 0%, #a78bfa 50%, #67d391 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Chat cards */
.chat-card {
    background: linear-gradient(145deg, #1c1c24 0%, #18181f 100%);
    border-radius: 14px;
    padding: 18px 20px;
    margin: 14px 0;
    border-left: 4px solid #7c3aed;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    animation: fadeIn 0.35s ease-out;
}
.chat-card:hover {
    box-shadow: 0 6px 24px rgba(124, 58, 237, 0.12);
}
.chat-card-label {
    font-weight: 600;
    font-size: 13px;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 8px;
    letter-spacing: 0.02em;
}
.chat-card-label.question { color: #a78bfa; }
.chat-card-label.result { color: #68d391; }
.chat-card-ts {
    font-size: 11px;
    color: #6b7280;
    margin-bottom: 8px;
    font-variant-numeric: tabular-nums;
}
.chat-card-content {
    color: #f0f0f5;
    font-size: 14px;
    line-height: 1.6;
    margin-top: 4px;
}

/* In-card SQL block */
.chat-sql-block {
    background: #16161e !important;
    padding: 14px 16px !important;
    border-radius: 10px !important;
    border: 1px solid #2d2d38 !important;
    overflow-x: auto !important;
    font-size: 13px !important;
    line-height: 1.5 !important;
    color: #a5b4fc !important;
    margin: 6px 0 0 0 !important;
}
.chat-card pre { border-radius: 10px; border: 1px solid #2d2d38; }

/* Caption under title */
.stCaptionContainer { color: #6b7280 !important; }

/* Section headers */
h2, h3 {
    color: #e0e0e8 !important;
    font-weight: 600;
    letter-spacing: -0.01em;
}
hr {
    border-color: #2d2d38 !important;
    opacity: 0.8;
}

/* Buttons */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}

/* Text area */
[data-testid="stTextArea"] textarea {
    border-radius: 12px !important;
    border: 1px solid #2d2d38 !important;
    background: #1a1a22 !important;
}

/* Expanders (sidebar) */
.streamlit-expanderHeader {
    background: #1c1c24 !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] {
    background: rgba(28,28,36,0.6) !important;
    border-radius: 10px !important;
    border: 1px solid #2d2d38 !important;
}

/* Table / dataframe container */
[data-testid="stTable"], .stTable {
    border-radius: 12px !important;
    overflow: hidden;
}

/* Info / success messages */
.stAlert {
    border-radius: 10px !important;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(8px); }
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
    print(f"[STARTUP] Cohere client initialized (model={COHERE_MODEL})")
except Exception as e:
    print(f"[ERROR] Cohere init failed: {e}")
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
if "last_query_truncated" not in st.session_state:
    st.session_state.last_query_truncated = False
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
if "needs_execute" not in st.session_state:
    st.session_state.needs_execute = False
if "last_schema_str" not in st.session_state:
    st.session_state.last_schema_str = ""
if "last_user_query" not in st.session_state:
    st.session_state.last_user_query = ""

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
        print(f"[DB UPLOADED] Saved to {temp_file.name}")
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
        print(f"[DB SCHEMA] Loaded {len(st.session_state.schema_dict)} tables")
        for table in list(st.session_state.schema_dict.keys())[:5]:
            print(f"  - {table}: {len(st.session_state.schema_dict[table])} columns")
        
        if "_error" not in st.session_state.schema_dict:
            SIDEBAR_MAX_TABLES = 5  # Only show first N tables in sidebar (full schema still used for LLM)
            schema_keys = list(st.session_state.schema_dict.keys())
            schema_for_display = {t: st.session_state.schema_dict[t] for t in schema_keys[:SIDEBAR_MAX_TABLES]}

            def format_schema(schema, fks):
                return "\n".join([
                    f"**{t}**: {', '.join(cols)}" +
                    (f"\nForeign Keys: {', '.join([f'{col} → {ref}' for col, ref in fks[t]])}" if fks.get(t) else "")
                    for t, cols in schema.items()
                ])

            st.header("📋 DB Schema")
            st.markdown(format_schema(schema_for_display, st.session_state.fk_dict), unsafe_allow_html=True)
            if len(schema_keys) > SIDEBAR_MAX_TABLES:
                st.caption(f"Showing first {SIDEBAR_MAX_TABLES} of {len(schema_keys)} tables.")

            # DB Records Preview (same limit)
            st.header("📊 Records Preview")
            with st.expander("View Sample Data (Max 10 records per table)"):
                for table in schema_keys[:SIDEBAR_MAX_TABLES]:
                    try:
                        df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 10", st.session_state.conn)
                        if len(df) > 0:
                            st.subheader(f"📋 {table} ({len(df)} records)")
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.info(f"{table}: No records")
                    except Exception as e:
                        st.warning(f"{table}: Error loading - {str(e)[:50]}")
                if len(schema_keys) > SIDEBAR_MAX_TABLES:
                    st.caption(f"Showing first {SIDEBAR_MAX_TABLES} of {len(schema_keys)} tables.")
        else:
            st.warning("🚨 DB is too big to be displayed here (>20 tables)")
    else:
        st.info("Upload a SQLite file above to get started.")

# Utilities
def clean_sql_output(text):
    return re.sub(r"```(?:sql)?\s*(.*?)```", r"\1", text, flags=re.DOTALL | re.IGNORECASE).strip()


def _clean_sql_artifacts(sql: str) -> str:
    """Remove [object Object] and fix leftover commas so SQL is executable."""
    if not sql or not sql.strip():
        return sql
    s = sql.strip()
    # Remove all [object Object] (any casing) and optional surrounding commas/spaces
    s = re.sub(r",?\s*\[object\s+object\]\s*,?", ",", s, flags=re.IGNORECASE)
    # Collapse multiple commas to single comma
    s = re.sub(r",\s*,+", ",", s)
    # Remove leading comma after SELECT (e.g. "SELECT ," -> "SELECT ")
    s = re.sub(r"\bSELECT\s+,", "SELECT ", s, flags=re.IGNORECASE)
    # Remove comma before keywords that start clauses (e.g. ", FROM" or ",FROM" -> " FROM")
    for kw in ["FROM", "LEFT", "RIGHT", "INNER", "JOIN", "WHERE", "GROUP", "ORDER", "HAVING", "LIMIT", "ON", "AND", "OR"]:
        s = re.sub(r",\s*" + kw + r"\b", " " + kw, s, flags=re.IGNORECASE)
    # Clean ", ," and ",  " -> ", "
    s = re.sub(r",\s*,", ", ", s)
    s = re.sub(r",\s{2,}", ", ", s)
    # Normalize whitespace
    s = re.sub(r"\n\s*\n", "\n", s)
    s = re.sub(r"  +", " ", s).strip()
    return s


def extract_sql_from_llm_response(text: str) -> str:
    """Extract a single SQL statement from LLM output that may contain explanation, markdown, or multiple queries."""
    if not text or not text.strip():
        return text
    raw = text.strip()
    # Remove [object Object] and comma artifacts before parsing
    raw = re.sub(r",?\s*\[object\s+object\]\s*,?", " ", raw, flags=re.IGNORECASE)
    raw = re.sub(r",\s*,\s*", ", ", raw)

    def return_cleaned(statement: str) -> str:
        out = _clean_sql_artifacts(statement)
        if not out.endswith(";") and out:
            out = out + ";"
        return out

    # 1) Try ```sql ... ``` or ``` ... ``` blocks first
    for pattern in [r"```sql\s*(.*?)```", r"```\s*(.*?)```"]:
        matches = re.findall(pattern, raw, flags=re.DOTALL | re.IGNORECASE)
        for m in matches:
            block = m.strip()
            if re.search(r"\b(select|with)\b", block, re.IGNORECASE):
                first_stmt = block.split(";")[0].strip()
                if first_stmt and len(first_stmt) > 10:
                    return return_cleaned(first_stmt)

    # 2) No code block: find line that starts with SELECT or WITH, then collect until ;
    lines = raw.split("\n")
    start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^\s*(select|with)\s+", stripped, re.IGNORECASE):
            start = i
            break
    if start is not None:
        collected = []
        for i in range(start, len(lines)):
            line = lines[i]
            s = line.strip().lower()
            if s.startswith("###") or (s.startswith("**") and not s.startswith("**`")) or "explanation" in s or "both queries" in s:
                break
            collected.append(line)
            if line.strip().endswith(";"):
                break
        out = "\n".join(collected).strip()
        if out and re.search(r"\b(select|with)\b", out, re.IGNORECASE):
            return return_cleaned(out)

    # 3) Fallback: use existing clean_sql_output then clean artifacts
    return return_cleaned(clean_sql_output(text))


SQL_KEYWORDS = {"select", "from", "where", "join", "on", "and", "or", "group", "by", "order", "limit", "distinct", "as", "count", "sum", "avg", "min", "max", "inner", "left", "right", "full", "outer", "having", "union", "all", "in", "not", "exists", "between", "like", "is", "null", "case", "when", "then", "else", "end", "offset"}


def is_probable_sql(text: str) -> bool:
    t = text.strip().lower()
    return bool(re.search(r"\b(select|insert|update|delete|with)\b", t))


def validate_sql_against_schema(sql_text: str, schema_dict: dict) -> bool:
    """Lightweight validation: ensure SQL references at least one allowed table/column."""
    if not sql_text:
        return False
    if sql_text.strip().upper() == "NO_VALID_SQL":
        return False
    if not is_probable_sql(sql_text):
        return False

    allowed = set()
    for t, cols in (schema_dict or {}).items():
        allowed.add(t.lower())
        for c in cols:
            allowed.add(c.lower())

    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", sql_text)
    tokens = [tok.lower() for tok in tokens]

    # Require at least one allowed identifier (table or column name)
    allowed_matches = sum(1 for tok in tokens if tok in allowed)
    if allowed_matches == 0:
        return False

    return True


def select_relevant_schema(schema_dict, user_request, max_tables=5):
    """Select up to `max_tables` from `schema_dict` that are most relevant to `user_request`.

    This is a lightweight heuristic to avoid sending the entire schema to the LLM
    while keeping relevance high. It matches tokens from the user request against
    table names and column names and ranks tables by match count.
    """
    if not schema_dict:
        return "No tables available"

    tokens = re.findall(r"\w+", user_request.lower())
    token_set = set(tokens)

    scores = []
    for table, cols in schema_dict.items():
        score = 0
        tname = table.lower()
        # match table name
        for tk in token_set:
            if tk in tname:
                score += 3
        # match columns
        for c in cols:
            cname = c.lower()
            for tk in token_set:
                if tk in cname:
                    score += 1

        scores.append((score, table))

    # sort by score desc
    scores.sort(reverse=True)
    selected = [t for s, t in scores if s > 0][:max_tables]

    # fallback: if nothing matched, include the first `max_tables` tables
    if not selected:
        selected = list(schema_dict.keys())[:max_tables]

    # build schema string from selected tables
    parts = []
    for t in selected:
        cols = schema_dict.get(t, [])
        parts.append(f"**{t}**: {', '.join(cols)}")
    return "\n".join(parts)

def generate_sql(prompt):
    # Try to use LangChain helper if available (preferred)
    try:
        from langchain_sql import generate_sql_with_langchain
        raw = generate_sql_with_langchain(prompt.get("schema", ""), prompt.get("request", ""))
        return extract_sql_from_llm_response(raw)
    except Exception:
        # Fallback: use direct Cohere client with the provided raw prompt string (no nested spinner)
        try:
            resp = co.chat(model=COHERE_MODEL, message=prompt if isinstance(prompt, str) else prompt.get("request", ""), temperature=0.3, max_tokens=400)
            return extract_sql_from_llm_response(resp.text)
        except Exception as e:
            st.error(f"Error generating SQL: {str(e)}")
            raise

def explain_sql(sql):
    prompt = f"""Explain briefly what this SQLite query does in simple terms:\n\n{sql}\n\nExplanation:"""
    resp = co.chat(model=COHERE_MODEL, message=prompt, temperature=0.2, max_tokens=50)
    return resp.text.strip()

def _escape(s):
    return s.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")

def render_chat():
    """Render chat in card style: Question (purple icon) + Result (green check) with (timestamp)."""
    i = 0
    while i < len(st.session_state.chat_history):
        entry = st.session_state.chat_history[i]
        if entry["type"] != "user":
            i += 1
            continue
        ts_q = entry["timestamp"].strftime("%H:%M:%S")
        user_content = _escape(entry["content"])
        results = []
        j = i + 1
        while j < len(st.session_state.chat_history) and st.session_state.chat_history[j]["type"] in ("assistant", "result"):
            results.append(st.session_state.chat_history[j])
            j += 1
        i = j

        parts = [
            '<div class="chat-card">',
            '<div class="chat-card-label question">💬 Question</div>',
            f'<div class="chat-card-ts">({ts_q})</div>',
            f'<div class="chat-card-content">{user_content}</div>',
        ]
        for r in results:
            ts_r = r["timestamp"].strftime("%H:%M:%S")
            content = r["content"]
            parts.append('<div class="chat-card-label result">✅ Result</div>')
            parts.append(f'<div class="chat-card-ts">({ts_r})</div>')
            if "SELECT" in content.upper() or "```" in content or content.strip().endswith(";"):
                sql = content.strip().replace("```sql", "").replace("```", "").strip()
                parts.append(f'<pre class="chat-sql-block"><code>{_escape(sql)}</code></pre>')
            else:
                parts.append(f'<div class="chat-card-content">{_escape(content)}</div>')
        parts.append("</div>")
        st.markdown("\n".join(parts), unsafe_allow_html=True)

# Main Layout: Left content area + Right sidebar
col_main, col_history = st.columns([3, 1])

with col_main:
    st.title("💬 Chat with Your Database")
    st.caption("Ask in plain English → get SQL → approve & see results.")
    query = st.text_area("Ask your database...", height=100, key="chat_input", placeholder="e.g., List all actors • Show top 10 orders by date • Count users by role")

    # Run execution every run when user clicked Execute (must be outside Send block)
    # Uses same LLM-generated query from session state (pending_sql) to fetch from DB
    if st.session_state.get("needs_execute"):
        st.session_state.needs_execute = False
        sql_to_run = st.session_state.pending_sql  # same query LLM generated, saved earlier
        if not sql_to_run or sql_to_run == "NO_VALID_SQL":
            st.session_state.approval_mode = False
            st.session_state.chat_history.append({"role": "assistant", "content": "❌ Could not generate a valid SQL query.", "timestamp": datetime.now(), "type": "result"})
            audit_logger.log_query("REJECTED", str(sql_to_run or ""), "Generator returned NO_VALID_SQL")
        elif st.session_state.conn:
            try:
                df = pd.read_sql_query(sql_to_run, st.session_state.conn)
                truncated = len(df) > 20
                display_df = df.head(20) if truncated else df
                st.session_state.last_query_df = display_df.copy()
                st.session_state.last_query_truncated = truncated
                st.session_state.last_sql_query = sql_to_run
                st.session_state.approval_mode = False
                row_count_msg = f"{len(df)}" if not truncated else f"{len(df)} (showing first 20)"
                audit_logger.log_approval(sql_to_run, approved=True)
                audit_logger.log_query("EXECUTED", sql_to_run, f"Success: {row_count_msg} rows")
                st.session_state.chat_history.append({"role": "assistant", "content": f"✅ Success! Returned {row_count_msg} rows.", "timestamp": datetime.now(), "type": "result"})
            except Exception as e:
                st.session_state.approval_mode = False
                st.session_state.chat_history.append({"role": "assistant", "content": f"❌ Query Error: {str(e)}", "timestamp": datetime.now(), "type": "result"})
                audit_logger.log_query("ERROR", sql_to_run, str(e))
        # No rerun—script continues and table is shown below in same run

    # Show Review SQL and Execute/Improve when in approval mode (runs every run so Execute click is handled)
    if st.session_state.approval_mode and st.session_state.pending_sql:
        st.markdown("### 📝 Review Generated SQL")
        st.caption("Approve and run, or ask the AI to improve the query.")
        st.code(st.session_state.pending_sql, language="sql")
        cols_action = st.columns(2, gap="medium")
        if cols_action[0].button("✅ Execute", use_container_width=True, key="btn_execute"):
            st.session_state.needs_execute = True
            st.rerun()
        if cols_action[1].button("🔄 Improve", use_container_width=True, key="btn_improve"):
            if st.session_state.improve_rounds >= 3:
                st.error("❌ Max improve attempts (3) reached.")
                audit_logger.log_query("REJECTED", st.session_state.pending_sql, "Max improve attempts exceeded")
            else:
                schema_str = st.session_state.get("last_schema_str") or select_relevant_schema(st.session_state.schema_dict, st.session_state.get("last_user_query", query), max_tables=6)
                improve_prompt = f"""Improve the SQLite query to better match the user's request:
{st.session_state.get("last_user_query", query)}

Current SQL:
{st.session_state.pending_sql}

Return ONLY the improved SQL query, nothing else:"""
                with st.spinner("✨ AI is improving your query..."):
                    try:
                        new_sql = generate_sql({"schema": schema_str, "request": improve_prompt})
                    except Exception:
                        new_sql = generate_sql(improve_prompt)
                    if not validate_sql_against_schema(new_sql, st.session_state.schema_dict):
                        new_sql = "NO_VALID_SQL"
                st.session_state.pending_sql = new_sql
                st.session_state.improve_rounds += 1
                st.session_state.chat_history.append({"role": "assistant", "content": f"Improved SQL (v{st.session_state.improve_rounds}):\n{new_sql}", "timestamp": datetime.now(), "type": "assistant"})
                audit_logger.log_query("PENDING", new_sql, f"Improved (round {st.session_state.improve_rounds})")
                st.rerun()

    if st.button("🚀 Send", use_container_width=True):
        if not query.strip():
            st.error("Please enter your query.")
        elif st.session_state.db_path is None:
            st.error("Please upload a SQLite DB first.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": query, "timestamp": datetime.now(), "type": "user"})
            print(f"\n[USER QUERY] {query}")

            # Generate SQL - select relevant subset of schema to avoid token limits
            # Semantic selection: try to use FAISS-based semantic retrieval of relevant tables
            try:
                schema_hash = "|".join([f"{t}:{','.join(v)}" for t, v in sorted(st.session_state.schema_dict.items())])
                if ("faiss_schema_hash" not in st.session_state) or (st.session_state.get("faiss_schema_hash") != schema_hash):
                    # build and cache
                    try:
                        st.session_state.faiss_store = build_faiss_from_schema(st.session_state.schema_dict)
                        st.session_state.faiss_schema_hash = schema_hash
                    except Exception:
                        st.session_state.faiss_store = None
                        st.session_state.faiss_schema_hash = None

                if st.session_state.get("faiss_store") is not None:
                    tables = query_relevant_tables(st.session_state.faiss_store, query, k=6)
                    if tables:
                        # build schema_str from selected tables
                        parts = [f"**{t}**: {', '.join(st.session_state.schema_dict[t])}" for t in tables]
                        schema_str = "\n".join(parts)
                    else:
                        schema_str = select_relevant_schema(st.session_state.schema_dict, query, max_tables=6)
                else:
                    schema_str = select_relevant_schema(st.session_state.schema_dict, query, max_tables=6)
            except Exception:
                # fallback to heuristic selection
                schema_str = select_relevant_schema(st.session_state.schema_dict, query, max_tables=6)
            
            # Print schema selection after it's defined
            print(f"[SCHEMA SELECTION] Selected {len(schema_str.split('**')) - 1} tables")

            prompt = f"""You are an expert SQLite assistant. Generate valid SQLite SQL based on user request and DB schema.

User Request:
{query}

Schema:
{schema_str}

Only SQL, no explanation:"""
            
            with st.spinner("💬  "):
                time.sleep(0.5)
                # Prefer structured call for LangChain helper (schema + request)
                try:
                    sql = generate_sql({"schema": schema_str, "request": query})
                except Exception:
                    # Fallback: call Cohere directly but pass a strict prompt that forbids
                    # using any external/world knowledge and limits references to the schema.
                    strict_prompt = (
                        "ONLY USE THE FOLLOWING SCHEMA. DO NOT INVENT TABLES OR COLUMNS. "
                        "If impossible, respond with EXACT TEXT: NO_VALID_SQL.\n\n"
                        f"Schema:\n{schema_str}\n\nUser Request:\n{query}\n\nSQL:"
                    )
                    sql = generate_sql(strict_prompt)

                # Validate and retry if needed: if validation fails, try a simpler fallback prompt
                try:
                    if not validate_sql_against_schema(sql, st.session_state.schema_dict):
                        # Retry with a simpler, forced prompt
                        print(f"[DEBUG] Initial SQL validation failed. Retrying with simple prompt...")
                        simple_prompt = f"Generate SQLite SQL for: {query}\n\nTables and columns:\n{schema_str}\n\nRespond with ONLY SQL:"
                        try:
                            sql = generate_sql(simple_prompt)
                            # Validate again
                            if not validate_sql_against_schema(sql, st.session_state.schema_dict):
                                sql = "NO_VALID_SQL"
                                print(f"[ERROR] Retry validation failed. Generated SQL: {sql[:100]}")
                            else:
                                print(f"[SUCCESS] Retry generated valid SQL: {sql[:100]}...")
                        except Exception as e:
                            sql = "NO_VALID_SQL"
                            print(f"[ERROR] Retry generation failed: {e}")
                except Exception as e:
                    sql = "NO_VALID_SQL"
                    print(f"[ERROR] Validation error: {e}")

                print(f"[GENERATED SQL] {sql[:200]}...")

            audit_logger.log_query("PENDING", sql, "Generated by AI, awaiting user approval")

            st.session_state.pending_sql = sql
            st.session_state.improve_rounds = 0
            st.session_state.last_schema_str = schema_str
            st.session_state.last_user_query = query
            st.session_state.chat_history.append({"role": "assistant", "content": sql, "timestamp": datetime.now(), "type": "assistant"})

            st.session_state.last_query_df = None
            st.session_state.last_sql_query = sql
            st.session_state.approval_mode = True
            st.rerun()


    # Render chat
    if not st.session_state.approval_mode:
        st.markdown("---")
        render_chat()
    
    # Display table results (always show when DataFrame exists)
    if st.session_state.last_query_df is not None:
        st.markdown("---")
        st.subheader("📊 Query Results")
        df_display = st.session_state.last_query_df
        if isinstance(df_display, pd.DataFrame) and len(df_display) > 0:
            # Ensure display-safe types for st.table (e.g. datetime/bytes can break rendering)
            try:
                display_copy = df_display.astype(str)
            except Exception:
                display_copy = df_display
            st.table(display_copy)
            if st.session_state.get("last_query_truncated"):
                st.info("⚠️ Too many records to be printed here")
        else:
            st.info("No rows returned by the query.")
    else:
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
