# QueryGenie

🪄 QueryGenie is a Streamlit-based AI-powered SQLite assistant that lets you upload SQLite databases and interact with them using natural language queries. It generates valid SQLite SQL queries using Cohere's API, executes **read-only** queries, and shows results inline in a modern chat UI with dark mode.

---

## Features

- Upload any SQLite `.sqlite` or `.db` file for instant access.
- Automatically detects and displays database schema and foreign keys.
- Natural language to SQL generation powered by Cohere's large language model.
- Executes **SELECT** queries only; detects and blocks any modifying queries.
- Displays all past queries, AI-generated SQL, and result tables in an interactive chat interface.
- Modern dark-themed UI with chat bubbles, timestamps, and sound notifications.
- Option to get a simple explanation of the last executed SQL query.

---

## Demo

Try it live at:  
``` https://ammy-querygenie.streamlit.app/ ```

---

## Usage

1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install streamlit cohere pandas sqlparse
   ```
3. Set your Cohere API key in secrets.toml or via Streamlit secrets manager:
[default]
COHERE_API_KEY = "your-cohere-api-key"

4. Run the app:
streamlit run app.py

5. Upload your SQLite DB file on the sidebar and start chatting!

---

Limitations

- Only supports read-only queries; modification queries are blocked for safety.
- Requires a valid Cohere API key.
- Works best with well-structured relational SQLite databases.

---

License & Copyright

© 2025 Ammy

This project is provided as-is for personal and educational use.
Do not redistribute or use commercially without permission.

---

Contact

For questions or suggestions, please contact Ammy Sharma.

---

