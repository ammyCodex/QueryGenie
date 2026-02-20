from typing import Dict
from config import COHERE_API_KEY, COHERE_MODEL
import re

try:
    from langchain.llms import Cohere as LCohere
    from langchain import LLMChain
    from langchain.prompts import PromptTemplate
except Exception:
    LCohere = None
    LLMChain = None
    PromptTemplate = None


def clean_sql_output(text: str) -> str:
    return re.sub(r"```(?:sql)?\s*(.*?)```", r"\1", text, flags=re.DOTALL | re.IGNORECASE).strip()


def generate_sql_with_langchain(schema_str: str, user_request: str) -> str:
    """Generate SQL using LangChain + Cohere LLM from a schema string and user request.

    IMPORTANT: This helper does NOT require a live SQL connection or SQLAlchemy/SQLDatabase.
    It only consumes a textual `schema_str` (table names and columns) and the user's
    natural language request to produce SQL. This keeps generation decoupled from
    database connections and allows offline generation.

    If LangChain or its Cohere wrapper isn't installed, this raises ImportError.
    """
    if LCohere is None:
        raise ImportError("LangChain or its Cohere LLM wrapper is not available. Install 'langchain' and try again.")

    # Strict prompt: explicitly forbid use of external/world data and require the model
    # to only reference tables/columns present in the provided schema string.
    # If the request cannot be satisfied using schema alone, the model MUST respond
    # with the exact token: NO_VALID_SQL
    prompt_template = (
        "You are an expert SQLite assistant. ONLY use the database schema provided below. "
        "Do NOT use any external knowledge or assume any tables/columns that are not listed.\n\n"
        "BE STRICT: If the user request cannot be satisfied with the given schema, reply with EXACT TEXT: NO_VALID_SQL (no SQL, no explanation).\n\n"
        "Respond with ONLY the SQL query and NOTHING ELSE when possible. Do not wrap in markdown fences.\n\n"
        "Schema (tables and columns):\n{schema}\n\nUser Request:\n{request}\n\nSQL:" 
    )

    template = PromptTemplate(template=prompt_template, input_variables=["schema", "request"])
    # Use deterministic settings to reduce hallucination
    llm = LCohere(api_key=COHERE_API_KEY, model=COHERE_MODEL, temperature=0.0, max_tokens=400)
    chain = LLMChain(llm=llm, prompt=template)
    resp = chain.run({"schema": schema_str, "request": user_request})
    return clean_sql_output(resp)
