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

    # Pragmatic prompt: be helpful while staying grounded in the schema
    prompt_template = (
        "You are a SQLite SQL expert. Generate a SQL query to answer the user's request.\n\n"
        "CRITICAL: Use ONLY tables and columns from the schema below. Do NOT invent or assume tables.\n"
        "If the request is impossible with the given schema, respond with: NO_VALID_SQL\n\n"
        "Output ONLY valid SQLite SQL. No explanation, no markdown, no prose.\n\n"
        "Schema:\n{schema}\n\nRequest:\n{request}\n\nSQL:"
    )

    template = PromptTemplate(template=prompt_template, input_variables=["schema", "request"])
    # Slightly relaxed temperature for flexibility while staying on-task
    llm = LCohere(api_key=COHERE_API_KEY, model=COHERE_MODEL, temperature=0.1, max_tokens=300)
    chain = LLMChain(llm=llm, prompt=template)
    resp = chain.run({"schema": schema_str, "request": user_request})
    return clean_sql_output(resp)
