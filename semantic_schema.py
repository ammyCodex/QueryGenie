from typing import Dict, List
from config import COHERE_API_KEY

try:
    from langchain.embeddings import CohereEmbeddings
    from langchain.vectorstores import FAISS
except Exception:
    CohereEmbeddings = None
    FAISS = None


def build_faiss_from_schema(schema_dict: Dict[str, List[str]]):
    """Build a FAISS vectorstore from schema_dict where each entry is a table description.

    Returns the FAISS store and the list of table keys in the same order as the store texts.
    """
    if CohereEmbeddings is None or FAISS is None:
        raise ImportError("LangChain or FAISS is not available. Install 'langchain' and 'faiss-cpu'.")

    texts = []
    metadatas = []
    for table, cols in schema_dict.items():
        desc = f"{table}: {', '.join(cols)}"
        texts.append(desc)
        metadatas.append({"table": table})

    emb = CohereEmbeddings(cohere_api_key=COHERE_API_KEY, model="embed-english-v2.0")
    store = FAISS.from_texts(texts, emb, metadatas=metadatas)
    return store


def query_relevant_tables(store, query: str, k: int = 5) -> List[str]:
    """Return up to `k` table names most relevant to `query` using the FAISS store."""
    if FAISS is None:
        raise ImportError("FAISS or LangChain not installed")

    docs = store.similarity_search(query, k=k)
    tables = []
    for d in docs:
        tbl = d.metadata.get("table") if hasattr(d, "metadata") else None
        if tbl and tbl not in tables:
            tables.append(tbl)
    return tables
