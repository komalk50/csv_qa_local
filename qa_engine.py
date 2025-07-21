import pandas as pd
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.schema import Document


class CSV_QA_Engine:
    def __init__(self, model_name="mistral", embedding_model="all-MiniLM-L6-v2"):
        self.llm = Ollama(model=model_name)
        self.embedder = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = None
        self.df = None  # Save original DataFrame

    def load_and_prepare(self, df: pd.DataFrame):
        self.df = df
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        full_text = df.to_csv(index=False)
        docs = text_splitter.split_documents([Document(page_content=full_text)])
        self.vectorstore = FAISS.from_documents(docs, self.embedder)

    def answer(self, query):
        # First, check for structured query
        if self.is_structured_query(query):
            try:
                return self.execute_structured_query(query)
            except Exception as e:
                return f"⚠️ Failed to run structured query: {e}"

        # Else: fallback to LLM
        retriever = self.vectorstore.as_retriever()
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=False
        )
        result = chain({"query": query})
        return result["result"]

    def is_structured_query(self, query):
        """Detect simple structured data queries using keywords."""
        keywords = ["top", "highest", "lowest", "filter", "where", "greater than", "less than"]
        return any(k in query.lower() for k in keywords)

    def execute_structured_query(self, query):
        df = self.df.copy()
        query_lower = query.lower()

    # --- Top N by column ---
        if "top" in query_lower and ("salary" in query_lower or "earning" in query_lower):
            n_match = re.search(r"top\s+(\d+)", query_lower)
            n = int(n_match.group(1)) if n_match else 5

            # Try finding salary-like column
            col = self._find_column("salary") or self._find_column("earnings")
            if col:
                return df.sort_values(by=col, ascending=False).head(n)

        # --- Filter by column = value ---
        elif "filter" in query_lower or "where" in query_lower:
            # crude handling: "where department is hr"
            match = re.search(r"where\s+(\w+)\s+is\s+(\w+)", query_lower)
            if match:
                col, val = match.group(1), match.group(2)
                col = self._find_column(col)
                return df[df[col].str.lower() == val.lower()]

        # --- Numeric Filters: greater/less than ---
        elif "greater than" in query_lower or "less than" in query_lower:
            match = re.search(r"(\w+)\s+(greater than|less than)\s+(\d+)", query_lower)
            if match:
                col, op, val = match.group(1), match.group(2), int(match.group(3))
                col = self._find_column(col)
                if op == "greater than":
                    return df[df[col] > val]
                else:
                    return df[df[col] < val]

        # --- Combined condition (age + salary) ---
        elif "older than" in query_lower and "salary over" in query_lower:
            age_match = re.search(r"older than\s+(\d+)", query_lower)
            sal_match = re.search(r"salary over\s+(\d+)", query_lower)
            if age_match and sal_match:
                age_val = int(age_match.group(1))
                sal_val = int(sal_match.group(1))
                age_col = self._find_column("age")
                sal_col = self._find_column("salary")
                return df[(df[age_col] > age_val) & (df[sal_col] > sal_val)]

        return "❌ Structured query detected, but not implemented yet."
    
    def _find_column(self, col_name):
        """
        Tries to find the closest matching column from the DataFrame
        using case-insensitive and partial matching.
        """
        col_name = col_name.strip().lower()
        for col in self.df.columns:
            if col_name in col.lower():
                return col
        return None


