import pandas as pd
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

    def load_and_prepare(self, df: pd.DataFrame):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        full_text = df.to_csv(index=False)
        docs = text_splitter.split_documents([Document(page_content=full_text)])
        self.vectorstore = FAISS.from_documents(docs, self.embedder)

    def answer(self, query):
        retriever = self.vectorstore.as_retriever()
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=False
        )
        result = chain({"query": query})
        return result["result"]
