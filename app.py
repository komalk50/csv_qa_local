import streamlit as st
import pandas as pd
from qa_engine import CSV_QA_Engine

st.set_page_config(page_title="Offline CSV Q&A App", layout="centered")
st.title("CSV Q&A (Offline with Ollama + LangChain)")

uploaded_file = st.file_uploader("Upload your CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    question = st.text_input("Ask a question about the data")

    if question:
        with st.spinner("Thinking..."):
            qa = CSV_QA_Engine()
            qa.load_and_prepare(df)
            answer = qa.answer(question)

            if isinstance(answer, pd.DataFrame):
                st.dataframe(answer)
            else:
                st.success(answer)
