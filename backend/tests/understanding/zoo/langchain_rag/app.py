"""PDF chat: Streamlit + LangChain + Chroma."""

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROMPT = ChatPromptTemplate.from_template(
    "Answer the question using the context.\n\nContext: {context}\n\nQuestion: {question}"
)


def build_vectorstore(path: str):
    docs = PyPDFLoader(path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
    chunks = splitter.split_documents(docs)
    return Chroma.from_documents(chunks, OpenAIEmbeddings())


def answer(question: str, pdf_path: str) -> str:
    store = build_vectorstore(pdf_path)                 # rebuilt for every question
    retriever = store.as_retriever(search_kwargs={"k": 10})
    docs = retriever.invoke(question)
    llm = ChatOpenAI(model="gpt-4o", temperature=0.9)
    chain = PROMPT | llm
    return chain.invoke({"context": "\n\n".join(d.page_content for d in docs), "question": question}).content


st.title("Chat with your PDF")
if "history" not in st.session_state:
    st.session_state.history = []
question = st.chat_input("Ask something")
if question:
    reply = answer(question, "docs/handbook.pdf")
    st.session_state.history.append((question, reply))
    for q, a in st.session_state.history:
        st.write(q, a)
