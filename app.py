import os
import tempfile
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyMuPDFLoader, UnstructuredWordDocumentLoader
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

st.title("Asistente PAT-Win")

uploaded_files = st.file_uploader(
    "Sube uno o varios documentos (.txt, .pdf, .docx)",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=True
)

if uploaded_files:
    all_docs = []

    for uploaded_file in uploaded_files:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        if suffix == ".txt":
            loader = TextLoader(tmp_path)
        elif suffix == ".pdf":
            loader = PyMuPDFLoader(tmp_path)
        elif suffix == ".docx":
            loader = UnstructuredWordDocumentLoader(tmp_path)
        else:
            st.error(f"Formato no soportado: {uploaded_file.name}")
            continue

        documents = loader.load()
        all_docs.extend(documents)
        os.remove(tmp_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(all_docs)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    retriever = db.as_retriever()
    llm = OpenAI(temperature=0)  # Usa la clave desde secrets
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    query = st.text_input("Hazme una pregunta:")
    if query:
        answer = qa_chain.run(query)
        st.write("Respuesta:", answer)
