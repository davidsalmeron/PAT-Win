import os
import tempfile
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyMuPDFLoader, UnstructuredWordDocumentLoader
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

st.title("Asistente IA local con documentación")

uploaded_file = st.file_uploader("Sube un documento (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])

if uploaded_file:
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
        st.error("Formato no soportado.")
        st.stop()

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)

    retriever = db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0), retriever=retriever)

    query = st.text_input("Haz una pregunta sobre el documento:")
    if query:
        answer = qa_chain.run(query)
        st.write("Respuesta:", answer)

    os.remove(tmp_path)
