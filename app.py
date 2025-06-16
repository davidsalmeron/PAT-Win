import openai
import os
import tempfile
import streamlit as st
import langchain.embeddings.openai as lc_openai
lc_openai.openai = openai

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyMuPDFLoader, UnstructuredWordDocumentLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(all_docs)
    texts = [doc.page_content for doc in docs]

    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-ada-002",
        request_timeout=30
    )
    db = FAISS.from_texts(texts, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Prompt personalizado
    prompt_template = """
Eres un asistente experto que responde preguntas basándote únicamente en el siguiente contexto. 
Si no encuentras la respuesta en el contexto, responde con "No tengo suficiente información para responder eso".

Contexto:
{context}

Pregunta:
{question}
"""
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    query = st.text_input("Hazme una pregunta:")
    if query:
        try:
            result = qa_chain(query)
            st.write("Respuesta:", result["result"])

            with st.expander("Ver contexto utilizado"):
                for i, doc in enumerate(result["source_documents"]):
                    st.markdown(f"**Fragmento {i+1}:**\n{doc.page_content}")
        except Exception as e:
            st.error(f"Error al obtener respuesta: {e}")
