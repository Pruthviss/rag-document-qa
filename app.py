import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

st.set_page_config(page_title="RAG Document QA")
st.title("📄 RAG Document Question Answering")

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

documents = []

if uploaded_files:

    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader(uploaded_file.name)
        docs = loader.load()
        documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    split_docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(split_docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    pipe = pipeline(
        "text-generation",
        model="gpt2",
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    query = st.text_input("Ask a question from the document", key="question")

    if query:
        with st.spinner("Generating answer..."):
            result = qa(query)

        answer = result["result"]

        clean_markers = [
            "Helpful Answer:",
            "Use the following pieces of context",
            "\nQuestion:"
        ]
        for marker in clean_markers:
            if marker in answer:
                answer = answer.split(marker)[-1].strip()

        st.subheader("Answer:")
        st.write(answer)

        st.subheader("Top Retrieved Chunks:")
        for doc in result["source_documents"]:
            st.write(doc.page_content)
            st.write("------")