import os
import streamlit as st
from dotenv import load_dotenv

from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_groq import ChatGroq

# LOAD ENV VARIABLES
load_dotenv()

# PAGE SETTINGS
st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI PDF Chatbot")
st.write("Upload one or more PDFs and ask questions.")

# MULTIPLE PDF UPLOAD
uploaded_files = st.file_uploader(
    "Upload PDF Files",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    all_text = ""

    # READ ALL PDFS
    for uploaded_file in uploaded_files:

        pdf_reader = PdfReader(uploaded_file)

        for page in pdf_reader.pages:

            text = page.extract_text()

            if text:
                all_text += text

    # SPLIT INTO CHUNKS
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = splitter.split_text(all_text)

    # EMBEDDINGS
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # VECTOR DATABASE
    vector_store = Chroma.from_texts(
        chunks,
        embedding=embeddings
    )

    st.success("Vector Database Created Successfully ✅")

    # USER QUESTION
    question = st.chat_input(
        "Ask a question about your PDFs..."
    )

    if question:

        with st.spinner("Searching documents..."):

            docs = vector_store.max_marginal_relevance_search(
                question,
                k=5,
                fetch_k=15
            )

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        # DEBUG SECTION
        st.subheader("Debug Context")
        st.write(context)

        # GROQ MODEL
        llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant"
        )

        prompt = f"""
You are an intelligent document assistant.

Your task is to answer questions based ONLY on the provided document context.

Rules:
- Use the context as your primary source of information.
- Give clear and well-structured answers.
- Use bullet points when listing items.
- Summarize long information when appropriate.
- If the answer is not found in the context, say:
  "I could not find that information in the uploaded document."
- Do not invent information.
- Do not mention missing PDFs or access limitations.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

        with st.spinner("Generating answer..."):

            response = llm.invoke(prompt)

        st.subheader("Answer")

        st.write(response.content)
