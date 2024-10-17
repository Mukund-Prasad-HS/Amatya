import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Increase chunk size and overlap for larger PDFs
CHUNK_SIZE = 500000
CHUNK_OVERLAP = 50000


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
        You are an expert at extracting information and answering questions based on the provided context. Answer the question based on the context provided. If the answer is not contained within the context, say "I don't have enough information to answer that question."

        Context: {context}

        Question: {question}

        Answer:
        """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def get_general_qa_model():
    return ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    pdf_chain = get_conversational_chain()
    pdf_response = pdf_chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    if "I don't have enough information to answer that question" in pdf_response["output_text"]:
        model = get_general_qa_model()
        general_response = model({"messages": [{"role": "user", "content": user_question}]})
        return general_response["text"]
    else:
        return pdf_response["output_text"]


def main():
    st.set_page_config(page_title="Chat with PDF", page_icon=":books:", layout="wide")
    st.header("Chat with PDF using Gemini 💬📚")

    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'user_name' not in st.session_state:
        st.session_state.user_name = ""

    # Sidebar
    with st.sidebar:
        st.title("Menu")
        st.image("logo.png", width=150)

        # User name input
        user_name = st.text_input("Your Name", value=st.session_state.user_name)
        if user_name != st.session_state.user_name:
            st.session_state.user_name = user_name

        # File uploader
        st.title("File Upload")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type="pdf")

        if st.button("Process PDFs"):
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("PDFs processed successfully!")

        # Add dark mode toggle
        if st.checkbox("Dark Mode"):
            st.markdown("""
                <style>
                    .stApp {
                        background-color: #2b2b2b;
                        color: white;
                    }
                </style>
            """, unsafe_allow_html=True)

    # Main chat interface
    st.subheader("Chat")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(f"{message['content']}")

    # User input
    user_question = st.chat_input("Ask a question about your PDFs or anything else")
    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)

        with st.spinner("Thinking..."):
            response = user_input(user_question)

        st.session_state.chat_history.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)

    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.experimental_rerun()

    # Footer
    st.markdown("---")
    st.markdown("Created with ❤️ by Your Name | [GitHub](https://github.com/yourusername)")

if __name__ == "__main__":
    main()
