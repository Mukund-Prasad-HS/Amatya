import streamlit as st
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
import tempfile
import concurrent.futures
import time
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Increase chunk size and overlap for larger PDFs
CHUNK_SIZE = 1000000
CHUNK_OVERLAP = 100000


class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = []

    def __call__(self, f):
        def wrapper(*args, **kwargs):
            now = time.time()
            self.calls = [c for c in self.calls if c > now - self.period]
            if len(self.calls) >= self.max_calls:
                raise Exception("Rate limit exceeded. Please try again later.")
            self.calls.append(now)
            return f(*args, **kwargs)

        return wrapper


@st.cache_resource
def get_pdf_text(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def process_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_file_path = tmp_file.name

    text = get_pdf_text(tmp_file_path)
    os.unlink(tmp_file_path)
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_text(text)
    return chunks


@st.cache_resource
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store


def get_conversational_chain():
    prompt_template = """
    Answer the question based on the context provided. If the answer is not contained within the context, say "I don't have enough information to answer that question."

    Context: {context}

    Question: {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
@RateLimiter(max_calls=10, period=60)  # 10 calls per minute
def user_input_with_retry(user_question, vector_store):
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    return response["output_text"]


def main():
    st.set_page_config(page_title="Chat with PDF", page_icon=":books:", layout="wide")
    st.header("AI Amatya 💬📚")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'user_name' not in st.session_state:
        st.session_state.user_name = ""
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None

    with st.sidebar:

        st.image("logo.png", width=150)
        st.title("Menu")
        user_name = st.text_input("Your Name", value=st.session_state.user_name)
        if user_name != st.session_state.user_name:
            st.session_state.user_name = user_name

        st.title("File Upload")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type="pdf")

        if st.button("Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing PDFs... This may take a while for large files."):
                    # Process PDFs in parallel
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        texts = list(executor.map(process_pdf, pdf_docs))

                    raw_text = "\n".join(texts)
                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.vector_store = get_vector_store(text_chunks)
                    st.success("PDFs processed successfully!")
            else:
                st.error("Please upload PDF files before processing.")

        if st.checkbox("Dark Mode"):
            st.markdown("""
                <style>
                    .stApp {
                        background-color: #2b2b2b;
                        color: white;
                    }
                </style>
            """, unsafe_allow_html=True)

    st.subheader("Chat")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(f"{message['content']}")

    user_question = st.chat_input("Ask a question about your PDFs")
    if user_question:
        if st.session_state.vector_store is None:
            st.error("Please process PDF files before asking questions.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.write(user_question)

            with st.spinner("Thinking...."):
                try:
                    response = user_input_with_retry(user_question, st.session_state.vector_store)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    with st.chat_message("assistant"):
                        st.write(response)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}. Please try again later.")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.experimental_rerun()

    st.markdown("---")
    st.markdown("Created with ❤️ by InnovAIt-ON")


if __name__ == "__main__":
    main()