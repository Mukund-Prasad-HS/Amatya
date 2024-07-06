from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
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

app = Flask(__name__)

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Global variable to store the vector store
global_vector_store = None

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


def get_pdf_text(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def process_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_file_path = tmp_file.name

    text = get_pdf_text(tmp_file_path)
    os.unlink(tmp_file_path)
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_text(text)
    return chunks


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


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_pdfs():
    global global_vector_store
    if 'pdf' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    files = request.files.getlist('pdf')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No selected file'}), 400

    texts = []
    for file in files:
        text = process_pdf(file)
        texts.append(text)

    raw_text = "\n".join(texts)
    text_chunks = get_text_chunks(raw_text)
    global_vector_store = get_vector_store(text_chunks)

    return jsonify({'message': 'PDFs processed successfully'}), 200


@app.route('/ask', methods=['POST'])
def ask_question():
    global global_vector_store
    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    if not global_vector_store:
        return jsonify({'error': 'No PDFs processed yet'}), 400

    try:
        response = user_input_with_retry(question, global_vector_store)
        return jsonify({'response': response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)