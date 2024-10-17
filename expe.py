from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import tempfile
import concurrent.futures
from tenacity import retry, stop_after_attempt, wait_exponential

app = Flask(__name__)

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Increase chunk size and overlap for larger PDFs
CHUNK_SIZE = 1000000
CHUNK_OVERLAP = 100000

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_pdf_text(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
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
    return vector_store


def get_conversational_chain():
    prompt_template = """
    You are an AI assistant capable of handling a wide range of queries. Answer the question based on the following guidelines:

    1. If the answer is in the provided context, use that information.
    2. For general knowledge questions not in the context, use your built-in knowledge.
    3. For resume-related questions, analyze the context and provide informed suggestions.
    4. For programming questions, provide explanations or code snippets as needed.
    5. For research paper analysis:
       - Summarize key findings and methodologies
       - Explain complex scientific concepts
       - Identify the main contributions of the paper
       - Suggest related papers or further areas of research
    6. Always indicate the basis of your answer (context, general knowledge, analysis, programming expertise, or research paper analysis).

    Context: {context}

    Question: {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
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
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        text = get_pdf_text(filepath)
        text_chunks = get_text_chunks(text)
        vector_store = get_vector_store(text_chunks)

        # Save vector_store to a file or database for later use
        vector_store.save_local("vector_store")

        return jsonify({'success': 'File uploaded and processed successfully'})
    return jsonify({'error': 'File type not allowed'})


@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data['question']

    # Load vector_store from file or database
    vector_store = FAISS.load_local("vector_store")

    try:
        answer = user_input_with_retry(question, vector_store)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)