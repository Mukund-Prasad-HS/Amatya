from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
import os
import tempfile
import pickle
from concurrent.futures import ThreadPoolExecutor
import google.generativeai as genai
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from helpers import process_document, get_text_chunks, get_vector_store, user_input_with_retry, is_programming_question

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

executor = ThreadPoolExecutor(max_workers=2)

UPLOAD_FOLDER = 'uploads'
FAISS_FOLDER = 'faiss_store'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['FAISS_FOLDER'] = FAISS_FOLDER

# Ensure the upload and FAISS folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FAISS_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        files = request.files.getlist('file')
        if not files:
            flash('No selected file')
            return redirect(request.url)

        session['user_name'] = request.form['user_name']
        session['dark_mode'] = 'dark_mode' in request.form

        file_paths = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                file_paths.append(file_path)

        # Process files asynchronously
        texts = executor.submit(lambda: [process_document(file) for file in file_paths]).result()
        raw_text = "\n".join(texts)
        text_chunks = get_text_chunks(raw_text)
        vector_store = get_vector_store(text_chunks)

        # Save the FAISS vector store to a file
        faiss_file = os.path.join(app.config['FAISS_FOLDER'], 'faiss_store.pkl')
        with open(faiss_file, 'wb') as f:
            pickle.dump(vector_store, f)

        session['faiss_file'] = faiss_file
        flash('Documents processed successfully!')
        return redirect(url_for('index'))

    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_question = request.form['user_question']
    session['chat_history'] = session.get('chat_history', [])
    session['chat_history'].append({"role": "user", "content": user_question})

    faiss_file = session.get('faiss_file')
    try:
        if not faiss_file and not is_programming_question(user_question):
            response = get_general_knowledge_answer(user_question)
        else:
            # Load the FAISS vector store from the file
            with open(faiss_file, 'rb') as f:
                vector_store = pickle.load(f)
            response = user_input_with_retry(user_question, vector_store)
        session['chat_history'].append({"role": "assistant", "content": response})
    except Exception as e:
        response = f"An error occurred: {str(e)}. Please try again later."
        session['chat_history'].append({"role": "assistant", "content": response})

    return redirect(url_for('index'))


@app.route('/clear_chat')
def clear_chat():
    session.pop('chat_history', None)
    return redirect(url_for('index'))


@app.route('/summarize')
def summarize():
    faiss_file = session.get('faiss_file')
    with open(faiss_file, 'rb') as f:
        vector_store = pickle.load(f)
    summary = user_input_with_retry("Summarize the key points of the uploaded documents", vector_store)
    return render_template('result.html', result=summary)


@app.route('/identify_topics')
def identify_topics():
    faiss_file = session.get('faiss_file')
    with open(faiss_file, 'rb') as f:
        vector_store = pickle.load(f)
    topics = user_input_with_retry("Identify the main topics discussed in the uploaded documents", vector_store)
    return render_template('result.html', result=topics)


@app.route('/suggest_reading')
def suggest_reading():
    faiss_file = session.get('faiss_file')
    with open(faiss_file, 'rb') as f:
        vector_store = pickle.load(f)
    suggestions = user_input_with_retry(
        "Suggest related topics or areas for further reading based on the uploaded documents", vector_store)
    return render_template('result.html', result=suggestions)


if __name__ == '__main__':
    app.run(debug=True)
