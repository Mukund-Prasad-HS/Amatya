from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential
import time

CHUNK_SIZE = 1000000
CHUNK_OVERLAP = 100000

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

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain():
    prompt_template = """
    You are an AI assistant specializing in analyzing research papers and scientific documents, in addition to your other capabilities. Answer the question based on the following guidelines:

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

def is_research_paper(text):
    keywords = ['abstract', 'introduction', 'methodology', 'results', 'conclusion', 'references']
    return sum(1 for keyword in keywords if keyword in text.lower()) >= 4

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def user_input_with_retry(user_question, vector_store):
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    return response["output_text"]

def is_programming_question(question):
    programming_keywords = ['code', 'program', 'function', 'algorithm', 'syntax', 'debug']
    return any(keyword in question.lower() for keyword in programming_keywords)

def get_general_knowledge_answer(question):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    response = model.predict(f"Answer this general knowledge question: {question}")
    return response

def process_document(file):
    text = get_pdf_text(file)
    if is_research_paper(text):
        sections = extract_research_paper_text(file)
        structured_text = "\n\n".join([f"{section.upper()}:\n{content}" for section, content in sections.items()])
        return structured_text
    else:
        return text

def extract_research_paper_text(pdf_file):
    text = get_pdf_text(pdf_file)
    sections = identify_paper_sections(text)
    return sections

def identify_paper_sections(text):
    sections = {
        "abstract": "",
        "introduction": "",
        "methodology": "",
        "results": "",
        "discussion": "",
        "conclusion": "",
        "references": ""
    }
    current_section = ""
    for line in text.split('\n'):
        lower_line = line.lower()
        if "abstract" in lower_line:
            current_section = "abstract"
        elif "introduction" in lower_line:
            current_section = "introduction"
        elif "method" in lower_line or "methodology" in lower_line:
            current_section = "methodology"
        elif "result" in lower_line:
            current_section = "results"
        elif "discussion" in lower_line:
            current_section = "discussion"
        elif "conclusion" in lower_line:
            current_section = "conclusion"
        elif "reference" in lower_line:
            current_section = "references"
        if current_section:
            sections[current_section] += line + "\n"
    return sections
