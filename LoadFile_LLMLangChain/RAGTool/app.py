import os
import requests
from dotenv import load_dotenv
from flask import Flask, request, render_template, redirect, url_for
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
from pydantic import Field

# Initial configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(current_dir, '.env')
load_dotenv(dotenv_path)

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not found in the .env file")

os.environ["OPENAI_API_KEY"] = openai_api_key

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(current_dir, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Custom LLM class
class CustomLLM(LLM):
    api_url: str = Field(...)
    model: str = Field(...)
    temperature: float = Field(...)

    def __init__(self, **data):
        super().__init__(**data)
        self.api_url = data.get("api_url")
        self.model = data.get("model")
        self.temperature = data.get("temperature")

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = requests.post(
            self.api_url,
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": 300  # You can adjust this as needed
            },
            headers={"Authorization": f"Bearer {openai_api_key}"}
        )
        try:
            response_data = response.json()
            if 'choices' in response_data:
                return response_data['choices'][0]['message']['content']
            else:
                raise ValueError(f"Unexpected response format: {response_data}")
        except Exception as e:
            raise ValueError(f"Error in API response: {str(e)}")

    @property
    def _llm_type(self) -> str:
        return "custom"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "api_url": self.api_url,
            "model": self.model,
            "temperature": self.temperature
        }

# Specify the LLM model and API details
LLM_API_URL = "https://api.openai.com/v1/chat/completions"
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.7

def load_document(file_path):
    _, file_extension = os.path.splitext(file_path)
    try:
        if file_extension.lower() == '.pdf':
            loader = PyPDFLoader(file_path)
            return loader.load()
        elif file_extension.lower() == '.txt':
            return TextLoader(file_path).load()
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    except Exception as e:
        print(f"Error loading the document: {str(e)}")
        raise

def get_uploaded_files():
    return [f for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]

def process_file(file_path):
    documents = load_document(file_path)
    text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(texts, embeddings)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = file.filename
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            process_file(file_path)
            return redirect(url_for('index'))
    files = get_uploaded_files()
    return render_template('index.html', files=files)

@app.route('/ask', methods=['POST'])
def ask_question():
    file_name = request.form['file']
    question = request.form['question']
    file_path = os.path.join(UPLOAD_FOLDER, file_name)

    try:
        vectorstore = process_file(file_path)
        llm = CustomLLM(api_url=LLM_API_URL, model=LLM_MODEL, temperature=LLM_TEMPERATURE)
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
        response = qa.run(question)
        return render_template('index.html', files=get_uploaded_files(), response=response, current_file=file_name)
    except Exception as e:
        error = f"Error processing the question: {str(e)}"
        return render_template('index.html', files=get_uploaded_files(), error=error)

@app.route('/delete/<filename>')
def delete_file(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
