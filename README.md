# File Upload and Q&A with Retrieval-Augmented Generation (RAG)

## Description

This project is a simple web application that allows users to upload files (PDF or text) and ask questions about the content of those files. It uses Retrieval-Augmented Generation (RAG) techniques to answer questions based on the information contained in the uploaded documents.

## Technologies Used

- **Flask:** Web framework for Python.
- **Langchain:** Library for document processing and manipulation.
- **FAISS:** Library for efficient similarity search between vectors.
- **OpenAI API:** Service for generating text with advanced language models.

## Features

- **File Upload:** Allows users to upload files in PDF or text format.
- **Document Processing:** Loads and splits the content of the documents into smaller parts.
- **Embedding Generation:** Converts text into embeddings using OpenAI.
- **Storage and Search with FAISS:** Stores embeddings in a FAISS index for efficient searches.
- **Question Answering:** Uses a language model (LLM) to answer questions based on retrieved information.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/your-repo.git
    ```
2. Navigate to the project directory:
    ```sh
    cd your-repo
    ```
3. Create a virtual environment and activate it:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```
4. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```
5. Create a `.env` file in the project root and add your OpenAI API key:
    ```env
    OPENAI_API_KEY=your_openai_api_key
    ```

## Usage

1. Run the application:
    ```sh
    flask run
    ```
2. Access the application in your web browser:
    ```
    http://127.0.0.1:5000/
    ```
3. Upload a file and then ask questions about the content of the uploaded file.

## Code Structure

- **app.py:** Contains all the logic for the Flask application.
- **templates/index.html:** HTML template for the web interface.
