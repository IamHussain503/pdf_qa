# PDF File QA Assistant

This project is a Django-based web application that allows users to upload PDF files, create vector stores for querying, and ask questions related to the content in the PDFs using OpenAI's API.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Requirements](#requirements)
3. [Setup and Installation](#setup-and-installation)
4. [Environment Variables](#environment-variables)
5. [Django Migrations](#django-migrations)
6. [Running the Application](#running-the-application)
7. [API Endpoints](#api-endpoints)
8. [Testing the API](#testing-the-api)
9. [Logging and Debugging](#logging-and-debugging)
10. [License](#license)

## Project Overview

This project includes:
- An API for uploading PDF files and creating vector stores.
- A question-answering API that uses the vector store to answer questions based on the uploaded PDF's content.
- A basic HTML form interface for interacting with the APIs through the browser.

## Requirements

Make sure you have the following installed:
- Python 3.8+
- Django 4.1.13
- MongoDB (or access to a MongoDB database via a URI)
- OpenAI API account and credentials

## Setup and Installation

1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd pdf_qa_assistant
## Setup enviornment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
export OPENAI_API_KEY="your_openai_api_key_here"
export MONGODB_URL="your_mongodb_url_here"
source ~/.bash_rc
```
## Django Migrations
```bash
python manage.py migrate
## Run the application
python manage.py runserver 0.0.0.0:8001
```

## API Endpoints
The following API endpoints are available:

### 1. Upload Document API
Endpoint: /api/upload_document/

Method: POST
Description: Upload a PDF document and create a vector store.
Request:
Form Data:
pdf_file: The PDF file to upload.
Response:
file_name: The name of the uploaded file.
vector_store_id: The vector store ID associated with the file.
### sample command
```bash
curl -X POST http://localhost:8001/api/upload_document/ -F "pdf_file=@path/to/yourfile.pdf"
```

## 2. Retrieve Document API
Endpoint: /api/retrieve_documents/

Method: GET
Description: Retrieve information on all uploaded documents.
Response: JSON array of documents, each containing file_id, file_name, and vector_store_id.
Sample curl command:
```bash
curl -X GET http://localhost:8001/api/retrieve_documents/
```
## 3. Ask Question API
Endpoint: /api/ask_question/

Method: POST
Description: Ask a question based on the content of an uploaded PDF document.
Request:
JSON:
question: The question to ask.
vector_store_id: The ID of the vector store created from the uploaded PDF.
Response:
question: The question asked.
answer: The answer based on the PDF content.
Sample curl command:
```bash
curl -X POST http://localhost:8001/api/ask_question/ -H "Content-Type: application/json" -d "{\"question\": \"What is the purpose of the document?\", \"vector_store_id\": \"your_vector_store_id_here\"}"
```
