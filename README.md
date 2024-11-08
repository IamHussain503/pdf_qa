
# Document QA Assistant

This project is a Django-based web application that allows users to upload PDF and Excel files, process them for querying, and ask questions related to the content in the documents using OpenAI's API.

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
- An API for uploading PDF and Excel files (XLSX) and processing their content.
- A question-answering API that searches within the document content (text for PDFs and CSV data for Excel files) to answer user queries.
- A MongoDB database to store uploaded file information and document content.
- A REST API interface for interacting with the application.

## Requirements

Ensure you have the following installed:
- Python 3.8+
- Django 4.1.13
- MongoDB (or access to a MongoDB database via a URI)
- OpenAI API account and credentials

## Setup and Installation

1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd document_qa_assistant
   ```

2. Set up the environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   ```bash
   export OPENAI_API_KEY="your_openai_api_key_here"
   export MONGODB_URL="your_mongodb_url_here"
   ```

4. Make sure to add these exports to your shell’s configuration file (e.g., `~/.bashrc`) and source it:
   ```bash
   source ~/.bashrc
   ```

## Django Migrations

1. Apply migrations:
   ```bash
   python manage.py migrate
   ```

## Running the Application

Start the Django development server:
```bash
python manage.py runserver 0.0.0.0:8001
```

## API Endpoints

The following API endpoints are available:

### 1. Upload Document API
- **Endpoint**: `/api/upload_document/`
- **Method**: `POST`
- **Description**: Upload a PDF or Excel document and store its content.
- **Request**:
  - **Form Data**:
    - `file`: The PDF or Excel file to upload. Accepts both `.pdf` and `.xlsx` file types.
- **Response**:
  - `file_name`: The name of the uploaded file.
  - `document_id`: The document ID associated with the file.
  - For PDFs: `full_text` with the entire extracted text.
  - For Excel files: `csv_chunks` as a list of CSV-formatted text chunks.
- **Sample command**:
  ```bash
  curl -X POST http://localhost:8001/api/upload_document/ -F "file=@path/to/yourfile.pdf"
  ```
  ```bash
  curl -X POST http://localhost:8001/api/upload_document/ -F "file=@path/to/yourfile.xlsx"
  ```

### 2. Retrieve Document API
- **Endpoint**: `/api/retrieve_documents/`
- **Method**: `GET`
- **Description**: Retrieve information on all uploaded documents.
- **Response**: JSON array of documents, each containing:
  - `document_id`
  - `file_name`
  - `upload_date`
  - `file_type` (either `"pdf"` or `"excel"`)
- **Sample command**:
  ```bash
  curl -X GET http://localhost:8001/api/retrieve_documents/
  ```

### 3. Ask Question API
- **Endpoint**: `/api/ask_question/`
- **Method**: `POST`
- **Description**: Ask a question based on the content of an uploaded document.
- **Request**:
  - **JSON**:
    - `question`: The question to ask.
    - `document_id`: The ID of the document to query (obtained from the upload response or retrieve documents API).
- **Response**:
  - `question`: The question asked.
  - `answer`: The answer based on the document content.
- **Sample command**:
  ```bash
  curl -X POST http://localhost:8001/api/ask_question/ -H "Content-Type: application/json" -d "{"question": "What is the purpose of the document?", "document_id": "your_document_id_here"}"
  ```

## Testing the API

Use `curl` commands provided in each section or tools like Postman to interact with the API endpoints.

## Logging and Debugging

- The application uses Django’s logging framework for debugging. Logs can be viewed in the terminal where the server is running.
- Modify logging levels in Django settings for additional details during development.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
