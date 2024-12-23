import os
import logging
import csv
import pandas as pd
from django.shortcuts import render, redirect
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from bson import ObjectId
from pymongo import MongoClient
import openai
from .models import UploadedDocument
import re
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI  # Updated import for ChatOpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


logger = logging.getLogger(__name__)

# MongoDB and OpenAI setup
client = MongoClient(os.getenv("MONGODB_URL"))
db = client.Todo
pdf_collection = db['pdf_documents']  # Collection for PDF documents
excel_collection = db['excel_documents']  # Collection for Excel documents
openai.api_key = os.getenv("OPENAI_API_KEY")

# Directory to store CSV files for Excel documents
csv_dir = "documents"
os.makedirs(csv_dir, exist_ok=True)


class EventHandler:
    """Handles OpenAI responses."""

    def __init__(self):
        self.response = ""

    def on_text_created(self, text):
        print(f"\nassistant > {text}", end="", flush=True)
        self.response += str(text)

    def on_message_done(self, response):
        message_content = response.choices[0].text
        if message_content:
            self.response += message_content
        print("\nMessage content:", message_content)


def upload_file_and_create_vector_store(pdf_file, vector_store_name: str):
    """Create a vector store for a PDF file using OpenAI."""
    try:
        response = openai.File.create(file=pdf_file, purpose="answers")
        return {"id": response["id"]}
    except OpenAIError as e:
        logger.error(f"Error creating vector store for PDF: {e}")
        return None


from openai import OpenAIError

def ask_question_with_file_search(question: str, vector_store_id: str):
    """Ask a question using the OpenAI API and simulate a vector search."""
    try:
        modified_question = f"{question} Please do not send any relevant links in the answer as well as remove any unwanted characters from the answer."

        # Create a chat completion to simulate file search
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=modified_question,
            max_tokens=150,
        )

        return response.choices[0].text.strip()

    except OpenAIError as e:
        # Handle errors here
        error_message = str(e)
        if "404" in error_message and "not found" in error_message:
            print(f"Vector store with ID '{vector_store_id}' not found.")
            # Handle specific error case
        else:
            print(f"Error during question processing: {e}")
        raise e  # Re-raise if you want to handle it further up the stack



def upload_pdf_page(request):
    """Frontend view to upload PDF and ask questions."""
    uploaded_files = list(pdf_collection.find({}, {'_id': 1, 'file_name': 1}))
    for file in uploaded_files:
        file['file_id'] = str(file['_id'])

    answer = None
    question = None

    if request.method == 'POST':
        if 'pdf_file' in request.FILES:
            pdf_file = request.FILES['pdf_file']
            file_name = pdf_file.name
            vector_store = upload_file_and_create_vector_store(pdf_file, file_name)

            document = {"file_name": file_name, "vector_store_id": vector_store['id']}
            pdf_collection.insert_one(document)
            return redirect('upload_pdf_page')

        elif 'uploaded_file' in request.POST and 'question' in request.POST:
            uploaded_file_id = request.POST['uploaded_file']
            question = request.POST['question']
            try:
                uploaded_file = pdf_collection.find_one({"_id": ObjectId(uploaded_file_id)})
                if uploaded_file:
                    answer = ask_question_with_file_search(question, uploaded_file['vector_store_id'])
            except Exception as e:
                logger.error(f"Error retrieving file: {e}")

    return render(request, 'upload_pdf.html', {
        'uploaded_files': uploaded_files,
        'answer': answer,
        'question': question
    })


class UploadDocumentAPI(APIView):
    """API to upload PDF documents and create a vector store for them."""

    def post(self, request):
        if 'pdf_files' not in request.FILES:
            return Response({"error": "No PDF files uploaded."}, status=status.HTTP_400_BAD_REQUEST)

        pdf_files = request.FILES.getlist('pdf_files')
        file_names = [pdf_file.name for pdf_file in pdf_files]

        vector_store = upload_file_and_create_vector_store(pdf_files[0], file_names[0])  # Assuming single vector store for all files
        if not vector_store:
            return Response({"error": "Failed to create vector store for PDF."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        documents = [{"file_name": name, "vector_store_id": vector_store['id']} for name in file_names]
        pdf_collection.insert_many(documents)

        return Response({
            "file_names": file_names,
            "vector_store_id": vector_store['id'],
            "status": "Files uploaded successfully"
        }, status=status.HTTP_201_CREATED)


import re

class UploadExcelAPI(APIView):
    """API to upload an Excel file, store as JSON in MongoDB, and prevent duplicates by document name."""

    def post(self, request):
        if 'excel_file' not in request.FILES:
            return Response({"error": "No Excel file uploaded."}, status=status.HTTP_400_BAD_REQUEST)

        excel_file = request.FILES['excel_file']
        document_name = excel_file.name.rsplit(".", 1)[0]  # Remove the file extension
        document_name = document_name.replace(" ", "_").lower()  # Normalize the name

        logger.debug(f"Normalized document_name: {document_name}")

        # Check if document name already exists in excel collection
        if excel_collection.find_one({"document_name": document_name}):
            return Response({"error": "Document with this name already exists."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Convert Excel to CSV and save
            df = pd.read_excel(excel_file)
            csv_path = f"documents/{document_name}.csv"
            df.to_csv(csv_path, index=False)

            # Store document metadata in MongoDB
            document = {
                "document_name": document_name,
                "csv_path": csv_path,
                "upload_date": datetime.utcnow()
            }
            result = excel_collection.insert_one(document)

            return Response({
                "document_id": str(result.inserted_id),
                "document_name": document_name,
                "csv_path": csv_path,
                "status": "Excel file uploaded and stored as CSV successfully"
            }, status=status.HTTP_201_CREATED)

        except Exception as e:
            logger.error(f"Error processing Excel file: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)






class RetrievePDFDocumentsAPI(APIView):
    """API to retrieve metadata for all uploaded PDF documents."""

    def get(self, request):
        try:
            documents = list(pdf_collection.find({}, {"file_name": 1, "vector_store_id": 1, "_id": 1}))
            documents_list = [
                {
                    "file_id": str(doc["_id"]),
                    "file_name": doc.get("file_name", "Unknown"),
                    "vector_store_id": doc.get("vector_store_id", "Unknown")
                }
                for doc in documents
            ]
            return JsonResponse(documents_list, safe=False)
        except Exception as e:
            logger.error(f"Error retrieving PDF documents: {e}")
            return Response({"error": "Failed to retrieve documents"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class RetrieveExcelAsCSVAPI(APIView):
    """API to retrieve Excel document by name and return CSV path if it exists."""

    def get(self, request, document_name):
        document = excel_collection.find_one({"document_name": document_name})
        if not document:
            return Response({"error": "Document not found."}, status=status.HTTP_404_NOT_FOUND)

        csv_file_path = document.get("csv_path")
        if not csv_file_path or not os.path.exists(csv_file_path):
            return Response({"error": "CSV file not found. Please upload and process the document."},
                            status=status.HTTP_404_NOT_FOUND)

        return Response({
            "message": f"CSV file is available at {csv_file_path}",
            "csv_file_path": csv_file_path
        }, status=status.HTTP_200_OK)

class AskQuestionAPI(APIView):
    """API to answer questions based on a PDF's vector store."""

    def post(self, request):
        question = request.data.get('question')
        vector_store_id = request.data.get('vector_store_id')

        if not question or not vector_store_id:
            return Response({"error": "Both 'question' and 'vector_store_id' are required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            answer = ask_question_with_file_search(question, vector_store_id)
            return Response({"question": question, "answer": answer}, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error while processing the question: {e}")
            return Response({"error": "Internal Server Error. Check logs for details."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)




import logging

logger = logging.getLogger(__name__)

import uuid
from pymongo import MongoClient
from datetime import datetime, timedelta
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

class AskExcelQuestionAPI(APIView):
    """API to answer questions based on an Excel document's CSV with persistent session context using MongoDB."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # MongoDB setup for session storage
        client = MongoClient(os.getenv("MONGODB_URL"))
        self.db = client.Todo
        self.session_collection = self.db['langchain_sessions']  # Initialize session collection
        self.excel_collection = self.db['excel_documents']  # Ensure the excel_collection is also initialized
        self.context_sessions = {}  # In-memory cache for active sessions

    def normalize_document_name(self, document_name):
        """Normalize the document name to ensure consistency in storage and retrieval."""
        return document_name.replace(" ", "_").lower()

    def initialize_langchain_session(self, csv_file_path, context_id=None):
        """Initialize a LangChain session and save context_id to MongoDB if necessary."""
        try:
            logger.debug(f"Initializing LangChain session for file: {csv_file_path}")

            # Load the CSV and set up embeddings and vector store
            loader = CSVLoader(file_path=csv_file_path)
            documents = loader.load()
            
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(documents, embeddings)
            retriever = vectorstore.as_retriever()
            
            # Initialize QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(),
                chain_type="stuff",
                retriever=retriever
            )

            # Use an existing context_id if provided, otherwise generate a new one
            context_id = context_id or str(uuid.uuid4())
            self.context_sessions[context_id] = qa_chain  # Cache session in memory

            # Save the context_id in MongoDB if it's a new session
            if not self.session_collection.find_one({"context_id": context_id}):
                self.session_collection.insert_one({
                    "context_id": context_id,
                    "csv_file_path": csv_file_path,
                    "created_at": datetime.utcnow()
                })

            logger.debug(f"LangChain session initialized with context_id: {context_id}")
            return context_id

        except Exception as e:
            logger.error(f"Failed to initialize LangChain session: {e}")
            raise

    def retrieve_session(self, csv_file_path, context_id):
        """Retrieve or reinitialize session if needed, ensuring a single context_id per document."""
        
        # Check if context_id is already in memory cache
        if context_id in self.context_sessions:
            logger.debug(f"Session found in memory for context_id: {context_id}")
            return self.context_sessions[context_id]
        
        # Check MongoDB for an existing session
        session_data = self.session_collection.find_one({"csv_file_path": csv_file_path})
        
        if session_data:
            logger.debug(f"Reusing existing context_id: {session_data['context_id']} for csv_file_path: {csv_file_path}")
            return self.initialize_langchain_session(csv_file_path, session_data["context_id"])

        # If no existing session, initialize a new one and return it
        return self.initialize_langchain_session(csv_file_path)

    def ask_question_in_session(self, context_id, question, csv_file_path):
        """Ask a question within an existing session, loading from MongoDB if needed."""
        qa_chain = self.retrieve_session(csv_file_path, context_id)
        if not qa_chain:
            return "Error: Context ID not found or session expired."

        try:
            logger.debug(f"Asking question in session context: {context_id}")
            answer = qa_chain.invoke({"query": question})
            logger.debug(f"Received answer: {answer}")
            return answer
        except Exception as e:
            logger.error(f"Error processing question in LangChain session: {e}")
            return "Error: Unable to process the question."

    def post(self, request):
        """Handle POST requests to answer questions based on the document's CSV with session context."""
        question = request.data.get('question')
        document_name = self.normalize_document_name(request.data.get('document_name'))
        context_id = request.data.get('context_id')

        if not question or not document_name:
            return Response({"error": "Both 'question' and 'document_name' are required."}, 
                            status=status.HTTP_400_BAD_REQUEST)

        try:
            # Retrieve document metadata from MongoDB
            logger.debug(f"Searching for document_name: {document_name}")
            document = self.excel_collection.find_one({"document_name": document_name})

            if not document:
                logger.error(f"Document with name '{document_name}' not found in MongoDB.")
                return Response({"error": "Document not found."}, status=status.HTTP_404_NOT_FOUND)
            
            csv_file_path = document.get("csv_path")

            # Ensure a single context_id is retrieved or created per document
            if not context_id:
                session_record = self.session_collection.find_one({"csv_file_path": csv_file_path})
                context_id = session_record["context_id"] if session_record else self.initialize_langchain_session(csv_file_path)
                logger.debug(f"Using context_id: {context_id} for document_name: {document_name}")

            # Use the context_id to ask the question in the session
            answer = self.ask_question_in_session(context_id, question, csv_file_path)

            return Response({"question": question, "answer": answer, "context_id": context_id}, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return Response({"error": "Internal Server Error. Check logs for details."}, 
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)






class RetrieveExcelDocumentsAPI(APIView):
    """API to retrieve metadata for all uploaded Excel documents."""

    def get(self, request):
        try:
            documents = list(excel_collection.find({}, {"document_name": 1, "_id": 1}))
            documents_list = [
                {
                    "file_id": str(doc["_id"]),
                    "document_name": doc.get("document_name", "Unknown")
                }
                for doc in documents
            ]
            return JsonResponse(documents_list, safe=False)
        except Exception as e:
            logger.error(f"Error retrieving Excel documents: {e}")
            return Response({"error": "Failed to retrieve documents"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
