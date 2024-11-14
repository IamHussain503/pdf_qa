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
    """API to upload an Excel file, store it as CSV in `documents` folder, and save in MongoDB."""

    def post(self, request):
        if 'excel_file' not in request.FILES:
            return Response({"error": "No Excel file uploaded."}, status=status.HTTP_400_BAD_REQUEST)

        excel_file = request.FILES['excel_file']
        
        # Clean the filename: remove quotes, replace spaces with underscores
        document_name = re.sub(r"[\'\"]", "", os.path.splitext(excel_file.name)[0]).strip().replace(" ", "_")
        csv_file_path = os.path.join(csv_dir, f"{document_name}.csv")

        # Check if document name already exists in the database
        if excel_collection.find_one({"document_name": document_name}):
            return Response({"error": "Document with this name already exists."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Load the Excel file into a DataFrame
            df = pd.read_excel(excel_file)
            # Save as CSV with the cleaned filename
            df.to_csv(csv_file_path, index=False)

            # Log the file path and document details
            logger.info(f"CSV saved at: {csv_file_path}")

            # Insert the document record into MongoDB
            document = {"document_name": document_name, "csv_path": csv_file_path}
            result = excel_collection.insert_one(document)

            return Response({
                "document_id": str(result.inserted_id),
                "document_name": document_name,
                "csv_path": csv_file_path,
                "status": "Excel file uploaded and stored as CSV successfully"
            }, status=status.HTTP_201_CREATED)

        except Exception as e:
            logger.error(f"Failed to save CSV file: {e}")
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
        self.session_collection = self.db['langchain_sessions']
        self.context_sessions = {}  # In-memory cache for active sessions

    def initialize_langchain_session(self, csv_file_path):
        """Initialize a LangChain session, save context_id to MongoDB."""
        try:
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

            # Generate a unique context_id
            context_id = str(uuid.uuid4())
            self.context_sessions[context_id] = qa_chain  # Cache session in memory

            # Persist session metadata in MongoDB
            self.session_collection.insert_one({
                "context_id": context_id,
                "csv_file_path": csv_file_path,
                "created_at": datetime.utcnow()
            })

            return context_id

        except Exception as e:
            logger.error(f"Failed to initialize LangChain session: {e}")
            raise

    def retrieve_session(self, context_id):
        """Retrieve session by context_id from memory or MongoDB if needed."""
        
        # Check if session is already in memory
        if context_id in self.context_sessions:
            return self.context_sessions[context_id]

        # Fetch session details from MongoDB if not found in memory
        session_data = self.session_collection.find_one({"context_id": context_id})
        
        if session_data:
            csv_file_path = session_data["csv_file_path"]
            # Reinitialize session and store it in memory
            qa_chain = self.initialize_langchain_session(csv_file_path)
            self.context_sessions[context_id] = qa_chain  # Cache restored session in memory
            return qa_chain

        # Return None if session cannot be found or reinitialized
        return None

    def ask_question_in_session(self, context_id, question):
        """Ask a question within an existing session, loading from MongoDB if needed."""
        qa_chain = self.retrieve_session(context_id)
        if not qa_chain:
            return "Error: Context ID not found or session expired."

        try:
            answer = qa_chain.invoke({"query": question})
            return answer
        except Exception as e:
            logger.error(f"Error processing question in LangChain session: {e}")
            return "Error: Unable to process the question."

    def post(self, request):
        """Handle POST requests to answer questions based on the document's CSV with session context."""
        question = request.data.get('question')
        document_name = request.data.get('document_name').replace(" ", "_")
        context_id = request.data.get('context_id')

        if not question or not document_name:
            return Response({"error": "Both 'question' and 'document_name' are required."}, 
                            status=status.HTTP_400_BAD_REQUEST)

        try:
            # Use existing context_id if provided
            if context_id:
                answer = self.ask_question_in_session(context_id, question)
            else:
                # Initialize a new session if no context_id is provided
                document = excel_collection.find_one({"document_name": document_name})
                if not document:
                    return Response({"error": "Document not found."}, status=status.HTTP_404_NOT_FOUND)
                
                csv_file_path = document.get("csv_path")
                context_id = self.initialize_langchain_session(csv_file_path)
                answer = self.ask_question_in_session(context_id, question)

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
