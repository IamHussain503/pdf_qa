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
from openai.error import OpenAIError
from .models import UploadedDocument

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


def ask_question_with_file_search(question: str, vector_store_id: str):
    """Ask a question using OpenAI API with a specified vector store ID."""
    try:
        modified_question = f"{question} Please do not send any relevant links in the answer as well as remove any unwanted characters from the answer."
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=modified_question,
            max_tokens=150,
        )
        event_handler = EventHandler()
        answer = response.choices[0].text.strip()
        event_handler.on_message_done(response)
        return answer

    except OpenAIError as e:
        if "404" in str(e) and "not found" in str(e):
            pdf_collection.delete_one({"vector_store_id": vector_store_id})
            return {"error": "The vector store ID was not found and has been removed from the database."}
        else:
            logger.error(f"Error during question processing: {e}")
            raise e


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


class UploadExcelAPI(APIView):
    """API to upload an Excel file, store it as CSV in `documents` folder, and save in MongoDB."""

    def post(self, request):
        if 'excel_file' not in request.FILES:
            return Response({"error": "No Excel file uploaded."}, status=status.HTTP_400_BAD_REQUEST)

        excel_file = request.FILES['excel_file']
        # Use the filename without the `.xlsx` extension
        document_name = os.path.splitext(excel_file.name)[0]  # This removes the `.xlsx` if present

        # Check if document name already exists in excel collection
        if excel_collection.find_one({"document_name": document_name}):
            return Response({"error": "Document with this name already exists."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Load the Excel file into a DataFrame
            df = pd.read_excel(excel_file)
            # Construct the CSV path without `.xlsx` in the filename
            csv_file_path = os.path.join(csv_dir, f"{document_name}.csv")
            # Save as CSV
            df.to_csv(csv_file_path, index=False)
            
            # Insert document record into MongoDB
            document = {"document_name": document_name, "csv_path": csv_file_path}
            result = excel_collection.insert_one(document)
            
            return Response({
                "document_id": str(result.inserted_id),
                "document_name": document_name,
                "csv_path": csv_file_path,
                "status": "Excel file uploaded and stored as CSV successfully"
            }, status=status.HTTP_201_CREATED)

        except Exception as e:
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


class AskExcelQuestionAPI(APIView):
    """API to answer questions based on the Excel document's CSV with session persistence for context."""

    def post(self, request):
        question = request.data.get('question')
        document_name = request.data.get('document_name')

        if not question or not document_name:
            return Response({"error": "Both 'question' and 'document_name' are required."},
                            status=status.HTTP_400_BAD_REQUEST)

        document = excel_collection.find_one({"document_name": document_name})
        if not document:
            return Response({"error": "Document not found."}, status=status.HTTP_404_NOT_FOUND)

        csv_file_path = document.get("csv_path")
        if not csv_file_path or not os.path.exists(csv_file_path):
            return Response({"error": "CSV file not found. Please ensure the document is available and processed."},
                            status=status.HTTP_404_NOT_FOUND)

        session_id = document.get("langchain_session_id")
        if not session_id or not self.is_session_active(session_id):
            session_id = self.initialize_langchain_session(csv_file_path)
            excel_collection.update_one(
                {"_id": document["_id"]},
                {"$set": {"langchain_session_id": session_id}}
            )

        answer = self.ask_question_in_session(session_id, question)
        return Response({"question": question, "answer": answer}, status=status.HTTP_200_OK)

    def initialize_langchain_session(self, csv_file_path):
        """Initialize a LangChain session context with CSV data."""
        from langchain.document_loaders import CSVLoader
        from langchain.vectorstores import FAISS
        from langchain.embeddings import OpenAIEmbeddings
        from langchain.chains.question_answering import load_qa_chain

        loader = CSVLoader(csv_file_path=csv_file_path)
        documents = loader.load()
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)
        session_id = load_qa_chain(vectorstore)
        return session_id

    def ask_question_in_session(self, session_id, question):
        """Ask a question within the LangChain session context."""
        try:
            response = session_id.run({"question": question})
            return response
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return "Error: Unable to process the question."

    def is_session_active(self, session_id):
        """Check if the session context is active in LangChain."""
        try:
            return session_id.is_active()
        except Exception as e:
            logger.error(f"Error checking session status: {e}")
            return False


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
