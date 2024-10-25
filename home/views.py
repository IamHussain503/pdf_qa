from django.shortcuts import render, redirect
from django.core.files.storage import default_storage
from .models import UploadedDocument
import openai
import os
from pymongo import MongoClient
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
import logging

logger = logging.getLogger(__name__)

# Set up OpenAI API and MongoDB connection using environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
client = MongoClient(os.getenv("MONGODB_URL"))


class UploadDocumentAPI(APIView):
    def post(self, request):
        logger.info("POST request received for document upload")
        if 'pdf_file' not in request.FILES:
            logger.error("No file in request")
            return Response({"error": "No PDF file uploaded."}, status=status.HTTP_400_BAD_REQUEST)

        pdf_file = request.FILES['pdf_file']
        logger.info(f"Received file: {pdf_file.name}")

        try:
            # Create vector store (assuming a function is defined to handle this)
            vector_store = upload_file_and_create_vector_store(pdf_file, pdf_file.name)
            logger.info(f"Vector store created with ID: {vector_store.id}")

            # Save the document to MongoDB
            document = UploadedDocument.objects.create(
                file_name=pdf_file.name,
                vector_store_id=vector_store.id
            )
            logger.info(f"Document {document.file_name} saved with vector_store_id {document.vector_store_id}")

            return Response({
                "file_name": document.file_name,
                "vector_store_id": document.vector_store_id
            }, status=status.HTTP_201_CREATED)

        except Exception as e:
            logger.error(f"Error during document upload: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class RetrieveDocumentAPI(APIView):
    """API to retrieve file_name and vector_store_id for all uploaded documents."""
    def get(self, request):
        db = client.Todo
        collection = db['home_uploadeddocument']

        # Fetch all documents from MongoDB
        documents = list(collection.find({}, {'_id': 1, 'file_name': 1, 'vector_store_id': 1}))

        # Prepare the response
        documents_list = [
            {"file_name": doc['file_name'], "vector_store_id": doc['vector_store_id']}
            for doc in documents
        ]
        return JsonResponse(documents_list, safe=False)


class AskQuestionAPI(APIView):
    """API to answer questions using the vector store."""
    def post(self, request):
        question = request.data.get('question', None)
        vector_store_id = request.data.get('vector_store_id', None)

        if not question or not vector_store_id:
            return Response({"error": "Both 'question' and 'vector_store_id' are required."}, status=status.HTTP_400_BAD_REQUEST)

        # Get the answer from the vector store
        answer = ask_question_with_file_search(question, vector_store_id)
        
        return Response({"question": question, "answer": answer}, status=status.HTTP_200_OK)


def upload_file_and_create_vector_store(pdf_file, vector_store_name: str):
    """Create a vector store and upload the file content."""
    if isinstance(pdf_file, str):
        raise ValueError("Expected a file-like object, but got a string instead.")

    vector_store = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are an assistant handling PDF uploads."}]
    )

    try:
        pdf_file.seek(0)
    except AttributeError as e:
        raise ValueError(f"pdf_file is not a file-like object: {e}")

    file_content = pdf_file.read()  # Read the file content as bytes
    vector_store.id = "example_id"  # Placeholder ID for the vector store
    print(f"Vector store created with ID: {vector_store.id}")

    return vector_store


def ask_question_with_file_search(question: str, vector_store_id: str):
    """Ask a question using the vector store and get a response."""
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an assistant who answers questions based on PDF file content."},
            {"role": "user", "content": question}
        ]
    )

    answer = response['choices'][0]['message']['content']
    return answer
