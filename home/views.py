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
client = MongoClient(os.getenv("MONGODB_URL"))
# Initialize the OpenAI client
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# MongoDB setup
db = client.Todo
collection = db['home_uploadeddocument']


def upload_file_and_create_vector_store(pdf_file, vector_store_name: str):
    """Create a vector store and upload the file content."""
    vector_store = openai_client.beta.vector_stores.create(name=vector_store_name)
    file_content = pdf_file.read()  # Read the file content as bytes

    file_batch = openai_client.beta.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id,
        files=[(pdf_file.name, file_content)]
    )

    print(f"Vector store created with ID: {vector_store.id}")
    print(f"File batch status: {file_batch.status}")

    return vector_store


def ask_question_with_file_search(question: str, vector_store_id: str):
    """Ask a question using the vector store and get a streaming response."""
    thread = openai_client.beta.threads.create(
        messages=[{"role": "user", "content": question}],
        tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}}
    )

    response_text = ""
    for message in thread.messages:
        response_text += message['content']['text']

    return response_text


# Views and API endpoints
class UploadDocumentAPI(APIView):
    """API to upload a PDF document and create a vector store."""

    def post(self, request):
        if 'pdf_file' not in request.FILES:
            return Response({"error": "No PDF file uploaded."}, status=status.HTTP_400_BAD_REQUEST)

        pdf_file = request.FILES['pdf_file']
        file_name = pdf_file.name

        # Create vector store and upload content
        vector_store = upload_file_and_create_vector_store(pdf_file, file_name)
        document = {"file_name": file_name, "vector_store_id": vector_store.id}
        result = collection.insert_one(document)

        return Response({
            "file_name": file_name,
            "vector_store_id": vector_store.id,
            "document_id": str(result.inserted_id)
        }, status=status.HTTP_201_CREATED)


class RetrieveDocumentAPI(APIView):
    """API to retrieve file_name and vector_store_id for all uploaded documents."""

    def get(self, request):
        documents = list(collection.find({}, {"file_name": 1, "vector_store_id": 1, "_id": 1}))
        documents_list = [
            {"file_id": str(doc["_id"]), "file_name": doc["file_name"], "vector_store_id": doc["vector_store_id"]}
            for doc in documents
        ]
        return JsonResponse(documents_list, safe=False)


class AskQuestionAPI(APIView):
    """API to answer questions using the vector store."""

    def post(self, request):
        question = request.data.get('question')
        vector_store_id = request.data.get('vector_store_id')

        if not question or not vector_store_id:
            return Response({"error": "Both 'question' and 'vector_store_id' are required."}, status=status.HTTP_400_BAD_REQUEST)

        answer = ask_question_with_file_search(question, vector_store_id)

        return Response({"question": question, "answer": answer}, status=status.HTTP_200_OK)


def upload_pdf_page(request):
    """Frontend view to upload PDF and ask questions."""
    uploaded_files = list(collection.find({}, {'_id': 1, 'file_name': 1}))
    for file in uploaded_files:
        file['file_id'] = str(file['_id'])  # Convert ObjectId to string for template use

    answer = None
    question = None

    if request.method == 'POST':
        if 'pdf_file' in request.FILES:
            pdf_file = request.FILES['pdf_file']
            file_name = pdf_file.name
            vector_store = upload_file_and_create_vector_store(pdf_file, file_name)

            document = {"file_name": file_name, "vector_store_id": vector_store.id}
            collection.insert_one(document)
            return redirect('upload_pdf_page')

        elif 'uploaded_file' in request.POST and 'question' in request.POST:
            uploaded_file_id = request.POST['uploaded_file']
            question = request.POST['question']
            try:
                uploaded_file = collection.find_one({"_id": ObjectId(uploaded_file_id)})
                if uploaded_file:
                    answer = ask_question_with_file_search(question, uploaded_file['vector_store_id'])
            except Exception as e:
                print(f"Error retrieving file: {e}")

    return render(request, 'upload_pdf.html', {
        'uploaded_files': uploaded_files,
        'answer': answer,
        'question': question
    })
