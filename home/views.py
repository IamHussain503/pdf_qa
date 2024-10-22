from django.shortcuts import render, redirect
from django.core.files.storage import default_storage
from .models import UploadedDocument
import openai
from pymongo import MongoClient
from openai import OpenAI, AssistantEventHandler
from typing_extensions import override


from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from .models import UploadedDocument
import openai


import logging
logger = logging.getLogger(__name__)

class UploadDocumentAPI(APIView):
    def post(self, request):
        logger.info("POST request received for document upload")
        if 'pdf_file' not in request.FILES:
            logger.error("No file in request")
            return Response({"error": "No PDF file uploaded."}, status=status.HTTP_400_BAD_REQUEST)

        pdf_file = request.FILES['pdf_file']
        logger.info(f"Received file: {pdf_file.name}")

        try:
            # Create vector store
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
        # Connect to MongoDB directly
        client = MongoClient("mongodb+srv://zaidali:IUzvdpQZ7MMaixB8@cluster0.qmehy3e.mongodb.net/Todo?retryWrites=true&w=majority")
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



# Initialize the OpenAI client
#api key here

# Define the event handler for streaming responses
class EventHandler(AssistantEventHandler):
    def __init__(self):
        super().__init__()  # Initialize the parent class
        self.response = ""  # Initialize response attribute

    @override
    def on_text_created(self, text) -> None:
        # Append streamed text to the response
        print(f"\nassistant > {text}", end="", flush=True)
        self.response += str(text)

    @override
    def on_tool_call_created(self, tool_call):
        print(f"\nassistant > Tool call created: {tool_call.type}\n", flush=True)

    @override
    def on_message_done(self, message) -> None:
        # Extract the actual value field from the message content
        message_content = message.content[0].text  # Get the message content
        value = message_content.value  # Extract only the 'value' field

        # Only append if the value is meaningful, avoid appending 'Text(...)' part
        if value:
            self.response += str(value)  # Append only the value part to response

        print("\nMessage content (value only):", value)





# Initialize the assistant
assistant = openai_client.beta.assistants.create(
    name="PDF File QA Assistant",
    instructions="You are an assistant who answers questions based on the content of uploaded PDF files.",
    model="gpt-4o",
    tools=[{"type": "file_search"}]
)

# Function to handle the file upload and create the vector store
from bson import ObjectId  # Add this import
from pymongo import MongoClient

from bson.objectid import ObjectId
from pymongo import MongoClient

from pymongo import MongoClient

def upload_pdf_page(request):
    client = MongoClient("mongodb+srv://zaidali:IUzvdpQZ7MMaixB8@cluster0.qmehy3e.mongodb.net/Todo?retryWrites=true&w=majority")
    db = client.Todo
    collection = db['home_uploadeddocument']

    if request.method == 'POST':
        if 'pdf_file' in request.FILES:
            pdf_file = request.FILES['pdf_file']
            file_name = pdf_file.name
            vector_store = upload_file_and_create_vector_store(pdf_file, file_name)

            # Insert manually into MongoDB
            document = {
                "file_name": file_name,
                "vector_store_id": vector_store.id
            }
            result = collection.insert_one(document)
            print(f"Document inserted with ID: {result.inserted_id}")

            return redirect('upload_pdf_page')
        
        # Other parts of the code

        elif 'uploaded_file' in request.POST and 'question' in request.POST:
            uploaded_file_id = request.POST['uploaded_file']
            question = request.POST['question']

            if uploaded_file_id:
                try:
                    uploaded_file = collection.find_one({"_id": ObjectId(uploaded_file_id)})

                    if uploaded_file:
                        answer = ask_question_with_file_search(question, uploaded_file['vector_store_id'])
                    else:
                        print("File not found in database.")
                except Exception as e:
                    print(f"Error retrieving file: {e}")
            else:
                print("No file selected.")

    return render(request, 'upload_pdf.html', {
        'uploaded_files': uploaded_files,
        'answer': answer,
        'question': question
    })





def upload_file_and_create_vector_store(pdf_file, vector_store_name: str):
    """Create a vector store and upload the file content."""
    # First, ensure we are working with a file-like object
    if isinstance(pdf_file, str):
        raise ValueError("Expected a file-like object, but got a string instead.")

    # Create a vector store with the provided name (vector_store_name = file_name in this case)
    vector_store = openai_client.beta.vector_stores.create(name=vector_store_name)

    # Ensure the file pointer is at the start (seek to the beginning)
    try:
        pdf_file.seek(0)
    except AttributeError as e:
        raise ValueError(f"pdf_file is not a file-like object: {e}")

    # Read the file as bytes and upload to OpenAI
    file_content = pdf_file.read()  # Read the file content as bytes

    # Upload the file to OpenAI using its byte content
    file_batch = openai_client.beta.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id,
        files=[(pdf_file.name, file_content)]  # Upload as a tuple (filename, file content in bytes)
    )

    print(f"Vector store created with ID: {vector_store.id}")
    print(f"File batch status: {file_batch.status}")

    return vector_store


def ask_question_with_file_search(question: str, vector_store_id: str):
    """Ask a question using the vector store and get a streaming response."""
    # Create a thread with the question and reference the vector store
    thread = openai_client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": question
            }
        ],
        tool_resources={
            "file_search": {
                "vector_store_ids": [vector_store_id]
            }
        }
    )

    # Use the event handler to stream the response
    event_handler = EventHandler()

    print(f"Question '{question}' is being asked with vector store ID '{vector_store_id}'...")

    # Handle the stream using OpenAI's internal stream management
    with openai_client.beta.threads.runs.stream(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions="Please address the user as Jane Doe. The user has a premium account.",
        event_handler=event_handler,
    ) as stream:
        stream.until_done()  # Wait for the stream to finish

    # Return the final streamed response
    return event_handler.response
