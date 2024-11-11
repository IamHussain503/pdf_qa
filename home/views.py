import os
import openai
import logging
from django.shortcuts import render, redirect
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from bson import ObjectId
from pymongo import MongoClient
from openai import OpenAI, AssistantEventHandler, OpenAIError
from .models import UploadedDocument

logger = logging.getLogger(__name__)

# MongoDB and OpenAI setup
client = MongoClient(os.getenv("MONGODB_URL"))
db = client.Todo
collection = db['home_uploadeddocument']

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize the assistant for OpenAI
assistant = openai_client.beta.assistants.create(
    name="PDF File QA Assistant",
    instructions="You are an assistant who answers questions based on the content of uploaded PDF files.",
    model="gpt-4o",
    tools=[{"type": "file_search"}]
)


# class EventHandler(AssistantEventHandler):
#     """Handles OpenAI streaming responses."""
    
#     def __init__(self):
#         super().__init__()
#         self.response = ""

#     def on_text_created(self, text) -> None:
#         print(f"\nassistant > {text}", end="", flush=True)
#         self.response += str(text)

#     def on_tool_call_created(self, tool_call):
#         print(f"\nassistant > Tool call created: {tool_call.type}\n", flush=True)

#     def on_message_done(self, message) -> None:
#         message_content = message.content[0].text
#         value = message_content.value
#         if value:
#             self.response += str(value)
#         print("\nMessage content (value only):", value)

class EventHandler(AssistantEventHandler):
    """Handles OpenAI streaming responses."""
    
    def __init__(self):
        super().__init__()
        self.response = ""

    def on_text_created(self, text) -> None:
        print(f"\nassistant > {text}", end="", flush=True)
        self.response += str(text)

    def on_tool_call_created(self, tool_call):
        print(f"\nassistant > Tool call created: {tool_call.type}\n", flush=True)

    def on_tool_response(self, tool_response):
        # Log the actual content of the tool response to debug if file contents are retrieved
        print(f"\nTool response content: {tool_response.content}", flush=True)
        self.response += tool_response.content  # Append tool response content to overall response

    def on_message_done(self, message) -> None:
        if message.content:
            message_content = message.content[0].text
            self.response += str(message_content)
        print("\nFull response captured:", self.response)



def upload_file_and_create_vector_store(pdf_file, vector_store_name: str):
    """Create a vector store and upload the file content."""
    vector_store = openai_client.beta.vector_stores.create(name=vector_store_name)
    file_content = pdf_file.read()

    file_batch = openai_client.beta.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id,
        files=[(pdf_file.name, file_content)]
    )

    logger.info(f"Vector store created with ID: {vector_store.id}")
    logger.info(f"File batch status: {file_batch.status}")

    return vector_store


# def ask_question_with_file_search(question: str, vector_store_id: str):
#     """Ask a question using the vector store and get a streaming response."""
#     try:
#         # Add a phrase to the question before sending
#         modified_question = f"{question} Please do not send any relevant links in the answer as well as remove any unwanted characters from the answer."

#         # Create a thread with the modified question and reference the vector store
#         thread = openai_client.beta.threads.create(
#             messages=[{"role": "user", "content": modified_question}],
#             tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}}
#         )

#         # Use the event handler to stream the response
#         event_handler = EventHandler()
#         print(f"Question '{modified_question}' is being asked with vector store ID '{vector_store_id}'...")

#         # Handle the stream using OpenAI's internal stream management
#         with openai_client.beta.threads.runs.stream(
#             thread_id=thread.id,
#             assistant_id=assistant.id,
#             instructions="Please address the user as Jane Doe. The user has a premium account.",
#             event_handler=event_handler,
#         ) as stream:
#             stream.until_done()  # Wait for the stream to finish

#         # Return the final streamed response
#         return event_handler.response

#     except OpenAIError as e:
#         error_message = str(e)
#         if "404" in error_message and "not found" in error_message:
#             print(f"Vector store with ID '{vector_store_id}' not found in OpenAI. Deleting from MongoDB.")
            
#             # Delete the document from MongoDB
#             collection.delete_one({"vector_store_id": vector_store_id})
#             print(f"Vector store ID '{vector_store_id}' deleted from MongoDB.")
#             return {"error": "The vector store ID was not found and has been removed from the database."}
        
#         else:
#             print(f"Error during question processing: {e}")
#             raise e


def ask_question_with_file_search(question: str, vector_store_id: str):
    """Ask a question using the vector store and get a streaming response."""
    try:
        modified_question = f"{question} Please focus on detailed responses and use the relevant documents in the vector store."

        # Create a thread with the question and reference the vector store
        thread = openai_client.beta.threads.create(
            messages=[{"role": "user", "content": modified_question}],
            tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}}
        )

        # Use the event handler to stream the response
        event_handler = EventHandler()
        print(f"Question '{modified_question}' is being asked with vector store ID '{vector_store_id}'...")

        # Stream the response with added instructions
        with openai_client.beta.threads.runs.stream(
            thread_id=thread.id,
            assistant_id=assistant.id,
            instructions="Use all content from the vector store to provide a comprehensive answer.",
            event_handler=event_handler,
        ) as stream:
            stream.until_done()  # Wait for the stream to finish

        # Log the full response for debugging
        logger.info(f"Full response: {event_handler.response}")
        return event_handler.response

    except OpenAIError as e:
        error_message = str(e)
        if "404" in error_message and "not found" in error_message:
            logger.error(f"Vector store with ID '{vector_store_id}' not found. Deleting from MongoDB.")
            collection.delete_one({"vector_store_id": vector_store_id})
            return {"error": "The vector store ID was not found and has been removed from the database."}
        
        else:
            logger.error(f"Error during question processing: {e}")
            raise e





# API Views
# class UploadDocumentAPI(APIView):
#     """API to upload a PDF document and create a vector store."""

#     def post(self, request):
#         if 'pdf_file' not in request.FILES:
#             return Response({"error": "No PDF file uploaded."}, status=status.HTTP_400_BAD_REQUEST)

#         pdf_file = request.FILES['pdf_file']
#         file_name = pdf_file.name

#         vector_store = upload_file_and_create_vector_store(pdf_file, file_name)
#         document = {"file_name": file_name, "vector_store_id": vector_store.id}
#         result = collection.insert_one(document)

#         return Response({
#             "file_name": file_name,
#             "vector_store_id": vector_store.id,
#             "document_id": str(result.inserted_id)
#         }, status=status.HTTP_201_CREATED)\

class UploadDocumentAPI(APIView):
    """API to upload multiple PDF documents and create a single vector store for them."""

    def post(self, request):
        if 'pdf_files' not in request.FILES:
            return Response({"error": "No PDF files uploaded."}, status=status.HTTP_400_BAD_REQUEST)

        pdf_files = request.FILES.getlist('pdf_files')  # Get multiple files
        file_names = [pdf_file.name for pdf_file in pdf_files]

        # Create a vector store with multiple files
        vector_store = openai_client.beta.vector_stores.create(name="Multi-Document Vector Store")
        file_streams = [(pdf_file.name, pdf_file.read()) for pdf_file in pdf_files]

        # Upload multiple files to the vector store and monitor the status
        file_batch = openai_client.beta.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id,
            files=file_streams
        )

        if file_batch.status == "completed":
            logger.info(f"Vector store created with ID: {vector_store.id}")
            documents = [{"file_name": name, "vector_store_id": vector_store.id} for name in file_names]
            collection.insert_many(documents)

            return Response({
                "file_names": file_names,
                "vector_store_id": vector_store.id,
                "status": "Files uploaded successfully"
            }, status=status.HTTP_201_CREATED)
        else:
            return Response({
                "error": "Failed to upload files to vector store. Please try again."
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



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

        try:
            answer = ask_question_with_file_search(question, vector_store_id)
            return Response({"question": question, "answer": answer}, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error while processing the question: {e}")
            return Response({"error": "Internal Server Error. Check logs for details."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def upload_pdf_page(request):
    """Frontend view to upload PDF and ask questions."""
    uploaded_files = list(collection.find({}, {'_id': 1, 'file_name': 1}))
    for file in uploaded_files:
        file['file_id'] = str(file['_id'])

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
                logger.error(f"Error retrieving file: {e}")

    return render(request, 'upload_pdf.html', {
        'uploaded_files': uploaded_files,
        'answer': answer,
        'question': question
    })
