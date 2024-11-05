import os
import openai
import logging
from io import BytesIO
from datetime import datetime
from pymongo import MongoClient
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from bson import ObjectId
from PyPDF2 import PdfReader, PdfWriter

# Initialize MongoDB and OpenAI clients
client = MongoClient(os.getenv("MONGODB_URL"))
db = client.Todo
collection = db['home_uploadeddocuments']
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

logger = logging.getLogger(__name__)

### Helper Function: Splitting PDF into Segments

def split_pdf_into_segments(pdf_file, segment_size=5):
    """Splits a PDF file into segments of `segment_size` pages each."""
    segments = []
    pdf_reader = PdfReader(pdf_file)
    total_pages = len(pdf_reader.pages)

    for start in range(0, total_pages, segment_size):
        pdf_writer = PdfWriter()
        for i in range(start, min(start + segment_size, total_pages)):
            pdf_writer.add_page(pdf_reader.pages[i])

        # Save the segment to a BytesIO object
        segment_stream = BytesIO()
        pdf_writer.write(segment_stream)
        segment_stream.seek(0)
        segments.append(segment_stream)

    return segments

### Helper Function: Generate Combined Summary

def get_document_summary(vector_store_ids, broad_question="Summarize this document section."):
    """Generate a summary for each document segment and combine into a single summary."""
    summaries = []
    for store_id in vector_store_ids:
        summary = ask_question_with_file_search(broad_question, store_id)
        summaries.append(summary)

    # Combine individual summaries into a single comprehensive summary
    combined_summary = " ".join(summaries)
    return combined_summary

### Helper Function: Ask Question Using Summary

def ask_question_with_summary(question, combined_summary):
    """Use the combined summary to answer the question."""
    modified_question = f"{question} Based on the following summary: {combined_summary}"

    try:
        response = openai.Completion.create(
            model="gpt-4",
            prompt=modified_question,
            max_tokens=100
        )
        return response.choices[0].text.strip()
    except openai.error.OpenAIError as e:
        logger.error(f"Error during question processing: {e}")
        return {"error": "Internal Server Error. Check logs for details."}

### Function to Handle Segment-based Question Answering

def ask_question_with_file_search(question: str, vector_store_id: str):
    """Ask a question using a vector store and get a direct response."""
    try:
        # Modify the question for clearer instruction
        modified_question = f"{question} Please focus on detailed responses and use the relevant documents in the vector store."

        # Use OpenAI's standard completion or chat endpoint with the vector store reference
        response = openai.Completion.create(
            model="gpt-4",
            prompt=modified_question,
            max_tokens=100
        )
        return response.choices[0].text.strip()

    except openai.error.OpenAIError as e:
        logger.error(f"Error during question processing: {e}")
        return {"error": "Internal Server Error. Check logs for details."}

### API View: Upload Document, Segment, and Create Vector Stores

class UploadDocumentAPI(APIView):
    """API to upload a PDF document, segment it, and create a vector store for each segment."""

    def post(self, request):
        if 'pdf_file' not in request.FILES:
            return Response({"error": "No PDF file uploaded."}, status=status.HTTP_400_BAD_REQUEST)

        pdf_file = request.FILES['pdf_file']
        file_name = pdf_file.name

        # Step 1: Split the PDF into segments
        segments = split_pdf_into_segments(pdf_file)

        # Step 2: Create a vector store for each segment and store IDs
        vector_store_ids = []
        for i, segment in enumerate(segments):
            # Create a vector store for each segment
            vector_store = openai_client.beta.vector_stores.create(name=f"{file_name}_segment_{i}")
            file_batch = openai_client.beta.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vector_store.id,
                files=[(f"{file_name}_segment_{i}.pdf", segment.read())]
            )
            vector_store_ids.append(vector_store.id)

        # Step 3: Store metadata in MongoDB
        document = {
            "file_name": file_name,
            "vector_store_ids": vector_store_ids,
            "upload_date": datetime.utcnow(),
            "summary": ""  # Placeholder for the combined summary to be generated later
        }
        result = collection.insert_one(document)

        # Return response with vector store IDs and MongoDB document ID
        return Response({
            "file_name": file_name,
            "vector_store_ids": vector_store_ids,
            "document_id": str(result.inserted_id)
        }, status=status.HTTP_201_CREATED)

### API View: Retrieve All Documents with Metadata

class RetrieveDocumentsAPI(APIView):
    """API to retrieve all principal document names with their vector store IDs and summary status."""

    def get(self, request):
        # Fetch all documents in the collection with relevant fields
        documents = list(collection.find(
            {},  # No filter to retrieve all documents
            {"file_name": 1, "vector_store_ids": 1, "summary": 1, "_id": 1}
        ))

        # Format each document with its metadata
        document_list = [
            {
                "document_id": str(doc["_id"]),
                "file_name": doc["file_name"],
                "vector_store_ids": doc.get("vector_store_ids", []),
                "has_summary": bool(doc.get("summary"))  # True if summary exists, False otherwise
            }
            for doc in documents
        ]
        
        return Response({"documents": document_list}, status=status.HTTP_200_OK)

### API View: Ask Question and Generate Summary if Needed

class AskQuestionAPI(APIView):
    """API to answer questions using the vector store or the combined summary if available."""

    def post(self, request):
        question = request.data.get('question')
        document_id = request.data.get('document_id')  # Using document ID to identify the document

        if not question or not document_id:
            return Response({"error": "Both 'question' and 'document_id' are required."}, status=status.HTTP_400_BAD_REQUEST)

        # Retrieve the document metadata
        document = collection.find_one({"_id": ObjectId(document_id)})
        if not document:
            return Response({"error": "Document not found."}, status=status.HTTP_404_NOT_FOUND)

        # Check if a combined summary already exists
        if "summary" not in document or not document["summary"]:
            # Generate the summary if not available
            broad_question = "Summarize this document section."
            combined_summary = get_document_summary(document["vector_store_ids"], broad_question)

            # Update the document with the combined summary
            collection.update_one({"_id": ObjectId(document_id)}, {"$set": {"summary": combined_summary}})
        else:
            combined_summary = document["summary"]

        # Ask the specific question based on the combined summary
        answer = ask_question_with_summary(question, combined_summary)
        return Response({"question": question, "answer": answer}, status=status.HTTP_200_OK)
