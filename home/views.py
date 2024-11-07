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

# Setup logging
logger = logging.getLogger(__name__)

# Initialize MongoDB and OpenAI clients
client = MongoClient(os.getenv("MONGODB_URL"))
db = client.students
collection = db['summarized_documents']
openai.api_key = os.getenv("OPENAI_API_KEY")


### Helper Function: Splitting PDF into Segments

def split_pdf_into_segments(pdf_file, segment_size=40):
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


### Helper Function: Summarize Each Segment

def summarize_segment(segment_content):
    """Generate a summary of a single segment using OpenAI's API."""
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Summarize the following text:\n\n{segment_content}",
            max_tokens=100,
            temperature=0.7
        )
        summary = response.choices[0].text.strip()
        return summary
    except openai.error.OpenAIError as e:
        logger.error(f"Error summarizing segment: {e}")
        return "Error: Unable to summarize segment."


### Helper Function: Generate Combined Summary

def get_document_summary(document_id, segments):
    """Summarize each segment and combine them into a single summary."""
    summaries = []
    for i, segment in enumerate(segments):
        try:
            segment_text = PdfReader(segment).pages[0].extract_text()
            summary = summarize_segment(segment_text)
            if summary:
                summaries.append(summary)
                # Store individual segment summary in MongoDB
                collection.update_one({"_id": ObjectId(document_id)}, {"$push": {"segment_summaries": summary}})
        except Exception as e:
            logger.error(f"Error summarizing segment {i}: {e}")

    # Combine individual summaries into a comprehensive summary
    combined_summary = " ".join(summaries)
    collection.update_one({"_id": ObjectId(document_id)}, {"$set": {"summary": combined_summary}})
    return combined_summary


### Helper Function: Answer Questions Based on Summary

def ask_question_with_summary(question, combined_summary):
    """Answer a question using the combined summary."""
    modified_question = f"{question} Based on the following summary: {combined_summary}"
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=modified_question,
            max_tokens=100,
            temperature=0.5
        )
        return response.choices[0].text.strip()
    except openai.error.OpenAIError as e:
        logger.error(f"Error answering question: {e}")
        return "Error: Unable to answer the question."


### API View: Upload Document, Segment, and Summarize

class UploadDocumentAPI(APIView):
    """API to upload a PDF document, segment it, summarize each segment, and store metadata in MongoDB."""

    def post(self, request):
        if 'pdf_file' not in request.FILES:
            return Response({"error": "No PDF file uploaded."}, status=status.HTTP_400_BAD_REQUEST)

        pdf_file = request.FILES['pdf_file']
        file_name = pdf_file.name

        # Step 1: Split the PDF into segments
        segments = split_pdf_into_segments(pdf_file)

        # Step 2: Insert document metadata in MongoDB without summaries
        document = {
            "file_name": file_name,
            "upload_date": datetime.utcnow(),
            "segment_summaries": [],
            "summary": ""  # Placeholder for the combined summary
        }
        result = collection.insert_one(document)

        # Step 3: Generate summaries for each segment and combine them
        combined_summary = get_document_summary(result.inserted_id, segments)

        # Step 4: Return response with MongoDB document ID
        return Response({
            "file_name": file_name,
            "document_id": str(result.inserted_id),
            "summary": combined_summary
        }, status=status.HTTP_201_CREATED)


### API View: Retrieve All Documents with Metadata

class RetrieveDocumentsAPI(APIView):
    """API to retrieve all documents with their metadata, including vector store IDs and summary status."""

    def get(self, request):
        # Fetch all documents in the collection with relevant fields
        documents = list(collection.find(
            {},  # No filter to retrieve all documents
            {"file_name": 1, "upload_date": 1, "summary": 1, "_id": 1}
        ))

        # Format each document with its metadata
        document_list = [
            {
                "document_id": str(doc["_id"]),
                "file_name": doc.get("file_name", ""),
                "upload_date": doc.get("upload_date", ""),
                "has_summary": bool(doc.get("summary"))  # True if summary exists, False otherwise
            }
            for doc in documents
        ]

        return Response({"documents": document_list}, status=status.HTTP_200_OK)


### API View: Ask Question Using Document Summary

class AskQuestionAPI(APIView):
    """API to answer questions using the combined summary stored in MongoDB."""

    def post(self, request):
        question = request.data.get('question')
        document_id = request.data.get('document_id')

        if not question or not document_id:
            return Response({"error": "Both 'question' and 'document_id' are required."}, status=status.HTTP_400_BAD_REQUEST)

        # Retrieve the document and its summary
        document = collection.find_one({"_id": ObjectId(document_id)})
        if not document:
            return Response({"error": "Document not found."}, status=status.HTTP_404_NOT_FOUND)

        combined_summary = document.get("summary", "")
        if not combined_summary:
            return Response({"error": "No summary available for this document."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Answer the question based on the summary
        answer = ask_question_with_summary(question, combined_summary)
        return Response({"question": question, "answer": answer}, status=status.HTTP_200_OK)
