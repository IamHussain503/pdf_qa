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

logger = logging.getLogger(__name__)

# Initialize MongoDB and OpenAI clients
client = MongoClient(os.getenv("MONGODB_URL"))
db = client.students
collection = db['summarized_documents']
openai.api_key = os.getenv("OPENAI_API_KEY")

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

### Helper Function: Generate Summary for Each Segment
def summarize_segment(segment_text):
    """Generate a summary for a single segment using OpenAI API."""
    prompt = f"Summarize the following text:\n\n{segment_text[:1500]}"
    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo",
            prompt=prompt,
            max_tokens=150
        )
        summary = response.choices[0].text.strip()
        return summary
    except Exception as e:
        logger.error(f"Error summarizing segment: {e}")
        return None

### Helper Function: Combine Summaries into a Single Summary
def get_document_summary(segments):
    """Generate a summary for each document segment and combine into a single summary."""
    summaries = []
    for i, segment in enumerate(segments):
        try:
            segment_text = segment.read().decode("utf-8")  # Decoding PDF segment to text
            summary = summarize_segment(segment_text)
            if summary:
                summaries.append(summary)
                logger.info(f"Summary for segment {i}: {summary[:100]}")  # Log first 100 chars
            else:
                logger.warning(f"No summary returned for segment {i}.")
        except Exception as e:
            logger.error(f"Error generating summary for segment {i}: {e}")

    combined_summary = " ".join(summaries)
    return combined_summary

### Helper Function: Ask Question Using Combined Summary
def ask_question_with_summary(question, combined_summary):
    """Use the combined summary to answer the question."""
    prompt = f"{question}\n\nBased on the following summary: {combined_summary}"
    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo",
            prompt=prompt,
            max_tokens=100
        )
        return response.choices[0].text.strip()
    except openai.error.OpenAIError as e:
        logger.error(f"Error during question processing: {e}")
        return {"error": "Internal Server Error. Check logs for details."}

### API View: Upload Document, Segment, and Summarize
class UploadDocumentAPI(APIView):
    """API to upload a PDF document, segment it, and summarize each part."""

    def post(self, request):
        if 'pdf_file' not in request.FILES:
            return Response({"error": "No PDF file uploaded."}, status=status.HTTP_400_BAD_REQUEST)

        pdf_file = request.FILES['pdf_file']
        file_name = pdf_file.name

        # Step 1: Split the PDF into segments
        segments = split_pdf_into_segments(pdf_file)

        # Step 2: Generate a combined summary from all segments
        combined_summary = get_document_summary(segments)

        # Step 3: Store metadata in MongoDB
        document = {
            "file_name": file_name,
            "upload_date": datetime.utcnow(),
            "summary": combined_summary
        }
        result = collection.insert_one(document)

        # Return response with MongoDB document ID
        return Response({
            "file_name": file_name,
            "document_id": str(result.inserted_id),
            "summary": combined_summary
        }, status=status.HTTP_201_CREATED)

### API View: Ask Question
class AskQuestionAPI(APIView):
    """API to answer questions based on the combined summary."""

    def post(self, request):
        question = request.data.get('question')
        document_id = request.data.get('document_id')

        if not question or not document_id:
            return Response({"error": "Both 'question' and 'document_id' are required."}, status=status.HTTP_400_BAD_REQUEST)

        # Retrieve the document metadata
        try:
            document = collection.find_one({"_id": ObjectId(document_id)})
            if not document:
                return Response({"error": "Document not found."}, status=status.HTTP_404_NOT_FOUND)
            combined_summary = document["summary"]
        except Exception as e:
            logger.error(f"Error retrieving document from MongoDB: {e}")
            return Response({"error": "Error retrieving document from database."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Answer the question based on the summary
        try:
            answer = ask_question_with_summary(question, combined_summary)
            return Response({"question": question, "answer": answer}, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error during question processing for document ID {document_id}: {e}")
            return Response({"error": "Error during question processing."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
