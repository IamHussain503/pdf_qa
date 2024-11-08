import os
import openai
import logging
import pandas as pd
from io import BytesIO, StringIO
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
collection = db['documents']
openai.api_key = os.getenv("OPENAI_API_KEY")


### Helper Function: Convert Excel to CSV and Store

def convert_excel_to_csv(excel_file):
    """Convert Excel file to CSV format and return as a string."""
    try:
        # Attempt to read the Excel file using pandas
        df = pd.read_excel(excel_file, engine="xlrd" if excel_file.name.endswith('.xls') else "openpyxl")
        logger.info("Excel file read successfully.")
        
        # Convert DataFrame to CSV format as a string
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        return csv_data
    except ValueError as e:
        logger.error(f"ValueError in converting Excel to CSV: {e}")
        return None
    except Exception as e:
        logger.error(f"Error converting Excel to CSV: {e}")
        return None



### Helper Function: Answer Questions Using CSV Data

def ask_question_with_csv_data(question, csv_data):
    """Answer a question using CSV data as context."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on CSV data."},
                {"role": "user", "content": f"{question} Based on the following CSV data: {csv_data}"}
            ],
            max_tokens=100,
            temperature=0.5
        )
        answer = response.choices[0].message['content'].strip()
        return answer
    except openai.error.OpenAIError as e:
        logger.error(f"Error answering question with CSV data: {e}")
        return "Error: Unable to answer the question."


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


### Helper Function: Store Full Text of Each Segment

def get_document_full_text(segments):
    """Combine the full text of each PDF segment."""
    full_text_segments = []
    for i, segment in enumerate(segments):
        try:
            segment_text = PdfReader(segment).pages[0].extract_text()
            if segment_text:
                full_text_segments.append(segment_text)
        except Exception as e:
            logger.error(f"Error processing segment {i}: {e}")

    # Combine the full text of all segments
    combined_full_text = " ".join(full_text_segments)
    return combined_full_text


### Helper Function for Answering PDF Questions (from Full Text)

def ask_question_with_full_text(question, full_text):
    """Answer a question using the full text of a PDF."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on full document text."},
                {"role": "user", "content": f"{question} Based on the following document text: {full_text}"}
            ],
            max_tokens=100,
            temperature=0.5
        )
        answer = response.choices[0].message['content'].strip()
        return answer
    except openai.error.OpenAIError as e:
        logger.error(f"Error answering question with full text: {e}")
        return "Error: Unable to answer the question."

class RetrieveDocumentsAPI(APIView):
    """API to retrieve all documents with their metadata."""

    def get(self, request):
        # Fetch all documents in the collection with relevant fields
        documents = list(collection.find(
            {},  # No filter to retrieve all documents
            {"file_name": 1, "upload_date": 1, "file_type": 1, "_id": 1}
        ))

        # Format each document with its metadata
        document_list = [
            {
                "document_id": str(doc["_id"]),
                "file_name": doc.get("file_name", ""),
                "upload_date": doc.get("upload_date", ""),
                "file_type": doc.get("file_type", "")
            }
            for doc in documents
        ]

        return Response({"documents": document_list}, status=status.HTTP_200_OK)





### API View: Upload Document, Handle Excel/CSV, and Store in MongoDB

class UploadDocumentAPI(APIView):
    """API to upload a document (PDF/Excel), convert and store in MongoDB."""

    def post(self, request):
        if 'file' not in request.FILES:
            return Response({"error": "No file uploaded."}, status=status.HTTP_400_BAD_REQUEST)

        file = request.FILES['file']
        file_name = file.name
        file_extension = file_name.split('.')[-1].lower()

        if file_extension == 'pdf':
            # Handle PDF file
            segments = split_pdf_into_segments(file)
            combined_full_text = get_document_full_text(segments)

            document = {
                "file_name": file_name,
                "file_type": "pdf",
                "upload_date": datetime.utcnow(),
                "full_text": combined_full_text
            }
            result = collection.insert_one(document)

            return Response({
                "file_name": file_name,
                "document_id": str(result.inserted_id),
                "full_text": combined_full_text
            }, status=status.HTTP_201_CREATED)

        elif file_extension in ['xls', 'xlsx']:
            # Handle Excel file
            csv_data = convert_excel_to_csv(file)
            if not csv_data:
                return Response({"error": "Error processing Excel file."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            document = {
                "file_name": file_name,
                "file_type": "excel",
                "upload_date": datetime.utcnow(),
                "csv_data": csv_data
            }
            result = collection.insert_one(document)

            return Response({
                "file_name": file_name,
                "document_id": str(result.inserted_id),
                "csv_data": csv_data
            }, status=status.HTTP_201_CREATED)

        else:
            return Response({"error": "Unsupported file type."}, status=status.HTTP_400_BAD_REQUEST)


### API View: Ask Question Using Document Content

class AskQuestionAPI(APIView):
    """API to answer questions based on document content (PDF or Excel) stored in MongoDB."""

    def post(self, request):
        question = request.data.get('question')
        document_id = request.data.get('document_id')

        if not question or not document_id:
            return Response({"error": "Both 'question' and 'document_id' are required."}, status=status.HTTP_400_BAD_REQUEST)

        # Retrieve the document from MongoDB
        document = collection.find_one({"_id": ObjectId(document_id)})
        if not document:
            return Response({"error": "Document not found."}, status=status.HTTP_404_NOT_FOUND)

        file_type = document.get("file_type")

        if file_type == "pdf":
            # Use full text for answering PDF questions
            full_text = document.get("full_text", "")
            if not full_text:
                return Response({"error": "No full text available for this document."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            answer = ask_question_with_full_text(question, full_text)
            return Response({"question": question, "answer": answer}, status=status.HTTP_200_OK)

        elif file_type == "excel":
            # Use CSV data for answering Excel questions
            csv_data = document.get("csv_data", "")
            if not csv_data:
                return Response({"error": "No CSV data available for this document."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            answer = ask_question_with_csv_data(question, csv_data)
            return Response({"question": question, "answer": answer}, status=status.HTTP_200_OK)

        else:
            return Response({"error": "Unsupported document type."}, status=status.HTTP_400_BAD_REQUEST)
