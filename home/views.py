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


### Helper Function: Convert Excel to CSV Chunks and Store

def convert_excel_to_csv_chunks(excel_file, chunk_size=50):
    """Convert Excel file to CSV format and split it into chunks."""
    try:
        # Read the Excel file into a DataFrame
        df = pd.read_excel(excel_file)
        csv_chunks = []

        # Split DataFrame into chunks
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size]
            csv_buffer = StringIO()
            chunk.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            csv_chunks.append(csv_data)
        
        return csv_chunks
    except Exception as e:
        logger.error(f"Error converting Excel to CSV chunks: {e}")
        return None


### Helper Function: Summarize CSV Chunks

def summarize_csv_chunks(csv_chunks):
    """Summarize each CSV chunk to reduce details."""
    summaries = []
    for chunk in csv_chunks:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Summarize the following CSV data."},
                    {"role": "user", "content": chunk}
                ],
                max_tokens=500,
                temperature=0.5
            )
            summary = response.choices[0].message['content'].strip()
            summaries.append(summary)
        except openai.error.OpenAIError as e:
            logger.error(f"Error summarizing CSV data chunk: {e}")
            return "Error: Unable to summarize the data."
    
    # Combine all summaries into a single summary text
    combined_summary = " ".join(summaries)
    return combined_summary


### Helper Function: Answer Questions Using the Combined Summary

def answer_question_from_summary(question, summary_text):
    """Answer a question using the combined summary text."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Answer the question based on the summary provided."},
                {"role": "user", "content": f"{question} Based on the following summary: {summary_text}"}
            ],
            max_tokens=500,
            temperature=0.5
        )
        answer = response.choices[0].message['content'].strip()
        return answer
    except openai.error.OpenAIError as e:
        logger.error(f"Error answering question from summary: {e}")
        return "Error: Unable to answer the question."


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
            # Handle Excel file and convert to CSV chunks
            csv_chunks = convert_excel_to_csv_chunks(file)
            if not csv_chunks:
                return Response({"error": "Error processing Excel file."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            document = {
                "file_name": file_name,
                "file_type": "excel",
                "upload_date": datetime.utcnow(),
                "csv_chunks": csv_chunks  # Store the CSV data in chunks
            }
            result = collection.insert_one(document)

            return Response({
                "file_name": file_name,
                "document_id": str(result.inserted_id),
                "csv_chunks": csv_chunks
            }, status=status.HTTP_201_CREATED)

        else:
            return Response({"error": "Unsupported file type."}, status=status.HTTP_400_BAD_REQUEST)


### API View: Ask Question Using Document Content with Summary-Based Approach

def search_csv_chunks_for_answer(question, csv_chunks):
    """Directly search each CSV chunk for the answer to the question."""
    for chunk in csv_chunks:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Answer the question based on the following CSV data."},
                    {"role": "user", "content": f"{question} Based on this data:\n{chunk}"}
                ],
                max_tokens=500,
                temperature=0.5
            )
            answer = response.choices[0].message['content'].strip()
            # Check if the response is relevant or not generic
            if "no specific details" not in answer.lower():
                return answer
        except openai.error.OpenAIError as e:
            logger.error(f"Error searching CSV data chunk: {e}")
            return "Error: Unable to search the data."
    
    # Default response if no answer is found
    return "The specific details could not be found in the provided CSV data."


### API View: Ask Question Using Document Content without Summarization

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
            # Directly search each CSV chunk for the answer to the question
            csv_chunks = document.get("csv_chunks", [])
            if not csv_chunks:
                return Response({"error": "No CSV data available for this document."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Search each chunk directly for an answer
            answer = search_csv_chunks_for_answer(question, csv_chunks)
            return Response({"question": question, "answer": answer}, status=status.HTTP_200_OK)

        else:
            return Response({"error": "Unsupported document type."}, status=status.HTTP_400_BAD_REQUEST)



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
            max_tokens=500,
            temperature=0.5
        )
        answer = response.choices[0].message['content'].strip()
        return answer
    except openai.error.OpenAIError as e:
        logger.error(f"Error answering question with full text: {e}")
        return "Error: Unable to answer the question."


### API View: Retrieve All Documents with Metadata

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
