# import os
# import openai
# import logging
# from io import BytesIO
# from datetime import datetime
# from pymongo import MongoClient
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# from bson import ObjectId
# from PyPDF2 import PdfReader, PdfWriter

# # Setup logging
# logger = logging.getLogger(__name__)

# # Initialize MongoDB and OpenAI clients
# client = MongoClient(os.getenv("MONGODB_URL"))
# db = client.students
# collection = db['summarized_documents']
# openai.api_key = os.getenv("OPENAI_API_KEY")


# ### Helper Function: Splitting PDF into Segments

# def split_pdf_into_segments(pdf_file, segment_size=5):
#     """Splits a PDF file into segments of `segment_size` pages each."""
#     segments = []
#     pdf_reader = PdfReader(pdf_file)
#     total_pages = len(pdf_reader.pages)

#     for start in range(0, total_pages, segment_size):
#         pdf_writer = PdfWriter()
#         for i in range(start, min(start + segment_size, total_pages)):
#             pdf_writer.add_page(pdf_reader.pages[i])

#         # Save the segment to a BytesIO object
#         segment_stream = BytesIO()
#         pdf_writer.write(segment_stream)
#         segment_stream.seek(0)
#         segments.append(segment_stream)

#     return segments


# ### Helper Function: Summarize Each Segment

# def summarize_segment(segment_content):
#     """Generate a summary of a single segment using OpenAI's ChatCompletion API."""
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant that summarizes text."},
#                 {"role": "user", "content": f"Summarize the following text:\n\n{segment_content}"}
#             ],
#             max_tokens=100,
#             temperature=0.7
#         )
#         summary = response.choices[0].message['content'].strip()
#         return summary
#     except openai.error.OpenAIError as e:
#         logger.error(f"Error summarizing segment: {e}")
#         return "Error: Unable to summarize segment."


# ### Helper Function: Generate Combined Summary

# def get_document_summary(document_id, segments):
#     """Summarize each segment and combine them into a single summary."""
#     summaries = []
#     for i, segment in enumerate(segments):
#         try:
#             segment_text = PdfReader(segment).pages[0].extract_text()
#             summary = summarize_segment(segment_text)
#             if summary:
#                 summaries.append(summary)
#                 # Store individual segment summary in MongoDB
#                 collection.update_one({"_id": ObjectId(document_id)}, {"$push": {"segment_summaries": summary}})
#         except Exception as e:
#             logger.error(f"Error summarizing segment {i}: {e}")

#     # Combine individual summaries into a comprehensive summary
#     combined_summary = " ".join(summaries)
#     collection.update_one({"_id": ObjectId(document_id)}, {"$set": {"summary": combined_summary}})
#     return combined_summary


# ### Helper Function: Answer Questions Based on Summary

# def ask_question_with_summary(question, combined_summary):
#     """Answer a question using the combined summary."""
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant that answers questions based on a summary."},
#                 {"role": "user", "content": f"{question} Based on the following summary: {combined_summary}"}
#             ],
#             max_tokens=100,
#             temperature=0.5
#         )
#         answer = response.choices[0].message['content'].strip()
#         return answer
#     except openai.error.OpenAIError as e:
#         logger.error(f"Error answering question: {e}")
#         return "Error: Unable to answer the question."


# ### API View: Upload Document, Segment, and Summarize

# class UploadDocumentAPI(APIView):
#     """API to upload a PDF document, segment it, summarize each segment, and store metadata in MongoDB."""

#     def post(self, request):
#         if 'pdf_file' not in request.FILES:
#             return Response({"error": "No PDF file uploaded."}, status=status.HTTP_400_BAD_REQUEST)

#         pdf_file = request.FILES['pdf_file']
#         file_name = pdf_file.name

#         # Step 1: Split the PDF into segments
#         segments = split_pdf_into_segments(pdf_file)

#         # Step 2: Insert document metadata in MongoDB without summaries
#         document = {
#             "file_name": file_name,
#             "upload_date": datetime.utcnow(),
#             "segment_summaries": [],
#             "summary": ""  # Placeholder for the combined summary
#         }
#         result = collection.insert_one(document)

#         # Step 3: Generate summaries for each segment and combine them
#         combined_summary = get_document_summary(result.inserted_id, segments)

#         # Step 4: Return response with MongoDB document ID
#         return Response({
#             "file_name": file_name,
#             "document_id": str(result.inserted_id),
#             "summary": combined_summary
#         }, status=status.HTTP_201_CREATED)


# ### API View: Retrieve All Documents with Metadata

# class RetrieveDocumentsAPI(APIView):
#     """API to retrieve all documents with their metadata, including vector store IDs and summary status."""

#     def get(self, request):
#         # Fetch all documents in the collection with relevant fields
#         documents = list(collection.find(
#             {},  # No filter to retrieve all documents
#             {"file_name": 1, "upload_date": 1, "summary": 1, "_id": 1}
#         ))

#         # Format each document with its metadata
#         document_list = [
#             {
#                 "document_id": str(doc["_id"]),
#                 "file_name": doc.get("file_name", ""),
#                 "upload_date": doc.get("upload_date", ""),
#                 "has_summary": bool(doc.get("summary"))  # True if summary exists, False otherwise
#             }
#             for doc in documents
#         ]

#         return Response({"documents": document_list}, status=status.HTTP_200_OK)


# ### API View: Ask Question Using Document Summary

# class AskQuestionAPI(APIView):
#     """API to answer questions using the combined summary stored in MongoDB."""

#     def post(self, request):
#         question = request.data.get('question')
#         document_id = request.data.get('document_id')

#         if not question or not document_id:
#             return Response({"error": "Both 'question' and 'document_id' are required."}, status=status.HTTP_400_BAD_REQUEST)

#         # Retrieve the document and its summary
#         document = collection.find_one({"_id": ObjectId(document_id)})
#         if not document:
#             return Response({"error": "Document not found."}, status=status.HTTP_404_NOT_FOUND)

#         combined_summary = document.get("summary", "")
#         if not combined_summary:
#             return Response({"error": "No summary available for this document."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

#         # Answer the question based on the summary
#         answer = ask_question_with_summary(question, combined_summary)
#         return Response({"question": question, "answer": answer}, status=status.HTTP_200_OK)



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

def get_document_full_text(document_id, segments):
    """Store the full text of each segment in MongoDB without summarizing."""
    full_text_segments = []
    for i, segment in enumerate(segments):
        try:
            segment_text = PdfReader(segment).pages[0].extract_text()
            if segment_text:
                full_text_segments.append(segment_text)
                # Store individual segment full text in MongoDB
                collection.update_one({"_id": ObjectId(document_id)}, {"$push": {"segment_texts": segment_text}})
        except Exception as e:
            logger.error(f"Error processing segment {i}: {e}")

    # Combine the full text of all segments
    combined_full_text = " ".join(full_text_segments)
    collection.update_one({"_id": ObjectId(document_id)}, {"$set": {"full_text": combined_full_text}})
    return combined_full_text


### Helper Function: Answer Questions Using Full Text

def ask_question_with_full_text(question, full_text):
    """Answer a question using the combined full text."""
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
        logger.error(f"Error answering question: {e}")
        return "Error: Unable to answer the question."


### API View: Upload Document, Segment, and Store Full Text

class UploadDocumentAPI(APIView):
    """API to upload a PDF document, segment it, store each segment's full text, and store metadata in MongoDB."""

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
            "segment_texts": [],  # Placeholder for individual segment full text
            "full_text": ""       # Placeholder for the combined full text
        }
        result = collection.insert_one(document)

        # Step 3: Store full text for each segment and combine them
        combined_full_text = get_document_full_text(result.inserted_id, segments)

        # Step 4: Return response with MongoDB document ID
        return Response({
            "file_name": file_name,
            "document_id": str(result.inserted_id),
            "full_text": combined_full_text
        }, status=status.HTTP_201_CREATED)


### API View: Retrieve All Documents with Metadata

class RetrieveDocumentsAPI(APIView):
    """API to retrieve all documents with their metadata, including whether full text is available."""

    def get(self, request):
        # Fetch all documents in the collection with relevant fields
        documents = list(collection.find(
            {},  # No filter to retrieve all documents
            {"file_name": 1, "upload_date": 1, "full_text": 1, "_id": 1}
        ))

        # Format each document with its metadata
        document_list = [
            {
                "document_id": str(doc["_id"]),
                "file_name": doc.get("file_name", ""),
                "upload_date": doc.get("upload_date", ""),
                "has_full_text": bool(doc.get("full_text"))  # True if full text exists, False otherwise
            }
            for doc in documents
        ]

        return Response({"documents": document_list}, status=status.HTTP_200_OK)


### API View: Ask Question Using Document Full Text

class AskQuestionAPI(APIView):
    """API to answer questions using the combined full text stored in MongoDB."""

    def post(self, request):
        question = request.data.get('question')
        document_id = request.data.get('document_id')

        if not question or not document_id:
            return Response({"error": "Both 'question' and 'document_id' are required."}, status=status.HTTP_400_BAD_REQUEST)

        # Retrieve the document and its full text
        document = collection.find_one({"_id": ObjectId(document_id)})
        if not document:
            return Response({"error": "Document not found."}, status=status.HTTP_404_NOT_FOUND)

        full_text = document.get("full_text", "")
        if not full_text:
            return Response({"error": "No full text available for this document."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Answer the question based on the full text
        answer = ask_question_with_full_text(question, full_text)
        return Response({"question": question, "answer": answer}, status=status.HTTP_200_OK)

