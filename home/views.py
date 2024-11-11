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
        logger.info(f"Requesting summary for vector store ID: {store_id}")
        try:
            # Attempt to generate a summary for the current segment
            summary = ask_question_with_file_search(broad_question, store_id)
            if summary:
                summaries.append(summary)
                logger.info(f"Received summary for vector store ID {store_id}: {summary[:100]}")  # Log first 100 chars
            else:
                logger.warning(f"No summary returned for vector store ID {store_id}.")
        except Exception as e:
            logger.error(f"Error generating summary for vector store ID {store_id}: {e}")

    # Combine individual summaries into a single comprehensive summary
    combined_summary = " ".join(summaries)
    if combined_summary:
        logger.info("Combined summary created successfully.")
    else:
        logger.error("Combined summary is empty after all segments were processed.")
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



def ask_question_with_file_search(question: str, vector_store_id: str):
    """Ask OpenAI for a summary of a document segment using the vector store."""
    try:
        # Modify the question to fit the new ChatCompletion format
        modified_question = f"{question} Please summarize the content relevant to this vector store ID."

        # Use the ChatCompletion endpoint with the "gpt-4-turbo" model
        logger.info(f"Sending request to OpenAI for vector store ID {vector_store_id} with question: {modified_question}")
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",  # or "gpt-4o" depending on your setup
            messages=[
                {"role": "system", "content": "You are an assistant that summarizes document segments."},
                {"role": "user", "content": modified_question}
            ],
            max_tokens=100
        )
        
        # Extract the response content
        summary = response.choices[0].message['content'].strip()
        logger.info(f"OpenAI response for vector store ID {vector_store_id}: {summary[:100]}")  # Log first 100 chars
        
        return summary

    except Exception as e:
        logger.error(f"Error during summary generation for vector store ID {vector_store_id}: {e}")
        return ""




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
        
        # Print all properties of the result
        for attr in dir(result):
            if not attr.startswith('_'):
                print(f"{attr}: {getattr(result, attr)}")

        

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
        document_id = request.data.get('document_id')

        if not question or not document_id:
            logger.error("Both 'question' and 'document_id' are required.")
            return Response({"error": "Both 'question' and 'document_id' are required."}, status=status.HTTP_400_BAD_REQUEST)

        # Retrieve the document metadata
        try:
            document = collection.find_one({"_id": ObjectId(document_id)})
        except Exception as e:
            logger.error(f"Error retrieving document from MongoDB: {e}")
            return Response({"error": "Error retrieving document from database."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        if not document:
            logger.error("Document not found.")
            return Response({"error": "Document not found."}, status=status.HTTP_404_NOT_FOUND)

        # Generate or retrieve the combined summary
        try:
            if "summary" not in document or not document["summary"]:
                logger.info(f"No existing summary found for document ID {document_id}. Generating summary.")

                # Generate the combined summary
                broad_question = "Summarize this document section."
                combined_summary = get_document_summary(document["vector_store_ids"], broad_question)

                # Check if the summary was successfully created
                if combined_summary:
                    # Update the document with the combined summary
                    collection.update_one({"_id": ObjectId(document_id)}, {"$set": {"summary": combined_summary}})
                    logger.info(f"Summary successfully generated and stored for document ID {document_id}.")
                else:
                    logger.error(f"Failed to generate summary for document ID {document_id}.")
                    return Response({"error": "Failed to generate summary for document."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            else:
                combined_summary = document["summary"]
                logger.info(f"Using existing summary for document ID {document_id}.")

        except Exception as e:
            logger.error(f"Error during summary generation or storage for document ID {document_id}: {e}")
            return Response({"error": "Error generating or storing summary."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Answer the question based on the summary
        try:
            answer = ask_question_with_summary(question, combined_summary)
            return Response({"question": question, "answer": answer}, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error during question processing for document ID {document_id}: {e}")
            return Response({"error": "Error during question processing."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)