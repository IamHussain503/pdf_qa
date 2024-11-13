# import os
# import openai
# import logging
# from django.shortcuts import render, redirect
# from django.http import JsonResponse
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# from pymongo import MongoClient
# from openai.embeddings_utils import cosine_similarity
# import numpy as np
# from datetime import datetime

# # Setup logging
# logger = logging.getLogger(__name__)

# # MongoDB and OpenAI setup
# client = MongoClient(os.getenv("MONGODB_URL"))
# db = client.YourDatabaseName
# collection = db['UploadedDocuments']

# openai.api_key = os.getenv("OPENAI_API_KEY")

# # Helper function to create embeddings
# def get_embedding(text):
#     response = openai.Embedding.create(
#         model="text-embedding-ada-002",
#         input=text
#     )
#     return response['data'][0]['embedding']

# # Function to create or update the vector database in MongoDB
# def upload_file_and_create_vector_store(pdf_files):
#     vector_store_data = []
#     for pdf_file in pdf_files:
#         try:
#             # Extract text from PDF
#             text_content = extract_text_from_pdf(pdf_file)
            
#             # Generate embeddings for document text
#             embeddings = get_embedding(text_content)
            
#             # Store the embeddings in MongoDB
#             document = {
#                 "file_name": pdf_file.name,
#                 "content": text_content,
#                 "embedding": embeddings,
#                 "upload_date": datetime.utcnow(),
#                 "file_type": "pdf"
#             }
#             collection.insert_one(document)
#             vector_store_data.append(document)
        
#         except Exception as e:
#             logger.error(f"Error processing file {pdf_file.name}: {e}")
#             return {"error": f"Failed to process {pdf_file.name}"}
#     return vector_store_data


# # Function to extract text from PDF
# def extract_text_from_pdf(pdf_file):
#     from PyPDF2 import PdfReader
#     reader = PdfReader(pdf_file)
#     text = ""
#     for page in reader.pages:
#         text += page.extract_text() or ""
#     return text

# # Function to find similar documents based on query
# def find_similar_documents(query):
#     query_embedding = get_embedding(query)
    
#     # Retrieve all documents from MongoDB
#     documents = list(collection.find({}, {"file_name": 1, "embedding": 1, "content": 1}))
#     similarities = []
    
#     for doc in documents:
#         doc_embedding = doc['embedding']
#         similarity_score = cosine_similarity(np.array(query_embedding), np.array(doc_embedding))
#         similarities.append((doc, similarity_score))
    
#     # Sort documents by similarity score in descending order
#     similarities.sort(key=lambda x: x[1], reverse=True)
    
#     # Return top 3 documents
#     top_documents = [doc for doc, _ in similarities[:3]]
#     return top_documents

# # API to answer a question using the vector store
# def ask_question(question):
#     top_docs = find_similar_documents(question)
#     context = "\n\n".join([doc["content"] for doc in top_docs])

#     # Prepare the chat prompt for gpt-4o
#     messages = [
#         {"role": "system", "content": "You are an assistant that answers questions based on provided documents."},
#         {"role": "user", "content": f"Here is some context:\n\n{context}\n\nNow, answer the following question:\n{question}"}
#     ]
    
#     # Call the chat completion endpoint
#     response = openai.ChatCompletion.create(
#         model="gpt-4o",
#         messages=messages,
#         max_tokens=8192,
#     )
    
#     # Return the assistant's reply
#     return response['choices'][0]['message']['content'].strip()


# # Django API Views
# class UploadDocumentAPI(APIView):
#     def post(self, request):
#         if 'pdf_files' not in request.FILES:
#             return Response({"error": "No PDF files uploaded."}, status=status.HTTP_400_BAD_REQUEST)

#         pdf_files = request.FILES.getlist('pdf_files')
#         vector_store_data = upload_file_and_create_vector_store(pdf_files)

#         file_names = [data["file_name"] for data in vector_store_data]
        
#         return Response({
#             "file_names": file_names,
#             "status": "Files uploaded successfully"
#         }, status=status.HTTP_201_CREATED)


# class AskQuestionAPI(APIView):
#     """API to answer questions using the vector store."""

#     def post(self, request):
#         question = request.data.get('question')
#         if not question:
#             return Response({"error": "The 'question' is required."}, status=status.HTTP_400_BAD_REQUEST)

#         try:
#             answer = ask_question(question)
#             return Response({"question": question, "answer": answer}, status=status.HTTP_200_OK)

#         except Exception as e:
#             logger.error(f"Error while processing the question: {e}")
#             return Response({"error": "Internal Server Error. Check logs for details."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# class RetrieveDocumentsAPI(APIView):
#     """API to retrieve all documents with their metadata."""

#     def get(self, request):
#         # Fetch all documents in the collection with relevant fields
#         documents = list(collection.find(
#             {},  # No filter to retrieve all documents
#             {"file_name": 1, "upload_date": 1, "file_type": 1, "_id": 1}
#         ))

#         # Format each document with its metadata
#         document_list = [
#             {
#                 "document_id": str(doc["_id"]),
#                 "file_name": doc.get("file_name", ""),
#                 "upload_date": doc.get("upload_date", ""),
#                 "file_type": doc.get("file_type", "")
#             }
#             for doc in documents
#         ]

#         return Response({"documents": document_list}, status=status.HTTP_200_OK)

# def upload_pdf_page(request):
#     """Frontend view to upload PDF and ask questions."""
#     uploaded_files = list(collection.find({}, {'_id': 1, 'file_name': 1}))
#     for file in uploaded_files:
#         file['file_id'] = str(file['_id'])

#     answer = None
#     question = None

#     if request.method == 'POST':
#         if 'pdf_files' in request.FILES:
#             pdf_files = request.FILES.getlist('pdf_files')
#             upload_file_and_create_vector_store(pdf_files)
#             return redirect('upload_pdf_page')

#         elif 'question' in request.POST:
#             question = request.POST['question']
#             answer = ask_question(question)

#     return render(request, 'upload_pdf.html', {
#         'uploaded_files': uploaded_files,
#         'answer': answer,
#         'question': question
#     })



import os
import openai
import logging
from django.shortcuts import render, redirect
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from pymongo import MongoClient
from openai.embeddings_utils import cosine_similarity
import numpy as np
from datetime import datetime
import pandas as pd
from PyPDF2 import PdfReader

# Setup logging
logger = logging.getLogger(__name__)

# MongoDB and OpenAI setup
client = MongoClient(os.getenv("MONGODB_URL"))
db = client.YourDatabaseName
collection = db['UploadedDocuments']

openai.api_key = os.getenv("OPENAI_API_KEY")

# Helper function to create embeddings
def get_embedding(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response['data'][0]['embedding']

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Function to process and store data from an XLSX file
def process_and_store_xlsx(xlsx_file):
    df = pd.read_excel(xlsx_file)
    
    # Check if the file has any columns
    if df.empty or df.columns.empty:
        logger.error("The uploaded XLSX file has no columns.")
        raise ValueError("The uploaded XLSX file is empty or has no columns.")

    for _, row in df.iterrows():
        # Create a dynamic text summary using all columns
        data_text = ", ".join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])])

        # Generate embedding for the row data
        embedding = get_embedding(data_text)

        # Store the row data and embedding in MongoDB
        document = {
            "data_text": data_text,
            "embedding": embedding,
            "row_data": row.to_dict(),  # Store the full row data for reference
            "upload_date": datetime.utcnow(),
            "file_type": "xlsx"
        }
        collection.insert_one(document)

    print("XLSX data processed and stored successfully.")


# Function to process and store data from a PDF file
def process_and_store_pdf(pdf_file):
    text_content = extract_text_from_pdf(pdf_file)
    embedding = get_embedding(text_content)

    document = {
        "file_name": pdf_file.name,
        "content": text_content,
        "embedding": embedding,
        "upload_date": datetime.utcnow(),
        "file_type": "pdf"
    }
    collection.insert_one(document)

    print("PDF data processed and stored successfully.")

# Django API View to handle both XLSX and PDF file uploads
class UploadDocumentAPI(APIView):
    """API to upload XLSX or PDF documents and create embeddings for their content."""

    def post(self, request):
        # Check for uploaded files
        uploaded_files = request.FILES.getlist('file')
        if not uploaded_files:
            return Response({"error": "No files uploaded."}, status=status.HTTP_400_BAD_REQUEST)

        # Process each uploaded file based on its extension
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            file_extension = file_name.split('.')[-1].lower()

            try:
                if file_extension == 'pdf':
                    # Process PDF file
                    process_and_store_pdf(uploaded_file)
                elif file_extension in ['xlsx', 'xls']:
                    # Process XLSX or XLS file
                    process_and_store_xlsx(uploaded_file)
                else:
                    logger.error(f"Unsupported file type: {file_extension}")
                    return Response({"error": f"Unsupported file type: {file_extension}"}, status=status.HTTP_400_BAD_REQUEST)

            except Exception as e:
                logger.error(f"Error processing file {file_name}: {e}")
                return Response({"error": f"Failed to process {file_name}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({
            "status": "Files processed and data stored successfully"
        }, status=status.HTTP_201_CREATED)

# Function to find similar documents based on query
def find_similar_documents(query):
    query_embedding = get_embedding(query)
    
    # Retrieve all documents from MongoDB
    documents = list(collection.find({}, {"file_name": 1, "embedding": 1, "content": 1}))
    similarities = []
    
    for doc in documents:
        doc_embedding = doc['embedding']
        similarity_score = cosine_similarity(np.array(query_embedding), np.array(doc_embedding))
        similarities.append((doc, similarity_score))
    
    # Sort documents by similarity score in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top 3 documents
    top_documents = [doc for doc, _ in similarities[:3]]
    return top_documents

# API to answer a question using the vector store
def ask_question(question):
    top_docs = find_similar_documents(question)
    context = "\n\n".join([doc["content"] for doc in top_docs])

    # Prepare the chat prompt for gpt-4o
    messages = [
        {"role": "system", "content": "You are an assistant that answers questions based on provided documents."},
        {"role": "user", "content": f"Here is some context:\n\n{context}\n\nNow, answer the following question:\n{question}"}
    ]
    
    # Call the chat completion endpoint
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=8192,
    )
    
    # Return the assistant's reply
    return response['choices'][0]['message']['content'].strip()

# Django API Views
class AskQuestionAPI(APIView):
    """API to answer questions using the vector store."""

    def post(self, request):
        question = request.data.get('question')
        if not question:
            return Response({"error": "The 'question' is required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            answer = ask_question(question)
            return Response({"question": question, "answer": answer}, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error while processing the question: {e}")
            return Response({"error": "Internal Server Error. Check logs for details."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

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

def upload_pdf_page(request):
    """Frontend view to upload PDF and ask questions."""
    uploaded_files = list(collection.find({}, {'_id': 1, 'file_name': 1}))
    for file in uploaded_files:
        file['file_id'] = str(file['_id'])

    answer = None
    question = None

    if request.method == 'POST':
        if 'pdf_files' in request.FILES:
            pdf_files = request.FILES.getlist('pdf_files')
            upload_file_and_create_vector_store(pdf_files)
            return redirect('upload_pdf_page')

        elif 'question' in request.POST:
            question = request.POST['question']
            answer = ask_question(question)

    return render(request, 'upload_pdf.html', {
        'uploaded_files': uploaded_files,
        'answer': answer,
        'question': question
    })
