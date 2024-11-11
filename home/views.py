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
logger.info("Django logging configuration test: Server started.")

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
    file_name = xlsx_file.name
    df = pd.read_excel(xlsx_file)
    
    if df.empty or df.columns.empty:
        logger.error("The uploaded XLSX file is empty or has no columns.")
        raise ValueError("The uploaded XLSX file is empty or has no columns.")
    
    for _, row in df.iterrows():
        data_text = ", ".join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])])
        embedding = get_embedding(data_text)
        document = {
            "file_name": file_name,
            "data_text": data_text,
            "embedding": embedding,
            "row_data": row.to_dict(),
            "upload_date": datetime.utcnow(),
            "file_type": "xlsx"
        }
        collection.insert_one(document)

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

# Parse question intent and filters
def parse_question(question):
    intents = {
        "count": ["how many", "total number", "count"],
        "sum": ["total amount", "sum of", "total"],
        "average": ["average", "mean"],
        "history": ["how many times", "history", "visit", "order"],
    }
    question_lower = question.lower()
    intent = None
    for key, keywords in intents.items():
        if any(keyword in question_lower for keyword in keywords):
            intent = key
            break
    filter_term = None
    if "black coffee" in question_lower:
        filter_term = "black coffee"
    return intent, filter_term

# Generate query based on intent and filter term
def generate_query(intent, filter_term, file_name=None):
    query = {}
    if file_name:
        query["file_name"] = file_name
    if filter_term:
        query["data_text"] = {"$regex": filter_term, "$options": "i"}
    aggregation = []
    if intent == "count":
        aggregation.append({"$match": query})
        aggregation.append({"$count": "total_count"})
    elif intent == "sum":
        aggregation.append({"$match": query})
        aggregation.append({"$group": {"_id": None, "total_amount": {"$sum": "$row_data.Order Value($)"}}})
    elif intent == "average":
        aggregation.append({"$match": query})
        aggregation.append({"$group": {"_id": None, "average_amount": {"$avg": "$row_data.Order Value($)"}}})
    elif intent == "history":
        aggregation.append({"$match": query})
        aggregation.append({"$group": {"_id": "$row_data.Customer Name", "visit_count": {"$sum": 1}}})
    return aggregation

def upload_pdf_page(request):
    """Frontend view to upload PDF/XLSX files and ask questions."""
    # Fetch all uploaded files from MongoDB
    uploaded_files = list(collection.find({}, {'_id': 1, 'file_name': 1, 'upload_date': 1, 'file_type': 1}))
    for file in uploaded_files:
        file['file_id'] = str(file['_id'])  # Convert ObjectId to string for easier handling

    answer = None
    question = None
    file_name = None

    if request.method == 'POST':
        if 'file' in request.FILES:
            # Process file upload
            files = request.FILES.getlist('file')
            for uploaded_file in files:
                try:
                    # Identify file type and store accordingly
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    if file_extension == 'pdf':
                        process_and_store_pdf(uploaded_file)
                    elif file_extension in ['xlsx', 'xls']:
                        process_and_store_xlsx(uploaded_file)
                    else:
                        logger.error(f"Unsupported file type: {file_extension}")
                        return render(request, 'upload_pdf.html', {
                            'uploaded_files': uploaded_files,
                            'answer': answer,
                            'question': question,
                            'error': f"Unsupported file type: {file_extension}"
                        })
                except Exception as e:
                    logger.error(f"Error processing file {uploaded_file.name}: {e}")
                    return render(request, 'upload_pdf.html', {
                        'uploaded_files': uploaded_files,
                        'answer': answer,
                        'question': question,
                        'error': f"Failed to process {uploaded_file.name}"
                    })

            # Reload the page after upload
            return redirect('upload_pdf_page')

        elif 'question' in request.POST:
            # Process question
            question = request.POST['question']
            file_name = request.POST.get('file_name')  # Optional: specify file to query against

            try:
                answer = ask_question(question, file_name)
            except Exception as e:
                logger.error(f"Error retrieving answer: {e}")
                answer = "An error occurred while processing your question."

    return render(request, 'upload_pdf.html', {
        'uploaded_files': uploaded_files,
        'answer': answer,
        'question': question
    })


# Ask question using direct query or OpenAI model
import logging
logger = logging.getLogger(__name__)

def ask_question(question, file_name=None):
    # Embed the question
    question_embedding = get_embedding(question)
    
    # Search for top-matching data chunks in the vector database
    query = {"file_name": file_name} if file_name else {}
    documents = list(collection.find(query, {"data_text": 1, "embedding": 1, "metadata": 1}))
    documents_with_similarity = [
        (doc, cosine_similarity(question_embedding, doc['embedding'])) for doc in documents
    ]
    documents_with_similarity.sort(key=lambda x: x[1], reverse=True)
    top_documents = [doc[0]['data_text'] for doc in documents_with_similarity[:3]]

    # Dynamically construct the prompt
    context = "\n\n".join(top_documents)
    prompt = f"Using the following context, answer the question accurately:\n\n{context}\n\nQuestion: {question}\nAnswer:"

    # Pass the prompt to GPT-4
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    answer = response.choices[0].message['content'].strip()

    return {"question": question, "answer": answer}



# Django API Views
class UploadDocumentAPI(APIView):
    def post(self, request):
        uploaded_files = request.FILES.getlist('file')
        if not uploaded_files:
            return Response({"error": "No files uploaded."}, status=status.HTTP_400_BAD_REQUEST)
        
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name

            # Check if a document with the same file name already exists
            if collection.find_one({"file_name": file_name}):
                return Response({"error": f"The document '{file_name}' already exists in the database."}, 
                                status=status.HTTP_400_BAD_REQUEST)
            
            # Proceed to process and store the file if it's not a duplicate
            file_extension = file_name.split('.')[-1].lower()
            try:
                if file_extension == 'pdf':
                    process_and_store_pdf(uploaded_file)
                elif file_extension in ['xlsx', 'xls']:
                    process_and_store_xlsx(uploaded_file)
                else:
                    logger.error(f"Unsupported file type: {file_extension}")
                    return Response({"error": f"Unsupported file type: {file_extension}"}, 
                                    status=status.HTTP_400_BAD_REQUEST)
            except Exception as e:
                logger.error(f"Error processing file {file_name}: {e}")
                return Response({"error": f"Failed to process {file_name}"}, 
                                status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({"status": "Files processed and data stored successfully"}, status=status.HTTP_201_CREATED)

class AskQuestionAPI(APIView):
    """API to answer questions using the vector store."""

    def post(self, request):
        question = request.data.get('question')
        file_name = request.data.get('file_name')  # Optional: specify the document name to limit search

        if not question:
            return Response({"error": "The 'question' is required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Process question with optional file name filter
            answer = ask_question(question, file_name)
            return Response({"question": question, "answer": answer}, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error while processing the question: {e}")
            return Response({"error": "Internal Server Error. Check logs for details."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class RetrieveDocumentsAPI(APIView):
    """API to retrieve all documents grouped by file name."""

    def get(self, request):
        # Fetch all documents with key fields, grouped by file name for easier identification
        documents = list(collection.find(
            {}, {"file_name": 1, "upload_date": 1, "file_type": 1, "_id": 1, "row_data": 1}
        ))

        # Organize documents by file name for structured output
        grouped_documents = {}
        for doc in documents:
            file_name = doc.get("file_name", "Unknown File")
            if file_name not in grouped_documents:
                grouped_documents[file_name] = []
            
            # Summary of each document row for easy reference
            doc_summary = {
                "document_id": str(doc["_id"]),
                "upload_date": doc.get("upload_date", ""),
                "file_type": doc.get("file_type", ""),
                "row_summary": {k: v for k, v in doc.get("row_data", {}).items() if k in ["Order id", "Order Value($)", "Customer Name", "Product name"]}
            }
            grouped_documents[file_name].append(doc_summary)

        # Structure response with document details grouped by file name
        response_data = [{"file_name": file_name, "documents": docs} for file_name, docs in grouped_documents.items()]
        return Response({"files": response_data}, status=status.HTTP_200_OK)