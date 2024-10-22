"""
URL configuration for zainwork project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
# from django.contrib import admin
# from django.urls import path
# from home import views

# urlpatterns = [
#     path('admin/', admin.site.urls),
#      path('upload_pdf/', views.upload_pdf_page, name='upload_pdf_page'),  # Without vector_store_id
#      path('upload_pdf/upload/', views.upload_pdf, name='upload_pdf'),  # To handle the upload
#      path('ask_question/', views.ask_question, name='ask_question'),  # To handle question submission
#      path('upload_pdf/<str:vector_store_id>/', views.upload_pdf_page, name='upload_pdf_page_with_id'),  # With vector_store_id
# ]

from django.contrib import admin
from django.urls import path
from home import views

from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),  # Admin panel URL
    path('api/upload_document/', views.UploadDocumentAPI.as_view(), name='upload_document'),  # API for document upload
    path('api/retrieve_documents/', views.RetrieveDocumentAPI.as_view(), name='retrieve_documents'),  # API for retrieving documents
    path('api/ask_question/', views.AskQuestionAPI.as_view(), name='ask_question_api'),  # API for asking questions
]

