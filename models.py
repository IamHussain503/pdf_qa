from django.apps import AppConfig
from django.db import models


class HomeConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'home'


# Create your models here.

class UploadedDocument(models.Model):
    file_name = models.CharField(max_length=255)  # Store the file name
    vector_store_id = models.CharField(max_length=255)  # Store the vector store ID
