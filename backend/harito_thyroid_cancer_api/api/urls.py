from django.urls import path
from . import views

urlpatterns = [
    path('post-upload-thyroid-cancer-image', views.post_upload_thyroid_cancer_image, name='post_upload_thyroid_cancer_image'),
    path('chat-response', views.chat_response, name='chat_response'),
]