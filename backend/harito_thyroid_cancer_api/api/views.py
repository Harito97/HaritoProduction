import sys
sys.path.append('/Data/Projects/HaritoProduction')
from backend.app.H97_ThyroidCancer import H97_ThyroidCancer

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import tempfile
import os
from PIL import Image
import torch
from torchvision import transforms
import json
import numpy as np
import base64
from io import BytesIO


# Initialize the model instance
app = H97_ThyroidCancer()

def tensor_to_list(tensor):
    return tensor.cpu().detach().numpy().tolist()

def numpy_array_to_base64(numpy_array):
    # Ensure the numpy array is in an acceptable format for conversion
    if not isinstance(numpy_array, np.ndarray):
        raise ValueError("Input must be a numpy array")
    
    # Convert numpy array to PIL image
    image = Image.fromarray(numpy_array)

    # Convert PIL image to bytes
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()

    # Encode bytes to base64
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_base64

@csrf_exempt
def post_upload_thyroid_cancer_image(request):
    if request.method == "POST":
        if "image" not in request.FILES:
            return JsonResponse({"error": "No image file provided"}, status=400)

        image_file = request.FILES["image"]

        # Save the image to a temporary file named 'temp'
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".jpg", prefix="temp_"
            ) as temp_file:
                for chunk in image_file.chunks():
                    temp_file.write(chunk)
                temp_file_path = temp_file.name
        except Exception as e:
            return JsonResponse({"error": f"Error saving image: {str(e)}"}, status=500)

        # Process the image using your model
        try:
            (
                predicted,
                output_model3,
                output_model2,
                patch_level_heatmap,
                image_level_heatmap,
                yolo_detect_image,
            ) = app.main(image_dir=temp_file_path)

            # Ensure that all data is correctly processed and converted
            response = {
                "predicted": tensor_to_list(predicted),  # Convert tensor to list
                "yolo_detect_image": numpy_array_to_base64(
                    yolo_detect_image
                ),  # Convert numpy array to Base64
                "patch_level_heatmap": numpy_array_to_base64(
                    patch_level_heatmap
                ),  # Convert numpy array to Base64
                "image_level_heatmap": numpy_array_to_base64(
                    image_level_heatmap
                ),  # Convert numpy array to Base64
                "output_model3": tensor_to_list(
                    output_model3
                ),  # Convert tensor to list
            }

            return JsonResponse(response)
        except Exception as e:
            return JsonResponse(
                {"error": f"Error processing image: {str(e)}"}, status=500
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)  # Ensure the temporary file is cleaned up

    else:
        return JsonResponse({"error": "Invalid request method"}, status=400)
