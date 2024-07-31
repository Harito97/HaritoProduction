import sys

sys.path.append("/Data/Projects/HaritoProduction")
from backend.app.H97_ThyroidCancer import (
    H97_ThyroidCancer,
    tensor_to_list,
    numpy_array_to_base64,
)
from backend.app.ChatBot import LLM_Model

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import tempfile
import os
import torch

# from torchvision import transforms
import json

# import numpy as np
# import base64
# from io import BytesIO


# Initialize the model instance
app_thyroid_cancer = H97_ThyroidCancer()
app_chat_bot = LLM_Model()

output_model3 = None
output_model2 = None
predicted = None


@csrf_exempt
def post_upload_thyroid_cancer_image(request):
    global output_model3, output_model2, predicted
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
            ) = app_thyroid_cancer.main(image_dir=temp_file_path)

            print("App_thyroid_cancer.main() done")

            predicted = tensor_to_list(predicted)
            output_model2 = tensor_to_list(output_model2)
            output_model3 = tensor_to_list(output_model3)

            print("Convert tensor to list done")

            # Ensure that all data is correctly processed and converted
            response = {
                "predicted": predicted,  # Convert tensor to list
                "yolo_detect_image": numpy_array_to_base64(
                    yolo_detect_image
                ),  # Convert numpy array to Base64
                "patch_level_heatmap": numpy_array_to_base64(
                    patch_level_heatmap
                ),  # Convert numpy array to Base64
                "image_level_heatmap": numpy_array_to_base64(
                    image_level_heatmap
                ),  # Convert numpy array to Base64
                "output_model3": output_model3,  # Convert tensor to list
            }

            print("Convert numpy array to Base64 done")

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


@csrf_exempt
def chat_response(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            print(data)
            message = data.get("message")
            if not message:
                return JsonResponse({"error": "No message provided"}, status=400)

            added_info = (
                f"+Nhãn dự đoán cuối cùng bởi mô hình là nhãn số: {predicted}."
                + f"+3 xác suất cho 3 nhãn 0, 1, 2 hay B2, B5, B6 lần lượt là: {output_model3}."
                + f"+Tuy nhiên mô hình cũng đặc biệt chú ý tới 12 vùng được cắt theo lưới kích cỡ 256x256, 5 vùng mật độ tế bào đáng quan tâm nhất trong ảnh và 1 góc nhìn bao quát toàn ảnh mức 1028x768. 18 vùng này, mỗi vùng đánh giá 1 trong 3 nhãn bằng 1 giá trị và kết quả thu được như sau: [B2_patch1, B5_patch1, B6_patch1, ..., B2_patch12, B5_patch12, B6_patch12, B2_crop1, B5_crop1, B6_crop1, ..., B2_crop5, B5_crop5, B6_crop5, B2_wholeimage, B5_wholeimage1, B6_wholeimage] = {output_model2}"
            )

            # Process the chat message using your model
            response = app_chat_bot.generate_response(added_info, message)
            return JsonResponse({"response": response})
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)
        except Exception as e:
            return JsonResponse(
                {"error": f"Error processing chat message: {str(e)}"}, status=500
            )
    else:
        return JsonResponse({"error": "Invalid request method"}, status=400)
