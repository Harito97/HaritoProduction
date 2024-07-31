from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import your_model_module

@csrf_exempt
def post_upload_thyroid_cancer_image(request):
    if request.method == 'POST':
        # Get the image data from the request
        image_data = request.FILES['image']

        # Process the image using your model
        cell_cluster_image, patch_heatmap_image, image_heatmap_image, label_probabilities = your_model_module.process_thyroid_cancer_image(image_data)

        # Prepare the response
        response = {
            'cell_cluster_image': cell_cluster_image.tolist(),
            'patch_heatmap_image': patch_heatmap_image.tolist(),
            'image_heatmap_image': image_heatmap_image.tolist(),
            'label_probabilities': label_probabilities
        }

        return JsonResponse(response)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)