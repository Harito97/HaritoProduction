import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import numpy as np
import torchvision.transforms as transforms
import random


def plot_heatmap(heatmap_np, reversed=False):
    # heatmap_np = 255 * heatmap_np
    # plt.imshow(heatmap_np, cmap="jet", alpha=0.5)
    if heatmap_np.ndim == 4:
        heatmap_np = heatmap_np[0, :, :, :]
    if reversed:
        heatmap_np = heatmap_np[..., ::-1]  # Chuyển từ BGR sang RGB
    plt.imshow(heatmap_np)  # , cmap="jet")
    plt.axis("off")  # Tắt các trục để hiển thị rõ hơn
    plt.show()


def read_images(image_paths: list = [], resize=(224, 224)):
    if image_paths is None or len(image_paths) == 0:
        print("No image paths: None or empty")
        return

    images = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        image = cv2.resize(image, resize)
        images.append(image)
    return np.array(images)


def __np_images_to_tensor_images(np_images):
    tensor_images = torch.from_numpy(np_images)
    # Chuyển đổi thứ tự các trục từ (H, W, C) về (C, H, W)
    tensor_images = tensor_images.permute(0, 3, 1, 2)
    tensor_images = tensor_images.float()
    return tensor_images


def __tensor_images_to_np_images(tensor_images):
    tensor_images = tensor_images.permute(1, 2, 0)
    np_images = tensor_images.cpu().detach().numpy()
    return np_images


def __identify_bounding_boxes(model, image_path):
    """
    Identify bounding boxes of cell clusters in an image.

    Args:
        model (YOLO): Loaded YOLO model.
        image_path (str): Path to the input image.

    Returns:
        torch.Tensor: Tensor of bounding boxes in [x1, y1, x2, y2] format.
    """
    result = model(image_path)
    bounding_boxes = result[0].boxes.xyxy
    return bounding_boxes, result[0].plot(conf=False)


def __remove_noise(bounding_boxes, image_np):
    """
    Remove noise from the image by keeping only the areas within bounding boxes.

    Args:
        bounding_boxes (torch.Tensor): Tensor of bounding boxes.
        image_np (numpy.ndarray): Input image as a numpy array.

    Returns:
        tuple: (PIL.Image, dict, list) Denoised image, areas of bounding boxes, and bounding boxes as list.
    """
    areas = {}
    bounding_boxes_list = []

    for index, box in enumerate(bounding_boxes):
        x1, y1, x2, y2 = map(int, box)
        areas[index] = (x2 - x1) * (y2 - y1)
        bounding_boxes_list.append((x1, y1, x2, y2))

    return areas, bounding_boxes_list


def __get_5_largest_areas(areas):
    """
    Get the indices of the 5 largest areas.

    Args:
        areas (dict): Dictionary of area indices and their sizes.

    Returns:
        list: Indices of the 5 largest areas (or fewer if there are less than 5).
    """
    sorted_areas = sorted(areas.items(), key=lambda x: x[1], reverse=True)
    return [index for index, _ in sorted_areas[:5]]


def __augment_images(images, num_augmentations, augmentation_transforms):
    """
    Augment the given images to reach a total of 5 images using various augmentation techniques.

    Args:
        images (list): List of PIL.Image objects.
        num_augmentations (int): Number of augmentations needed.

    Returns:
        list: List of augmented PIL.Image objects.
    """
    augmented = images.copy()

    while len(augmented) < 5:
        # Randomly select an image to augment
        base_image = random.choice(images)

        # Apply the basic augmentation transforms
        augmented_image = augmentation_transforms(base_image)
        augmented_image = __tensor_images_to_np_images(augmented_image)

        augmented.append(augmented_image)

    return augmented[:5]  # Ensure we return exactly 5 images


def __crop_5_largest_areas(image, areas, bounding_boxes, augmentation_transforms):
    """
    Crop the 5 largest areas from the image or create 5 sub-images if no bounding boxes.

    Args:
        image (np.ndarray): Input image.
        areas (dict): Dictionary of area indices and their sizes.
        bounding_boxes (list): List of bounding boxes.
        path_to_save (str): Path to save the cropped images (not used in this function).

    Returns:
        list: List of 5 np.ndarray objects (cropped or sub-images).
    """
    images = []
    if len(areas) > 0:
        for index in __get_5_largest_areas(areas):
            x1, y1, x2, y2 = bounding_boxes[index]
            cropped_image = image[y1:y2, x1:x2]
            images.append(cropped_image)

        # Augment the image to have 5 areas
        if len(images) < 5:
            images = __augment_images(images, 5 - len(images), augmentation_transforms)
    else:
        image_height, image_width = image.shape[:2]
        x_mid, y_mid = image_width // 2, image_height // 2
        images = [
            image[:y_mid, :x_mid],  # Top-left
            image[:y_mid, x_mid:],  # Top-right
            image[y_mid:, :x_mid],  # Bottom-left
            image[y_mid:, x_mid:],  # Bottom-right
            image,  # Original
        ]
    return images


def get_importance_slice_images(model, image_path):
    augmentation_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Resize((256, 256)),
            # transforms.RandomCrop(224),
            transforms.RandomRotation(180),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    )
    image_np = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    bounding_boxes, yolo_detect_image = __identify_bounding_boxes(model, image_path)
    print(f"{image_path} have {len(bounding_boxes)} cell importance clusters")

    areas, bounding_boxes_list = __remove_noise(bounding_boxes, image_np)

    importance_slice_images = __crop_5_largest_areas(
        image_np, areas, bounding_boxes_list, augmentation_transforms
    )

    return importance_slice_images, bounding_boxes, yolo_detect_image
