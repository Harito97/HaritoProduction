import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import numpy as np

def plot_heatmap(heatmap_np):
    # heatmap_np = 255 * heatmap_np
    # plt.imshow(heatmap_np, cmap="jet", alpha=0.5)
    plt.imshow(heatmap_np) #, cmap="jet")
    plt.axis("off")  # Tắt các trục để hiển thị rõ hơn
    plt.show()

def read_images(image_paths, resize=(224, 224)):
    images = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        image = cv2.resize(image, resize)
        images.append(image)
    return np.array(images)