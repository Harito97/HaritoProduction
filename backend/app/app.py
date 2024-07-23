from backend.app.load_model import load_model1, load_model2
from backend.app.class_activation_maps import ClassActivationMaps
from backend.utils.image import read_images, plot_heatmap

import torch

# model1 = load_model1()
model2 = load_model2()
model2.to(device)

class App:
    def __init__(self):
        self.model1 = load_model1()
        self.model2 = load_model2()
        target_layers = model2.feature_extractor[-2]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cam = ClassActivationMaps(self.model2, target_layers)

    def run(self, image_paths):
        images = read_images(image_paths)
        self.cam.make_cam_images(images)
        cam_images = self.cam.cam_on_images(images, use_rgb=True)
        for cam_image in cam_images:
            plot_heatmap(cam_image)