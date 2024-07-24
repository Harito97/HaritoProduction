from backend.app.load_model import load_model1, load_model2
from backend.app.class_activation_maps import ClassActivationMaps

from backend.utils.image import get_importance_slice_images  # read_images, plot_heatmap

import torch
import cv2
import numpy as np


class App:
    def __init__(self, model1_path=None, model2_path=None):
        self.model1 = load_model1(model1_path)
        self.model2 = load_model2(model2_path)
        target_layers = self.model2.feature_extractor[-2]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cam = None, # ClassActivationMaps(self.model2, target_layers)
        self.image_origin_dir = None
        self.origin_image = None

        self.cam_origin_image = None
        self.result_cam_origin_image = None

        self.images = None
        self.yolo_detect_image = None
        self.bounding_boxes = None
        self.cam_images = None

    def get_prediction(self):
        tensor1 = self.cam.cam.outputs
        tensor2 = self.result_cam_origin_image
        tensor_combined = torch.cat((tensor1, tensor2), dim=0)
        return tensor_combined.flatten()

    def run(self, image_origin_dir):
        self.image_origin_dir = image_origin_dir
        self.images = self.get_images()

        self.cam.make_cam_images(np.array([self.origin_image.copy()]))
        self.cam_origin_image = self.cam.cam_on_images(
            np.array([self.origin_image.copy()]), use_rgb=True
        )
        self.result_cam_origin_image = self.cam.cam.outputs

        self.cam.make_cam_images(self.images)
        self.cam_images = self.cam.cam_on_images(self.images, use_rgb=True)

    def get_images(self, image_origin_dir=None):
        # Đọc ảnh gốc
        if image_origin_dir is not None:
            self.image_origin_dir = image_origin_dir
        self.origin_image = cv2.imread(self.image_origin_dir)
        if self.origin_image is None:
            print(f"Error reading image {self.image_origin_dir}")
            return
        self.origin_image = cv2.cvtColor(self.origin_image, cv2.COLOR_BGR2RGB)
        self.origin_image = self.origin_image / 255.0
        self.origin_image = cv2.resize(self.origin_image, (1024, 768))

        images = []

        # Chia ảnh thành các phần
        part0 = self.origin_image[0:256, :256]
        part1 = self.origin_image[0:256, 256:512]
        part2 = self.origin_image[0:256, 512:768]
        part3 = self.origin_image[0:256, 768:1024]

        part4 = self.origin_image[256:512, :256]
        part5 = self.origin_image[256:512, 256:512]
        part6 = self.origin_image[256:512, 512:768]
        part7 = self.origin_image[256:512, 768:1024]

        part8 = self.origin_image[512:768, :256]
        part9 = self.origin_image[512:768, 256:512]
        part10 = self.origin_image[512:768, 512:768]
        part11 = self.origin_image[512:768, 768:1024]

        # Thêm các phần ảnh vào danh sách
        images.append(part0)
        images.append(part1)
        images.append(part2)
        images.append(part3)
        images.append(part4)
        images.append(part5)
        images.append(part6)
        images.append(part7)
        images.append(part8)
        images.append(part9)
        images.append(part10)
        images.append(part11)

        # print(part0.shape, part1.shape, part2.shape, part3.shape)
        # print(part4.shape, part5.shape, part6.shape, part7.shape)
        # print(part8.shape, part9.shape, part10.shape, part11.shape)

        importance_slice_images, self.bounding_boxes, self.yolo_detect_image = (
            get_importance_slice_images(self.model1, self.image_origin_dir)
        )
        for image in importance_slice_images:
            image = cv2.resize(image, (256, 256)) / 255.0
            images.append(image)

        images.append(cv2.resize(self.origin_image, (256, 256)) / 255.0)
        return np.array(images)

    def get_analyst_image1(self):
        # Ghép các phần lại với nhau
        top_row = np.hstack(
            (
                self.cam_images[0],
                self.cam_images[1],
                self.cam_images[2],
                self.cam_images[3],
            )
        )
        middle_row = np.hstack(
            (
                self.cam_images[4],
                self.cam_images[5],
                self.cam_images[6],
                self.cam_images[7],
            )
        )
        bottom_row = np.hstack(
            (
                self.cam_images[8],
                self.cam_images[9],
                self.cam_images[10],
                self.cam_images[11],
            )
        )

        # Ghép các hàng lại với nhau để tạo thành ảnh tổng hợp
        analyse_image1 = np.vstack((top_row, middle_row, bottom_row))
        return analyse_image1

    def get_analyst_image2(self):
        return self.cam_origin_image

    def get_analyst_image3(self):
        return self.yolo_detect_image


if __name__ == "__main__":
    image_path = "/Data/Projects/ThyroidCancer/Phase1/Data/origin_data/B256/B2/z5010203657572_d225c21872a6e6eb8342a0fe4a0beced.jpg"
    app = App(image_path)
    # then can change with other data without re-create new instance by using: app.rerun(image_path)
    # Eg to use the get_analyst_image1
    analyst_image1, _ = app.get_analyst_image1()
    plot_heatmap(analyst_image1)
    import torch

    model_outputs = app.cam.cam.outputs
    max_indices = torch.argmax(model_outputs, dim=1)
    print(max_indices)
