from backend.app.load_model import load_model1, load_model2
from backend.utils.image import get_importance_slice_images
import torch
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image
from pytorch_grad_cam.utils.image import show_cam_on_image


class H97ThyroidCancer:
    def __init__(self, model1_path=None, model2_path=None):
        self.model1 = load_model1(model1_path)
        self.model2 = load_model2(model2_path)
        # self.target_layers = self.model2.feature_extractor[-2]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self, image_origin_dir, use_rgb=True):
        images, yolo_detect_image = self.get_images(image_origin_dir=image_origin_dir)
        grayscale_cams = []
        output_model2 = []
        cam_on_images = []

        with GradCAM(
            model=self.model2,
            target_layers=self.model2.feature_extractor[-2],
        ) as cam:
            for i in range(len(images)):
                print(f'Image i th: {i}')
                rgb_img = np.float32(images[i])  # / 255
                input_tensor = preprocess_image(
                    rgb_img,
                ).requires_grad_(True).to(self.device)

                grayscale_cam = cam(
                    input_tensor=input_tensor,
                    targets=None,
                    # aug_smooth=False,
                    # eigen_smooth=False,
                )

                grayscale_cams.append(grayscale_cam)
                output_model2.append(cam.outputs)
                # if i == 17:
                #     cam_on_image = show_cam_on_image(rgb_img * 255.0, grayscale_cam[0], use_rgb=True)
                # else:
                cam_on_image = show_cam_on_image(rgb_img, grayscale_cam[0], use_rgb=use_rgb)
                cam_on_images.append(cam_on_image)
        self.cam_on_images = cam_on_images
        self.output_model2 = output_model2
        self.grayscale_cams = grayscale_cams
        self.yolo_detect_image = yolo_detect_image
        self.images = images

    def plot_results(self):
        import matplotlib.pyplot as plt
        num_images = len(self.images)

        fig, axs = plt.subplots(num_images, 4, figsize=(20, 5 * num_images))

        for i in range(num_images):
            # Hiển thị ảnh gốc
            axs[i, 0].imshow(self.images[i])
            axs[i, 0].set_title('Original Image')
            axs[i, 0].axis('off')

            # Hiển thị ảnh YOLO detect
            axs[i, 1].imshow(self.yolo_detect_image[i])
            axs[i, 1].set_title('YOLO Detected Image')
            axs[i, 1].axis('off')

            # Hiển thị ảnh CAM chồng lên ảnh gốc
            if self.cam_on_images:
                axs[i, 2].imshow(self.cam_on_images[i])
                axs[i, 2].set_title('CAM on Image')
                axs[i, 2].axis('off')

            # Hiển thị ảnh CAM (grayscale)
            grayscale_cam = self.grayscale_cams[i][0]  # Lấy ảnh đầu tiên nếu là batch
            axs[i, 3].imshow(grayscale_cam, cmap='jet')
            axs[i, 3].set_title('Grayscale CAM')
            axs[i, 3].axis('off')

        plt.tight_layout()
        plt.show()


    def get_images(self, image_origin_dir=None):
        # Đọc ảnh gốc
        self.origin_image = cv2.imread(image_origin_dir)
        if self.origin_image is None:
            print(f"Error reading image {image_origin_dir}")
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
        # images now contains 12 parts of the image

        importance_slice_images, __bounding_boxes, yolo_detect_image = (
            get_importance_slice_images(self.model1, image_origin_dir)
        )
        for image in importance_slice_images:
            image = cv2.resize(image, (256, 256)) / 255.0
            images.append(image)
        # images has 17 images, 12 parts of the image and 5 importance slice images

        images.append(
            cv2.resize(self.origin_image, (256, 256)) # / 255.0
        )  # tai trong model ANN lo lam cai anh nay bi chia 255
        # images has 18 images, 12 parts of the image, 5 importance slice images and 1 resized image

        return np.array(images), yolo_detect_image

    def get_analyst_image1(self):
        # Ghép các phần lại với nhau
        top_row = np.hstack(
            (
                self.cam_on_images[0],
                self.cam_on_images[1],
                self.cam_on_images[2],
                self.cam_on_images[3],
            )
        )
        middle_row = np.hstack(
            (
                self.cam_on_images[4],
                self.cam_on_images[5],
                self.cam_on_images[6],
                self.cam_on_images[7],
            )
        )
        bottom_row = np.hstack(
            (
                self.cam_on_images[8],
                self.cam_on_images[9],
                self.cam_on_images[10],
                self.cam_on_images[11],
            )
        )

        # Ghép các hàng lại với nhau để tạo thành ảnh tổng hợp
        analyse_image1 = np.vstack((top_row, middle_row, bottom_row))
        return analyse_image1

    def get_analyst_image2(self):
        return cv2.resize(self.cam_on_images[17], (1024, 768))
        

    def get_analyst_image3(self):
        return self.yolo_detect_image
        
