import torch
import torch.nn.functional as F
from ultralytics import YOLO
from backend.app.models.H97 import H97_EfficientNet, H97_ANN
from backend.utils.visualize_metrics import Tool
from backend.utils.image import get_importance_slice_images
import cv2
import numpy as np
import wandb
import os
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize


class ThyroidCancerClassificationModel:
    def __init__(self, model1_path=None, model2_path=None, model3_path=None):
        if model1_path is None:
            model1_path = "/Data/Projects/HaritoProduction/backend/app/models/cell_cluster_detect_300_epoches_best.onnx"
        if model2_path is None:
            model2_path = "/Data/Projects/HaritoProduction/backend/app/models/best_h97_retrainEfficientNet_B2_B5_B6_dataver3_model.pt"
        if model3_path is None:
            model3_path = (
                "/Data/Projects/HaritoProduction/backend/app/models/model_best.pt"
            )

        self.image_origin_dir = None
        self.origin_image = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model1 = YOLO(model1_path)
        self.model2 = H97_EfficientNet()
        self.target_layer = self.model2.feature_extractor[
            -2
        ]  # draw the CAM from the last layer of the feature extractor
        self.model2.load_state_dict(torch.load(model2_path, map_location=self.device))
        self.model2.eval()
        self.model3 = H97_ANN()
        self.model3.load_state_dict(torch.load(model3_path, map_location=self.device))
        self.model3.eval()

    def get_images(self, image_origin_dir=None):
        # Đọc ảnh gốc
        self.image_origin_dir = image_origin_dir
        self.origin_image = cv2.imread(image_origin_dir)
        if self.image_origin_dir is None:
            print(f"Error reading image {image_origin_dir}")
            return
        self.origin_image = cv2.cvtColor(self.origin_image, cv2.COLOR_BGR2RGB)
        self.origin_image = cv2.resize(self.origin_image, (1024, 768))
        self.origin_image = self.origin_image / 255.0

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

        part0 = cv2.resize(part0, (224, 224))
        part1 = cv2.resize(part1, (224, 224))
        part2 = cv2.resize(part2, (224, 224))
        part3 = cv2.resize(part3, (224, 224))
        part4 = cv2.resize(part4, (224, 224))
        part5 = cv2.resize(part5, (224, 224))
        part6 = cv2.resize(part6, (224, 224))
        part7 = cv2.resize(part7, (224, 224))
        part8 = cv2.resize(part8, (224, 224))
        part9 = cv2.resize(part9, (224, 224))
        part10 = cv2.resize(part10, (224, 224))
        part11 = cv2.resize(part11, (224, 224))

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

        importance_slice_images, __bounding_boxes, self.yolo_detect_image = (
            get_importance_slice_images(self.model1, image_origin_dir)
        )
        for image in importance_slice_images:
            image = (
                cv2.resize(image, (224, 224)) / 255.0
            )  # 255 o day thi chac chan can chia 255 vi qua YOLO no da khoi phuc ve khoang 0-255 roi
            images.append(image)

        images.append(
            cv2.resize(self.origin_image, (224, 224)) # / 255.0
        )  # sua lai co le khong nen chia 255 o day - nhung trc khi sua thi nen train lai voi H97_ANN voi input la anh goc chua chia 255
        return np.array(images)

    def get_output_model1(self, image_origin_dir):
        output_model1 = self.get_images(image_origin_dir)
        print(type(output_model1), output_model1.shape, output_model1.dtype)
        return output_model1


    def get_output_model2(self, images):
        images_tensor = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
        images_tensor.requires_grad_()

        cam_images = []
        features = []

        def hook_fn(module, input, output):
            features.append(output)
            # Ensure that the gradients are calculated for this layer
            output.register_hook(lambda grad: features.append(grad))

        hook = self.target_layer.register_forward_hook(hook_fn)

        output_model2 = self.model2(images_tensor)

        hook.remove()

        # Calculate gradients w.r.t the output
        self.model2.zero_grad()
        output_model2.backward(gradient=torch.ones_like(output_model2))

        # Get gradients and feature maps
        feature_maps = features[0].detach()
        gradients = features[1].detach()
        print(type(gradients), gradients.shape, gradients.dtype)
        print(feature_maps.shape, feature_maps.dtype)

        # Ensure that gradients and feature_maps are available
        if gradients is None or feature_maps is None:
            raise ValueError("No gradients or feature maps were computed.")

        # Calculate weights
        num_channels = feature_maps.size(1)
        weights = torch.mean(gradients, dim=[0, 2, 3])  # Mean across spatial dimensions
        weights = weights.view(-1)  # Flatten to a 1D tensor
        print(weights.shape, weights.dtype)

         # Calculate CAM
        H, W = feature_maps.size(2), feature_maps.size(3)
        feature_maps_reshaped = feature_maps.reshape(18, num_channels, H * W)  # (num_channels, H*W)
        cam = torch.matmul(weights, feature_maps_reshaped)  # (H*W)
        cam = cam.view(feature_maps.size(2), feature_maps.size(3))  # (H, W)

        # Apply ReLU
        cam = F.relu(cam)

        # Normalize CAM
        cam -= torch.min(cam)
        cam /= torch.max(cam)

        # Resize CAM to match input image size
        for i in range(cam.size(0)):  # Assuming batch dimension
            cam_img = cam.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            cam_img = F.interpolate(cam_img, size=images_tensor.shape[2:], mode='bilinear', align_corners=False)
            cam_images.append(cam_img.squeeze().cpu().numpy())

        # Print output information
        print(type(output_model2), output_model2.shape, output_model2.dtype)

        return output_model2, cam_images


    def get_output_model3(self, output_model2):
        input_model3 = output_model2.view(
            -1
        )  # flatten the tensor from torch.Size([18, 3]) to torch.Size([54])
        input_model3 = input_model3.unsqueeze(
            0
        )  # add batch dimension (mean from torch.Size([54]) to torch.Size([1, 54])

        # input of model 3 is tensor of shape (batch_size, 54)
        output_model3 = self.model3(input_model3)
        # _, predicted = torch.max(output_model3, 1)
        # return predicted
        print(type(output_model3), output_model3.shape, output_model3.dtype)
        return output_model3

    def forward(self, image_origin_dir):
        output_model1 = self.get_output_model1(image_origin_dir)
        output_model2, cam_images = self.get_output_model2(
            output_model1
        )
        output_model3 = self.get_output_model3(output_model2)
        _, predicted = torch.max(output_model3, 1)
        # cam_images = self.get_CAM(output_model2, images_tensor_requires_grad, predicted)
        return predicted, (output_model3, output_model2, output_model1), cam_images

    def run_app(self, image_origin_dir):
        predicted, outputs, cam_images = self.forward(image_origin_dir)
        # Ghép các phần lại với nhau
        top_row = np.hstack(
            (
                cam_images[0],
                cam_images[1],
                cam_images[2],
                cam_images[3],
            )
        )
        middle_row = np.hstack(
            (
                cam_images[4],
                cam_images[5],
                cam_images[6],
                cam_images[7],
            )
        )
        bottom_row = np.hstack(
            (
                cam_images[8],
                cam_images[9],
                cam_images[10],
                cam_images[11],
            )
        )
        # Ghép các hàng lại với nhau để tạo thành ảnh tổng hợp
        patch_level_heatmap = np.vstack((top_row, middle_row, bottom_row))
        # Image level heatmap
        image_level_heatmap = cam_images[17]
        # YOLO detect image
        yolo_detect_image = self.yolo_detect_image

        return (
            predicted,
            outputs,
            patch_level_heatmap,
            image_level_heatmap,
            yolo_detect_image,
        )

    def test_with_each_part(
        self, data_dir, name_dataset, print_image_dir_processing=False
    ):
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        wandb.init(
            project="ThyroidCancer",
            entity="harito",
            name=f"final test in {name_dataset} set",
        )
        preds = []
        probs = []
        true_labels = []
        for label_index, label in enumerate([".B2", "B5", "B6"]):
            for image in os.listdir(os.path.join(data_dir, label)):
                image_dir = os.path.join(data_dir, label, image)
                if print_image_dir_processing:
                    print("Processing:", image_dir)
                pred, prob = self.forward(image_dir)
                pred = pred.tolist()
                prob = prob[0].squeeze().tolist()
                preds += pred
                probs.append(prob)
                true_labels.append(label_index)

        preds = np.array(preds)
        probs = np.array(probs)
        true_labels = np.array(true_labels)

        print(preds.shape, probs.shape, true_labels.shape)
        print(preds, probs, true_labels)

        test_acc = np.mean(preds == true_labels)
        test_f1 = f1_score(true_labels, preds, average="weighted")

        # Save confusion matrix
        cm = Tool.save_confusion_matrix(
            true_labels,
            preds,
            ["B2", "B5", "B6"],
            "confusion_matrix.png",
        )
        # Save classification report
        cr = Tool.save_classification_report(
            true_labels,
            preds,
            "classification_report.png",
        )
        # Save ROC AUC plot
        Tool.save_roc_auc_plot(true_labels, probs, 3, "roc_auc.png")

        wandb.log(
            {
                "test_acc": test_acc,
                "test_f1": test_f1,
                "confusion_matrix": (
                    wandb.Image("confusion_matrix.png")
                    if os.path.exists("confusion_matrix.png")
                    else None
                ),
                "classification_report": (
                    wandb.Image("classification_report.png")
                    if os.path.exists("classification_report.png")
                    else None
                ),
                "roc_auc_plot": (
                    wandb.Image("roc_auc.png")
                    if os.path.exists("roc_auc.png")
                    else None
                ),
            }
        )
        wandb.finish()

        cm_path = "confusion_matrix.png"
        cr_path = "classification_report.png"
        roc_auc_path = "roc_auc.png"

        # Hiển thị các ảnh
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(mpimg.imread(cm_path))
        axs[0].set_title("Confusion Matrix")
        axs[0].axis("off")

        axs[1].imshow(mpimg.imread(cr_path))
        axs[1].set_title("Classification Report")
        axs[1].axis("off")

        axs[2].imshow(mpimg.imread(roc_auc_path))
        axs[2].set_title("ROC AUC Plot")
        axs[2].axis("off")

        plt.show()

        print(f"Test accuracy: {test_acc}")
        print(f"Test F1 score: {test_f1}")
        print(f"Classification Report: {cr}")
        print(f"Confusion Matrix: {cm}")

        return test_acc, test_f1, cm, cr, cm_path, cr_path, roc_auc_path
