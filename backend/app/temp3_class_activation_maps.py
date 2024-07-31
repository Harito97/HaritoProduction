from pytorch_grad_cam import GradCAM #, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import numpy as np

class ClassActivationMaps:
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.model.eval()
        self.cam = None
        self.device = next(model.parameters()).device
        self.grayscale_cams = None
        self.outputs = None

    def make_cam_images(self, np_images:np.ndarray, targets:list=None):
        """ 
        Inputs: Numpy array images: [number images, size, size, number channel], list of target classes (1D)
        Outputs: Numpy array images: [number images, size, size, number channel] - the mask of the target classes
        """
        tensor_images = self.__np_images_to_tensor_images(np_images)
        tensor_images = tensor_images.to(self.device)
        tensor_images.requires_grad = True
        self.cam = GradCAM(model=self.model, target_layers=self.target_layers)
        self.grayscale_cams = self.cam(input_tensor=tensor_images, targets=targets)
        self.outputs = self.cam.outputs
        result = []
        for image, mask in zip(np_images, self.grayscale_cams):
            output = show_cam_on_image(image, mask, use_rgb=use_rgb)
            result.append(output)
        self.cam = None
        return np.array(result)


    def cam_on_images(self, np_images:np.ndarray, use_rgb=False):
        result = []
        for image, mask in zip(np_images, self.grayscale_cams):
            output = show_cam_on_image(image, mask, use_rgb=use_rgb)
            result.append(output)
        return np.array(result)


    def __np_images_to_tensor_images(self, np_images):
        tensor_images = torch.from_numpy(np_images)
        # Chuyển đổi thứ tự các trục từ (H, W, C) về (C, H, W)
        tensor_images = tensor_images.permute(0, 3, 1, 2)
        tensor_images = tensor_images.float()
        return tensor_images

    def __tensor_images_to_np_images(self, tensor_images):
        tensor_images = tensor_images.permute(0, 2, 3, 1)
        np_images = tensor_images.cpu().detach().numpy()
        return np_images