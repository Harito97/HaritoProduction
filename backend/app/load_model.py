import torch
from ultralytics import YOLO
from backend.app.models.H97 import H97_EfficientNet


def load_model1(model1_path):
    if model1_path is None:
        model1_path = "/Data/Projects/HaritoProduction/backend/app/models/cell_cluster_detect_300_epoches_best.onnx"
    model = YOLO(
        model1_path,
        # "/Data/Projects/HaritoProduction/backend/app/models/cell_cluster_detect_300_epoches_best.pt",
    )
    # model.eval() # Hien tai dong nay dang lam viec load vao bi loi va minh khong hieu tai sao
    return model


def load_model2(model2_path):
    if model2_path is None:
        model2_path = "/Data/Projects/HaritoProduction/backend/app/models/best_h97_retrainEfficientNet_B2_B5_B6_dataver3_model.pt"
    model = H97_EfficientNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    model.load_state_dict(
        torch.load(
            model2_path,
            # "/Data/Projects/HaritoProduction/backend/app/models/best_h97_retrainEfficientNet_B2_B5_B6_dataver3_model.pt",
            map_location=device,
        ),
    )
    model.eval()
    return model
