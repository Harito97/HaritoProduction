# import sys
# import os

# # Đảm bảo rằng bạn đang ở thư mục gốc của dự án hoặc xác định đúng đường dẫn tuyệt đối
# # Lấy đường dẫn thư mục hiện tại (thư mục chứa script)
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # Lấy đường dẫn thư mục gốc của dự án (giả sử models nằm trong app)
# project_root = os.path.abspath(os.path.join(current_dir, '..'))

# # Thêm thư mục gốc của dự án vào sys.path
# if project_root not in sys.path:
#     sys.path.append(project_root)

import torch
from ultralytics import YOLO
from backend.app.models.H97 import H97_EfficientNet


def load_model1():
    model = YOLO(
        "/Data/Projects/HaritoProduction/backend/app/models/cell_cluster_detect_300_epoches_best.onnx",
        # "/Data/Projects/HaritoProduction/backend/app/models/cell_cluster_detect_300_epoches_best.pt",
    )
    # model.eval() # Hien tai dong nay dang lam viec load vao bi loi va minh khong hieu tai sao
    return model


def load_model2():
    model = H97_EfficientNet()
    model.load_state_dict(
        torch.load(
            "/Data/Projects/HaritoProduction/backend/app/models/best_h97_retrainEfficientNet_B2_B5_B6_dataver3_model.pt",
            map_location=torch.device("cpu"),
        ),
    )
    model.eval()
    return model
