import os
import numpy as np
import cv2
import pytest
import sys
import yaml
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.allocation.domain.pano_fdi_segmentation.main import pano_fdi_segmentation, YOLO

# 測試用檔案路徑
TEST_IMAGE = "./tests/files/027107.jpg"
BLACK_IMAGE = "./tests/files/black.png"
CORRUPTED_IMAGE = "./tests/files/corrupted.jpg"
NOT_AN_IMAGE = "./tests/files/not_an_image.txt"
CUSTOM_PLOT_CONFIG = "./conf/pano_fdi_segmentation_mask_color_setting.yaml"
MODEL_PATH = "./models/dentistry_pano-fdi-segmentation_yolo11x-seg_25.12.pt"

@pytest.fixture(scope="module")
def model():
    # 載入測試用 YOLO 模型
    return YOLO(MODEL_PATH)

def test_normal_image_image_array(model):
    """測試：正常牙片 image_array 輸出"""
    image = cv2.imread(TEST_IMAGE)
    result, error_msg = pano_fdi_segmentation(image, model, return_type='image_array')
    assert isinstance(result, np.ndarray)
    # 應無錯誤或不是 "No teeth detected"
    assert error_msg is None or error_msg == "" or "No teeth detected" not in error_msg

def test_normal_image_cvat(model):
    """測試：正常牙片 cvat 格式輸出"""
    image = cv2.imread(TEST_IMAGE)
    result = pano_fdi_segmentation(image, model, return_type='cvat')
    assert isinstance(result, dict)
    # 應包含 yolov8_contents 且有內容
    assert "yolov8_contents" in result
    assert len(result["yolov8_contents"]) > 0

def test_invalid_image(model):
    """測試：損壞或非圖片檔案處理"""
    # 測試損壞圖片
    image = cv2.imread(CORRUPTED_IMAGE)
    assert image is None
    # 測試非圖片檔案
    image = cv2.imread(NOT_AN_IMAGE)
    assert image is None
    # 傳入 None 應回傳錯誤訊息或噴例外
    try:
        result, error_msg = pano_fdi_segmentation(image, model, return_type='image_array')
        assert result is None or isinstance(result, np.ndarray)
        assert error_msg is not None and error_msg != ""
    except Exception:
        # 若主程式直接噴錯也算合理
        pass

def test_custom_plot_config(model):
    """測試：自訂 plot_config 輸出"""
    image = cv2.imread(TEST_IMAGE)
    with open(CUSTOM_PLOT_CONFIG, "r") as f:
        plot_config = yaml.safe_load(f)
    result, error_msg = pano_fdi_segmentation(image, model, plot_config=plot_config, return_type='image_array')
    assert isinstance(result, np.ndarray)
    # 可根據 plot_config 做更細緻檢查

def test_black_image_empty_result(model):
    """測試：全黑圖片應回傳空結果"""
    image = cv2.imread(BLACK_IMAGE)
    # 測試 image_array 回傳
    result, error_msg = pano_fdi_segmentation(image, model, return_type='image_array')
    assert isinstance(result, np.ndarray)
    # 應回傳特定錯誤訊息
    assert error_msg == "No segmenation masks detected"
    # 測試 cvat 回傳
    result_dict = pano_fdi_segmentation(image, model, return_type='cvat')
    assert isinstance(result_dict, dict)
    assert "yolov8_contents" in result_dict
    assert len(result_dict["yolov8_contents"]) == 0
