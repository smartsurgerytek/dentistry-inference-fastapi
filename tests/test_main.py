import numpy as np
import cv2
import sys
import os
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.dental_measure.main import *

@pytest.fixture
def setup_test_environment():
    """設置測試用的假圖像和遮罩"""
    test_image = np.zeros((500, 500, 3), dtype=np.uint8)  # 創建一個黑色圖像
    masks_dict = {
        'dental_crown': np.ones((500, 500), dtype=np.uint8) * 255,  # 模擬牙冠的遮罩
        'dentin': np.ones((500, 500), dtype=np.uint8) * 255,        # 模擬牙本質的遮罩
        'gum': np.ones((500, 500), dtype=np.uint8) * 255            # 模擬牙齦的遮罩
    }
    return test_image, masks_dict

def test_extract_features(setup_test_environment):
    """測試 extract_features 函數"""
    test_image, masks_dict = setup_test_environment
    print("正在測試 extract_features 函數...")
    overlay, line_image, non_masked_area = extract_features(masks_dict, test_image)

    # 確保返回的圖像是三通道
    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR) if len(overlay.shape) == 2 else overlay
    line_image = cv2.cvtColor(line_image, cv2.COLOR_GRAY2BGR) if len(line_image.shape) == 2 else line_image
    non_masked_area = cv2.cvtColor(non_masked_area, cv2.COLOR_GRAY2BGR) if len(non_masked_area.shape) == 2 else non_masked_area

    # 驗證返回的 overlay 是否為正確形狀
    assert overlay.shape == (500, 500, 3)
    assert line_image.shape == (500, 500, 3)
    assert non_masked_area.shape == (500, 500, 3)
    print("extract_features 測試通過！")

def test_locate_points(setup_test_environment):
    """測試 locate_points 函數"""
    test_image, masks_dict = setup_test_environment
    print("正在測試 locate_points 函數...")
    component_mask = np.ones((500, 500), dtype=np.uint8) * 255  # 模擬完整的遮罩
    binary_images = {
        'dental_crown': masks_dict['dental_crown'],
        'gum': masks_dict['gum'],
        'dentin': masks_dict['dentin']
    }
    overlay = test_image.copy()
    prediction = locate_points(test_image, component_mask, binary_images, idx=0, overlay=overlay)

    # 驗證返回的預測是否包含預期的鍵
    assert "teeth_center" in prediction
    print("locate_points 測試通過！")

def test_dental_estimation(setup_test_environment):
    """測試 dental_estimation 函數"""
    test_image, _ = setup_test_environment
    print("正在測試 dental_estimation 函數...")
    result_image = dental_estimation(test_image, return_type='image')

    # 驗證返回的圖像是否為正確形狀
    assert result_image.shape == test_image.shape
    print("dental_estimation 測試通過！")

# 執行所有測試
if __name__ == "__main__":
    print("開始執行所有測試...")
    test_extract_features(setup_test_environment())
    test_locate_points(setup_test_environment())
    test_dental_estimation(setup_test_environment())
    print("所有測試執行完畢！")