import unittest
import os
import numpy as np
from PIL import Image
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.allocation.domain.pano_caries_detection.main import (
    create_pano_caries_detection_model,
    pano_caries_detecion
)

def override_model_transform(model):
    """覆寫模型的 transform 參數，使其符合測試圖片尺寸"""
    if hasattr(model.transform, 'min_size') and hasattr(model.transform, 'max_size'):
        model.transform.min_size = (960,) # 高度
        model.transform.max_size = 1280   # 寬度

class TestPanoCariesDetection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 模型權重檔案
        cls.model_path = './models/dentistry_pano-CariesDetection_resNetFpn_25.12.pth'
        if not os.path.exists(cls.model_path):
            raise FileNotFoundError(f"模型檔案不存在: {cls.model_path}")
        cls.model = create_pano_caries_detection_model(num_classes=1)
        override_model_transform(cls.model)
        # 正常牙片圖片
        cls.normal_image_path = './tests/files/caries-0.6741573-260-760_1_2022052768.png'
        if not os.path.exists(cls.normal_image_path):
            raise FileNotFoundError(f"正常牙片圖片不存在: {cls.normal_image_path}")
        cls.normal_img = Image.open(cls.normal_image_path).convert('RGB')
        # 全黑圖片（若不存在則自動產生）
        cls.black_image_path = './tests/files/black.png'
        if not os.path.exists(cls.black_image_path):
            black_img = Image.new('RGB', (1280, 960), (0, 0, 0))
            os.makedirs(os.path.dirname(cls.black_image_path), exist_ok=True)
            black_img.save(cls.black_image_path)
        cls.black_img = Image.open(cls.black_image_path).convert('RGB')

    def test_black_image_handling(self):
        """測試：全黑圖片應回傳無齲齒"""
        result, msg = pano_caries_detecion(
            self.model,
            self.model_path,
            self.black_img,
            return_type='image_array'
        )
        # 全黑圖片應回傳 None，訊息為 "No caries found"
        #self.assertIsNone(result, "全黑圖片應回傳 None")
        #self.assertEqual(result, np.array(self.black_img))
        
        self.assertEqual(msg, "No caries found", "錯誤訊息應為 'No caries found'")

    def test_normal_image_processing(self):
        """測試：正常牙片應有檢測結果"""
        result, msg = pano_caries_detecion(
            self.model,
            self.model_path,
            self.normal_img,
            return_type='image_array'
        )
        # 應有檢測結果且型態為 PIL Image
        self.assertIsNotNone(result, "正常牙片應回傳檢測結果")
        self.assertIsInstance(result, np.ndarray, "回傳應為 numpy array")

if __name__ == '__main__':
    unittest.main()
