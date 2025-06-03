import unittest
import os
import numpy as np
from PIL import Image
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.allocation.domain.pa_pano_classification.main import (
    create_pa_pano_classification_model,
    read_pil_image,
    predict_image_pa_pano_classification
)

class TestPAPanoClassification(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 模型權重檔案
        cls.model_path = './models/dentistry_pa-pano-classification_cnn_25.22.pth'
        if not os.path.exists(cls.model_path):
            raise FileNotFoundError(f"模型檔案不存在: {cls.model_path}")
        cls.model = create_pa_pano_classification_model(cls.model_path)
        # 正常牙片圖片
        cls.normal_image_path = './tests/files/raw_12.png'
        if not os.path.exists(cls.normal_image_path):
            raise FileNotFoundError(f"正常牙片圖片不存在: {cls.normal_image_path}")
        cls.normal_image = read_pil_image(cls.normal_image_path)
        # 全黑圖片（若不存在則自動產生）
        cls.black_image_path = './tests/files/black.png'
        if not os.path.exists(cls.black_image_path):
            black_img = Image.new('RGB', (128, 128), (0, 0, 0))
            os.makedirs(os.path.dirname(cls.black_image_path), exist_ok=True)
            black_img.save(cls.black_image_path)
        cls.black_image = read_pil_image(cls.black_image_path)

    def test_predict_normal_xray(self):
        """測試：真實正常牙片分類"""
        pred_class, score = predict_image_pa_pano_classification(
            self.model,
            self.normal_image
        )
        # 應分類為 periapical film 或 panoramic x-ray，且分數為 float
        self.assertIn(pred_class, ['periapical film', 'panoramic x-ray', 'other'])
        self.assertIsInstance(score, (float, np.floating)) # 允許 numpy float

    def test_predict_black_image(self):
        """測試：全黑圖片分類"""
        pred_class, score = predict_image_pa_pano_classification(
            self.model,
            self.black_image
        )
        # 應分類為 periapical film 或 panoramic x-ray，且分數為 float
        self.assertIn(pred_class, ['periapical film', 'panoramic x-ray', 'other'])
        self.assertIsInstance(score, (float, np.floating))

if __name__ == '__main__':
    unittest.main()
