import unittest
import torch
import numpy as np
import os
import sys
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from allocation.domain.leyan_clinic_scenario_classfication.main import (
    create_Leyan_clinic_scenario_classfication,
    read_pil_image,
    predict_image_Leyan_clinic_scenario_classfication
)

class TestLeyan_clinic_scenario_classfication(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 模型權重檔案
        cls.model_path = './models/dentistry_leyan_clinic-classification_cnn_25.28.pth'
        if not os.path.exists(cls.model_path):
            raise FileNotFoundError(f"模型檔案不存在: {cls.model_path}")
        cls.model = create_Leyan_clinic_scenario_classfication(cls.model_path)

        # 正面閉口圖片
        cls.close_image_path = './tests/files/close.jpg'
        if not os.path.exists(cls.close_image_path):
            raise FileNotFoundError(f"正面閉口圖片不存在: {cls.close_image_path}")
        cls.close_image = read_pil_image(cls.close_image_path)
        # 微笑圖片
        cls.smile_image_path = './tests/files/smile.jpg'
        if not os.path.exists(cls.smile_image_path):
            raise FileNotFoundError(f"微笑圖片不存在: {cls.smile_image_path}")
        cls.smile_image = read_pil_image(cls.smile_image_path)
        # xray圖片
        cls.xray_image_path = './tests/files/xray.jpg'
        if not os.path.exists(cls.xray_image_path):
            raise FileNotFoundError(f"xray圖片不存在: {cls.xray_image_path}")
        cls.xray_image_path = read_pil_image(cls.xray_image_path)
        # 非xray圖片
        cls.non_xray_image_path = './tests/files/non_xray.jpg'
        if not os.path.exists(cls.non_xray_image_path):
            raise FileNotFoundError(f"非xray圖片不存在: {cls.non_xray_image_path}")
        cls.non_xray_image_path = read_pil_image(cls.non_xray_image_path)

        # 全黑圖片（若不存在則自動產生）
        cls.black_image_path = './tests/files/black.png'
        if not os.path.exists(cls.black_image_path):
            black_img = Image.new('RGB', (128, 128), (0, 0, 0))
            os.makedirs(os.path.dirname(cls.black_image_path), exist_ok=True)
            black_img.save(cls.black_image_path)
        cls.black_image = read_pil_image(cls.black_image_path)

    def test_predict_close_image(self): # 函式名稱可以更精確
        """測試：正面閉口圖片應被正確分類"""
        pred_class, score = predict_image_Leyan_clinic_scenario_classfication(
            self.model,
            self.close_image
        )
        # 斷言預測結果必須是 'close'
        self.assertEqual(pred_class, 'close')
        self.assertIsInstance(score[pred_class], (float, np.floating))

    def test_predict_smile_image(self):
        """測試：微笑圖片應被正確分類"""
        pred_class, score = predict_image_Leyan_clinic_scenario_classfication(
            self.model,
            self.smile_image
        )
        self.assertEqual(pred_class, 'smile')
        self.assertIsInstance(score[pred_class], (float, np.floating))

    def test_predict_xray_image(self):
        """測試：xray圖片應被正確分類"""
        pred_class, score = predict_image_Leyan_clinic_scenario_classfication(
            self.model,
            self.xray_image_path # 假設已在 setUpClass 中載入
        )
        self.assertEqual(pred_class, 'other')
        self.assertIsInstance(score[pred_class], (float, np.floating))

    def test_predict_black_image(self):
        """測試：全黑圖片應被分類為 'other'"""
        pred_class, score = predict_image_Leyan_clinic_scenario_classfication(
            self.model,
            self.black_image
        )
        # 全黑圖片預期應該是 'other'
        self.assertEqual(pred_class, 'other')
        self.assertIsInstance(score[pred_class], (float, np.floating))

if __name__ == '__main__':
    unittest.main()
