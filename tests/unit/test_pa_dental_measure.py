import unittest
import numpy as np
import cv2
import os
from ultralytics import YOLO
from unittest import mock
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.allocation.domain.pa_dental_measure.main import dental_estimation, get_mask_dict_from_model

class TestDentalEstimation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 設定模型路徑
        cls.component_model_path = './models/dentistry_pa-segmentation_yolov11x-seg-all_24.42.pt'
        cls.contour_model_path = './models/dentistry_pa-contour_yolov11n-seg_24.46.pt'
        # 測試圖片路徑
        cls.normal_image_path = './tests/files/nomal-x-ray-0.8510638-270-740_0_2022011008.png'
        cls.black_image_path = './tests/files/black.png'
        # 檢查檔案存在
        missing = [p for p in [cls.component_model_path, cls.contour_model_path,
                               cls.normal_image_path, cls.black_image_path]
                   if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"缺失檔案: {missing}")
        # 載入模型
        cls.component_model = YOLO(cls.component_model_path)
        cls.contour_model = YOLO(cls.contour_model_path)
        # 載入測試圖片
        cls.normal_image = cv2.imread(cls.normal_image_path)
        cls.black_image = np.zeros_like(cls.normal_image)
        cls.white_image = np.ones_like(cls.normal_image)*255

    def test_normal_image_processing(self):
        """測試：正常牙片處理流程"""
        result = dental_estimation(
            self.normal_image,
            self.component_model,
            self.contour_model,
            return_type='dict'
        )
        # 應回傳 list，且至少有一顆牙齒
        self.assertIsInstance(result, list, "應返回 list 格式")
        self.assertGreater(len(result), 0, "預期至少有一顆牙齒")
        for tooth in result:
            # 應包含 mid、enamel_left、pair_measurements 等欄位
            self.assertIn('mid', tooth, "缺少 mid 特徵點")
            self.assertIn('enamel_left', tooth, "缺少 enamel_left 特徵點")
            self.assertIn('pair_measurements', tooth, "缺少測量資料")
            # 驗證所有特徵點座標範圍
            img_h, img_w = self.normal_image.shape[:2]
            for key, pt in tooth.items():
                if isinstance(pt, tuple) and len(pt) == 2 and all(isinstance(x, int) for x in pt):
                    self.assertTrue(0 <= pt[0] <= img_w, f"{key} x超出範圍")
                    self.assertTrue(0 <= pt[1] <= img_h, f"{key} y超出範圍")
            # 測量值合理性
            for pair in tooth['pair_measurements']:
                if 'length' in pair:
                    self.assertGreater(pair['length'], 0, "長度應大於0")

    def test_missing_input_handling(self):
        """測試：異常輸入處理（全黑圖片/部分mask缺失）"""
        test_cases = [
            # (測試名稱, 輸入圖像, mock行為)
            ("全黑圖片", self.black_image, None),
            ("全白圖片", self.white_image, None),
        ]

        for case_name, test_image, mock_behavior in test_cases:
            with self.subTest(case_name=case_name):

                # 執行被測函數
                result = dental_estimation(
                    test_image,
                    self.component_model,
                    self.contour_model,
                    return_type='dict'
                )

                # 共同斷言
                self.assertIsInstance(result, list)
                self.assertEqual(len(result), 0, "全黑圖片/缺失mask時應回傳空list")

    def test_output_format_validation(self):
        """測試：不同 return_type 的輸出格式"""
        # 測試 CVAT 格式
        cvat_data = dental_estimation(
            self.normal_image,
            self.component_model,
            self.contour_model,
            return_type='cvat'
        )
        self.assertIsInstance(cvat_data, list, "CVAT 格式應為 list")
        self.assertTrue(
            any('points' in item for item in cvat_data),
            "CVAT 數據應至少包含一個有 'points' 的項目"
        )
        # 測試 image_array 格式
        img_array, _ = dental_estimation(
            self.normal_image,
            self.component_model,
            self.contour_model,
            return_type='image_array'
        )
        self.assertEqual(img_array.shape[:2], self.normal_image.shape[:2], "影像尺寸不符")

    def test_model_file_not_found(self):
        """測試：模型路徑不存在時應拋出 FileNotFoundError"""
        with self.assertRaises(FileNotFoundError):
            YOLO('./models/not_exist_model.pt')

    def test_image_file_not_found(self):
        """測試：圖片路徑不存在時應拋出 FileNotFoundError"""
        with self.assertRaises(FileNotFoundError):
            cv2.imread('./tests/files/not_exist_img.png')

if __name__ == '__main__':
    unittest.main()