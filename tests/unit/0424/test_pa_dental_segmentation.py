import unittest
import numpy as np
import cv2
import os
import yaml
from ultralytics import YOLO
import sys

# 將專案路徑加入 sys.path，方便 import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.allocation.domain.pa_dental_segmentation.main import yolo_transform

class TestYoloTransform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 設定模型與設定檔路徑
        cls.model_path = './models/dentistry_pa-segmentation_yolov11x-seg-all_24.42.pt'
        cls.config_path = './conf/pa_segmentation_mask_color_setting.yaml'
        # 測試用圖片路徑
        cls.normal_image_path = './tests/files/raw_12.png'
        cls.black_image_path = './tests/files/black.png'
        # 檢查檔案是否存在
        for path in [cls.model_path, cls.config_path, cls.normal_image_path, cls.black_image_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"檔案不存在: {path}")
        # 載入模型與設定
        cls.model = YOLO(cls.model_path)
        with open(cls.config_path, 'r') as f:
            cls.plot_config = yaml.safe_load(f)
        # 載入圖片
        cls.normal_image = cv2.imread(cls.normal_image_path)
        cls.black_image = cv2.imread(cls.black_image_path)

    def test_yolo_transform_normal_image_dict(self):
        """測試：正常牙片 dict 輸出應包含主要 mask（如 Dentin）"""
        mask_dict = yolo_transform(self.normal_image, self.model, return_type='dict')
        self.assertIn('Dentin', mask_dict, '正常圖片應有 Dentin mask')
        self.assertTrue(mask_dict['Dentin'].any(), 'Dentin mask 應有內容')

    def test_yolo_transform_normal_image_image_array(self):
        """測試：正常牙片 image_array 輸出應無錯誤訊息且高度正確，寬度應增加（因有圖例）"""
        result_img, error_msg = yolo_transform(
            self.normal_image, self.model, return_type='image_array', plot_config=self.plot_config
        )
        self.assertEqual(error_msg, '', '正常圖片不應有錯誤訊息')
        self.assertEqual(result_img.shape[0], self.normal_image.shape[0], '高度應相同')
        self.assertGreater(result_img.shape[1], self.normal_image.shape[1], '寬度應增加（因圖例）')

    def test_yolo_transform_black_image_dict(self):
        """測試：全黑圖片 dict 輸出應為空"""
        mask_dict = yolo_transform(self.black_image, self.model, return_type='dict')
        self.assertEqual(len(mask_dict), 0, '全黑圖片應無任何 mask')

    def test_yolo_transform_black_image_image_array(self):
        """測試：全黑圖片 image_array 輸出應有錯誤訊息且為黑底"""
        result_img, error_msg = yolo_transform(
            self.black_image, self.model, return_type='image_array', plot_config=self.plot_config
        )
        self.assertIn('No segmenation', error_msg, '應回傳分割失敗訊息')
        self.assertLess(np.mean(result_img), 10, '全黑圖片應回傳近乎全黑影像')

    def test_yolo_transform_invalid_return_type(self):
        """測試：當 return_type 為 'image_array' 且無 plot_config 時應拋出 ValueError"""
        with self.assertRaises(ValueError):
            yolo_transform(
                self.normal_image, self.model, return_type='image_array', plot_config=None
            )

    def test_yolo_transform_normal_image_cvat(self):
        """測試：正常牙片 cvat 輸出應包含 yolov8_contents 且格式正確"""
        result = yolo_transform(self.normal_image, self.model, return_type='cvat')
        self.assertIn('yolov8_contents', result, 'cvat 輸出應含 yolov8_contents')
        yolov8_contents = result['yolov8_contents']
        self.assertIsInstance(yolov8_contents, list, 'yolov8_contents 應為 list')
        if yolov8_contents:
            item = yolov8_contents[0]
            self.assertIn('label', item)
            self.assertIn('confidence', item)
            self.assertIn('points', item)

    def test_yolo_transform_normal_image_cvat_mask(self):
        """測試：正常牙片 cvat_mask 輸出應包含 mask 欄位"""
        result = yolo_transform(self.normal_image, self.model, return_type='cvat_mask')
        self.assertIn('yolov8_contents', result, 'cvat_mask 輸出應含 yolov8_contents')
        yolov8_contents = result['yolov8_contents']
        self.assertIsInstance(yolov8_contents, list, 'yolov8_contents 應為 list')
        if yolov8_contents:
            item = yolov8_contents[0]
            self.assertIn('mask', item, 'cvat_mask 模式下應含 mask 欄位')

    def test_yolo_transform_normal_image_yolov8(self):
        """測試：正常牙片 yolov8 輸出應為 list 且每列第一個元素為 class_id"""
        result = yolo_transform(self.normal_image, self.model, return_type='yolov8')
        self.assertIn('yolov8_contents', result, 'yolov8 輸出應含 yolov8_contents')
        yolov8_contents = result['yolov8_contents']
        self.assertIsInstance(yolov8_contents, list, 'yolov8_contents 應為 list')
        if yolov8_contents:
            item = yolov8_contents[0]
            self.assertIsInstance(item, list, 'yolov8_contents 元素應為 list')
            self.assertIsInstance(item[0], int, '第一個元素應為 class_id (int)')

if __name__ == '__main__':
    unittest.main()
