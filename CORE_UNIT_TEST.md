# 以下是這 5 個 unit test 的測試內容摘要：


---

## 1. test_pa_dental_segmentation.py

**測試內容：**

- 驗證正常牙片影像輸入時，`yolo_transform` 回傳的 dict 包含主要 mask（如 Dentin），且 mask 有內容。
- 驗證正常牙片影像輸入時，`yolo_transform` 回傳的 image_array 格式無錯誤訊息，高度與原圖一致，寬度因圖例而變大。
- 驗證全黑圖片輸入時，dict 輸出為空（無任何 mask）。
- 驗證全黑圖片輸入時，image_array 格式有分割失敗訊息，且回傳影像接近全黑。
- 驗證當 return_type 為 image_array 且未提供 plot_config 時，會拋出 ValueError。
- 驗證正常牙片影像輸入時，`yolo_transform` 回傳 cvat 格式，包含 yolov8_contents，且內容格式正確（含 label、confidence、points）。
- 驗證正常牙片影像輸入時，`yolo_transform` 回傳 cvat_mask 格式，yolov8_contents 內每個 item 含有 mask 欄位。
- 驗證正常牙片影像輸入時，`yolo_transform` 回傳 yolov8 格式，yolov8_contents 為 list，且每列第一個元素為 class_id（int）。
- 驗證未知 return_type 時，會走 fallback，回傳 class_names 與 yolov8_contents。

---

## 2. test_pano_caries_detection.py

**測試內容：**

- 驗證全黑圖片輸入時，齲齒檢測結果為 None，訊息為 "No caries found"。
- 驗證正常牙片圖片輸入時，齲齒檢測會有結果，且回傳型別為 PIL Image。

---

## 3. test_pano_fdi_segmentation.py

**測試內容：**

- 驗證正常牙片圖片輸入時，image_array 格式有結果且無明顯錯誤訊息。
- 驗證正常牙片圖片輸入時，cvat 格式結果為 dict，且 yolov8_contents 有內容。
- 驗證損壞圖片或非圖片檔案時，能正確處理（回傳 None、錯誤訊息或噴例外）。
- 驗證自訂 plot_config 時，image_array 格式能正常輸出。
- 驗證全黑圖片時，image_array 格式回傳特定錯誤訊息，cvat 格式 yolov8_contents 為空。

---

## 4. test_pa_pano_classification.py

**測試內容：**

- 驗證正常牙片圖片輸入時，分類結果為 "periapical film" 或 "panoramic x-ray"，且分數為 float。
- 驗證全黑圖片輸入時，分類結果同樣為 "periapical film" 或 "panoramic x-ray"，且分數為 float。

---

## 5. test_pa_dental_measure.py

**測試內容：**

- 驗證正常牙片圖片輸入時，dental_estimation 回傳 list 格式，且至少有一顆牙齒，並檢查每顆牙齒的特徵點與測量值合理性。
- 驗證全黑圖片輸入時，回傳 list 格式且長度為 0（無特徵點）。
- 驗證不同 return_type（dict、cvat、image_array）時的輸出格式與內容正確性。
- 模擬部分 mask 缺失時，檢查容錯與回傳結果。
- 驗證模型或圖片路徑不存在時會拋出 FileNotFoundError。

---