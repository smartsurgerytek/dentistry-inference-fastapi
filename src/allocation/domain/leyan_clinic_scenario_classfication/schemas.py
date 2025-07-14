from typing import Dict
from pydantic import BaseModel, Field

class LeyanClinicScenarioClassificationResponse(BaseModel):
    """
    定義了「樂衍診所場景分類模型」的完整 API 回應結構。
    
    這個結構包含了從請求追蹤、模型預測結果、各類別信賴分數，
    到最終處理狀態的全部資訊，旨在提供一個清晰、自解釋的資料契約。
    """
    request_id: int = Field(
        ...,
        description=(
            "對應前端發起『樂衍診所拍照流程』的唯一請求識別碼。"
            "此 ID 由後端主服務在接收到圖片時生成，用於追蹤單次推理的請求。"
            "此值應為一個正整數。"
            "特殊值 `0` 表示這是一次測試或調試請求，其結果將不會被寫入資料庫。"
        ),
        examples=[1689340800123]
    )

    predicted_class: str = Field(
        ...,
        description=(
            "模型對使用者上傳影像所預測出的最可能的牙科場景標籤。"
            "這些標籤對應樂衍診所定義的標準拍攝角度或患者口部狀態。"
            "例如：`gag_open` 表示患者使用張口器並張開嘴，`left_45` 表示患者臉部朝左側轉45度。"
            "標籤的值為："
            "(人臉部分)'close','open','smile','left_45','left_90','right_45','right_90'"
            "(牙齒部分)'top','lower','left','right','gag_open','gag_bite'"
            "(其他部分)'other'。"
        ),
        examples=["gag_open"]
    )

    scores: Dict[str, float] = Field(
        ...,
        description=(
            "包含所有可能的分類標籤及其對應的信賴分數。"
            "分數的範圍在 0.0 到 1.0 之間，代表模型對該分類的信心程度。"
            "所有分數的總和約等於 1.0 。"
        ),
        examples=[{
            'close': 0.01, 'gag_bite': 0.02, 'gag_open': 0.85, 'left': 0.01, 
            'left_45': 0.01, 'left_90': 0.01, 'lower': 0.01, 'open': 0.01, 
            'other': 0.01, 'right': 0.01, 'right_45': 0.01, 'right_90': 0.01, 
            'smile': 0.01, 'top': 0.01
        }]
    )

    message: str = Field(
        ...,
        description=(
            "### 處理結果訊息\n"
            "描述本次請求從圖片接收到推理完成的最終狀態。\n\n"
            "**可能的成功訊息：**\n"
            "- `Classification completed successfully`: 影像符合要求且模型推理正常完成。\n\n"
            "**可能的警告或錯誤訊息（業務邏輯層面）：**\n"
            "- `Face not detected in the image`: 影像中未偵測到人臉，無法進行分類。\n"
            "- `Image resolution too low`: 影像解析度過低，可能影響模型準確度。\n"
            "- `Inference server error`: 模型推理服務器發生內部錯誤。\n\n"
            "**注意：** 此處的錯誤是指業務邏輯上的失敗。網路或請求格式錯誤會由 FastAPI 直接以 4xx/5xx HTTP 狀態碼回應。"
        ),
        examples=["Classification completed successfully"]
    )

