# tests/test_schemas.py
import sys
import os
from pydantic import ValidationError

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.allocation.domain.pa_dental_measure.schemas import *

test_values = {
    "valid_numbers": [0, 1, 2, 3, 10, 20],
    "invalid_numbers": [None, -1, "invalid", [], False],  # 確保這些無效值會被正確處理
    "valid_stages": [0, 1, 2, 3, "I", "II", "III"],
    "invalid_stages": ["x", 4, None, [], -1]  # 同樣確保無效值被正確處理
}

def generate_test_data(valid=True):
    numbers = test_values["valid_numbers"] if valid else test_values["invalid_numbers"]
    stages = test_values["valid_stages"] if valid else test_values["invalid_stages"]
    return [
        {"side_id": num, "CEJ": (1, 2), "ALC": (3, 4), "APEX": (5, 6), "CAL": 1.5, "TRL": 2.5, "ABLD": 3.5, "stage": stage}
        for num in numbers for stage in stages
    ]

def test_dental_measurements():
    # 測試 dental_measurements 函數
    print("正在測試 dental_measurements 函數...")
    valid_data_list = generate_test_data(valid=True)
    invalid_data_list = generate_test_data(valid=False)

    # Valid data test
    for data in valid_data_list:
        measurement = DentalMeasurements(**data)
        assert measurement.side_id == data["side_id"]
        assert measurement.CEJ == data["CEJ"]
        assert measurement.stage in test_values["valid_stages"]

    # Invalid data test
    for i, invalid_data in enumerate(invalid_data_list):
        pass_bool=False
        try:
            DentalMeasurements(**invalid_data)
        except ValidationError as e:
            print(f"Test {i} validation failed as expected: {e}")
            pass_bool=True
        assert pass_bool, f"Test {i} 應該引發 ValueError"
            
    print("dental_measurements 測試通過！")

def test_stage_values():
    # 測試 stage_values 函數
    print("正在測試 stage_values 函數...")
    for stage in test_values["valid_stages"]:
        data = {"side_id": 1, "CEJ": (1, 2), "ALC": (3, 4), "APEX": (5, 6), "CAL": 1.5, "TRL": 2.5, "ABLD": 3.5, "stage": stage}
        measurement = DentalMeasurements(**data)
        assert measurement.stage == stage

    for invalid_stage in test_values["invalid_stages"]:
        data = {"side_id": 1, "CEJ": (1, 2), "ALC": (3, 4), "APEX": (5, 6), "CAL": 1.5, "TRL": 2.5, "ABLD": 3.5, "stage": invalid_stage}
        pass_bool=False
        try:
            DentalMeasurements(**data)
        except ValidationError as e:
            print(f"Test {invalid_stage} validation failed as expected: {e}")
            pass_bool=True
        assert pass_bool, f"Test {invalid_stage} 應該引發 ValueError"
    print("stage_values 測試通過！")

def test_dental_measurements_serialization():
    # 測試 dental_measurements 的序列化
    print("正在測試 dental_measurements 的序列化...")
    for stage in test_values["valid_stages"]:
        data = {"side_id": 1, "CEJ": (1, 2), "ALC": (3, 4), "APEX": (5, 6), "CAL": 1.5, "TRL": 2.5, "ABLD": 3.5, "stage": stage}
        measurement = DentalMeasurements(**data)
        serialized = measurement.model_dump()
        assert serialized["side_id"] == 1
        assert serialized["CEJ"] == (1, 2)
        assert serialized["stage"] == stage
    print("dental_measurements 序列化測試通過！")

def test_boundary_conditions():
    # 測試邊界條件
    print("正在測試邊界條件...")
    boundary_data = [
        {"side_id": 0, "CEJ": (0, 0), "ALC": (0, 0), "APEX": (0, 0), "CAL": 0.0, "TRL": 0.0, "ABLD": 0.0, "stage": 0},
        {"side_id": 1, "CEJ": (1, 1), "ALC": (1, 1), "APEX": (1, 1), "CAL": 1.0, "TRL": 1.0, "ABLD": 1.0, "stage": "I"},
        {"side_id": 10, "CEJ": (10, 10), "ALC": (10, 10), "APEX": (10, 10), "CAL": 10.0, "TRL": 10.0, "ABLD": 10.0, "stage": "III"}
    ]
    for data in boundary_data:
        measurement = DentalMeasurements(**data)
        assert measurement.side_id == data["side_id"]
        assert measurement.CEJ == data["CEJ"]
        assert measurement.ALC == data["ALC"]
        assert measurement.APEX == data["APEX"]
        assert measurement.CAL == data["CAL"]
        assert measurement.TRL == data["TRL"]
        assert measurement.ABLD == data["ABLD"]
        assert measurement.stage == data["stage"]
    print("邊界條件測試通過！")

def test_exception_handling():
    # 測試異常處理
    print("正在測試異常處理...")
    exception_data = [
        {"side_id": "invalid", "CEJ": (1, 2), "ALC": (3, 4), "APEX": (5, 6), "CAL": 1.5, "TRL": 2.5, "ABLD": 3.5, "stage": 0},
        {"side_id": 1, "CEJ": "invalid", "ALC": (3, 4), "APEX": (5, 6), "CAL": 1.5, "TRL": 2.5, "ABLD": 3.5, "stage": 0},
        {"side_id": 1, "CEJ": (1, 2), "ALC": "invalid", "APEX": (5, 6), "CAL": 1.5, "TRL": 2.5, "ABLD": 3.5, "stage": 0},
        {"side_id": 1, "CEJ": (1, 2), "ALC": (3, 4), "APEX": "invalid", "CAL": 1.5, "TRL": 2.5, "ABLD": 3.5, "stage": 0},
        {"side_id": 1, "CEJ": (1, 2), "ALC": (3, 4), "APEX": (5, 6), "CAL": "invalid", "TRL": 2.5, "ABLD": 3.5, "stage": 0},
        {"side_id": 1, "CEJ": (1, 2), "ALC": (3, 4), "APEX": (5, 6), "CAL": 1.5, "TRL": "invalid", "ABLD": 3.5, "stage": 0},
        {"side_id": 1, "CEJ": (1, 2), "ALC": (3, 4), "APEX": (5, 6), "CAL": 1.5, "TRL": 2.5, "ABLD": "invalid", "stage": 0},
        {"side_id": 1, "CEJ": (1, 2), "ALC": (3, 4), "APEX": (5, 6), "CAL": 1.5, "TRL": 2.5, "ABLD": 3.5, "stage": "invalid"}
    ]
    for i, data in enumerate(exception_data):
        pass_bool=False
        try:
            DentalMeasurements(**data)
        except ValidationError as e:
            print(f"Test {i} validation failed as expected: {e}")
            pass_bool=True
        assert pass_bool, f"Test {i} 應該引發 ValidationError"
    print("異常處理測試通過！")

# 執行所有測試
if __name__ == "__main__":
    print("開始執行所有測試...")
    test_dental_measurements()
    test_stage_values()
    test_dental_measurements_serialization()
    test_boundary_conditions()
    test_exception_handling()
    print("所有測試執行完畢！")