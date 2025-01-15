
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from src.allocation.domain.dental_measure.main import *

def restructure_dataframe(df):
    """重構 DataFrame 結構"""
    if 'dentin_id' in df.columns:
        df = df.drop(columns=['dentin_id'])
    df['tooth_id'] = range(1, len(df) + 1)
    df_left = df[['tooth_id', 'mid', 'enamel_left', 'gum_left', 'dentin_left']]
    df_left.columns = ['tooth_id', 'mid', 'enamel', 'gum', 'dentin']
    df_right = df[['tooth_id', 'mid', 'enamel_right', 'gum_right', 'dentin_right']]
    df_right.columns = ['tooth_id', 'mid', 'enamel', 'gum', 'dentin']
    return pd.concat([df_left, df_right]).sort_values(by=['tooth_id', 'enamel']).reset_index(drop=True)

def combine_and_clean_dataframe(df_combined):
    """合併 DataFrame 並清理資料"""
    dentin_id = 1
    dentin_ids = [dentin_id]
    for i in range(1, len(df_combined)):
        if df_combined.iloc[i]['dentin'] == df_combined.iloc[i - 1]['dentin']:
            dentin_ids.append(dentin_id)
        else:
            dentin_id += 1
            dentin_ids.append(dentin_id)

    df_combined['dentin_id'] = dentin_ids
    df_combined[['enamel_x', 'enamel_y']] = pd.DataFrame(df_combined['enamel'].tolist(), index=df_combined.index)
    df_combined[['gum_x', 'gum_y']] = pd.DataFrame(df_combined['gum'].tolist(), index=df_combined.index)
    df_combined[['dentin_x', 'dentin_y']] = pd.DataFrame(df_combined['dentin'].tolist(), index=df_combined.index)
    return df_combined.drop(columns=['mid', 'enamel', 'gum', 'dentin'])

def prepare_true_dataframe(correct_df):
    """準備真實資料的 DataFrame"""
    df_true_cleaned = correct_df.drop(columns=[col for col in ['length', 'stage'] if col in correct_df.columns])
    #df_true_cleaned = correct_df.rename(columns={'珐瑯質跟象牙質交接點x':'enamel_x', "珐瑯質跟象牙質交接點y":"enamel_y"})
    df_true_cleaned['true_stage'] = df_true_cleaned.apply(calculate_true_stage, axis=1)
    return df_true_cleaned

def merge_dataframes(df_cleaned, df_true_cleaned):
    """合併預測資料與真實資料"""
    df_merged_list = []
    for index, row_true in df_true_cleaned.iterrows():
        if df_cleaned.empty:
            row_true_reset = df_true_cleaned.iloc[[index]].reset_index(drop=True)
            #row_true_reset.loc[0, ['class', 'denture']] = np.nan
            empty_predicted_columns = pd.DataFrame(np.nan, index=row_true_reset.index, columns=df_cleaned.columns)
            merged_row = pd.concat([row_true_reset, empty_predicted_columns], axis=1)
        else:
            distances = df_cleaned.apply(lambda row_cleaned: calculate_distance(row_true, row_cleaned), axis=1)
            closest_index = distances.idxmin()
            min_distance = distances[closest_index]

            if min_distance > DISTANCE_THRESHOLD:
                row_true_reset = df_true_cleaned.iloc[[index]].reset_index(drop=True)
                #row_true_reset.loc[0, ['class', 'denture']] = np.nan
                empty_predicted_columns = pd.DataFrame(np.nan, index=row_true_reset.index, columns=df_cleaned.columns)
                merged_row = pd.concat([row_true_reset, empty_predicted_columns], axis=1).reset_index(drop=True)
            else:
                row_true_reset = df_true_cleaned.iloc[[index]].reset_index(drop=True)
                row_cleaned_reset = df_cleaned.loc[[closest_index]].reset_index(drop=True)
                merged_row = pd.concat([row_true_reset, row_cleaned_reset], axis=1).reset_index(drop=True)
                df_cleaned = df_cleaned.drop(closest_index)

        df_merged_list.append(merged_row)
    return pd.concat(df_merged_list, ignore_index=True)


# 計算基於 ['enamel_x'] 和 ['enamel_y'] 的距離函數
def calculate_distance(row_true, row_cleaned):
    true_values = np.array([row_true['enamel_x'], row_true['enamel_y']])# enamel: 珐瑯質跟象牙質交接點
    cleaned_values = np.array([row_cleaned['enamel_x'], row_cleaned['enamel_y']])
    return np.linalg.norm(true_values - cleaned_values)



# 計算 true_stage 並轉換為毫米
def calculate_true_stage(row):
    enamel_x, enamel_y = row['enamel_x'], row['enamel_y']
    gum_x, gum_y = row['gum_x'], row['gum_y']
    dentin_x, dentin_y = row['dentin_x'], row['dentin_y']
    
    # 圖片比例轉換 (像素轉毫米)
    x_scale = 40 / 1280
    y_scale = 30 / 960

    # 將像素轉換為毫米座標
    enamel_x_mm = enamel_x * x_scale
    enamel_y_mm = enamel_y * y_scale
    gum_x_mm = gum_x * x_scale
    gum_y_mm = gum_y * y_scale
    dentin_x_mm = dentin_x * x_scale
    dentin_y_mm = dentin_y * y_scale
    
    # 計算距離 (毫米)
    CEJ_ALC = np.sqrt((enamel_x_mm - gum_x_mm) ** 2 + (enamel_y_mm - gum_y_mm) ** 2)
    CEJ_APEX = np.sqrt((enamel_x_mm - dentin_x_mm) ** 2 + (enamel_y_mm - dentin_y_mm) ** 2)
    
    # 計算 ABLD
    ABLD = ((CEJ_ALC - 2) / (CEJ_APEX - 2)) * 100
    
    # 判定 true_stage
    if ABLD <= 0:
        stage = "0"
    elif ABLD <= 15:
        stage = "I"
    elif ABLD <= 33.3:
        stage = "II"
    else:
        stage = "III"
    
    return stage               
# 計算 percentage 和期數，預測資料使用
def calculate_predicted_stage(row):
    enamel_x, enamel_y = row['enamel_x'], row['enamel_y']
    gum_x, gum_y = row['gum_x'], row['gum_y']
    dentin_x, dentin_y = row['dentin_x'], row['dentin_y']
    
    # 圖片比例轉換 (像素轉毫米)
    x_scale = 40 / 1280
    y_scale = 30 / 960

    # 將像素轉換為毫米座標
    enamel_x_mm = enamel_x * x_scale
    enamel_y_mm = enamel_y * y_scale
    gum_x_mm = gum_x * x_scale
    gum_y_mm = gum_y * y_scale
    dentin_x_mm = dentin_x * x_scale
    dentin_y_mm = dentin_y * y_scale
    
    # 計算距離 (毫米)
    CEJ_ALC = np.sqrt((enamel_x_mm - gum_x_mm) ** 2 + (enamel_y_mm - gum_y_mm) ** 2)
    CEJ_APEX = np.sqrt((enamel_x_mm - dentin_x_mm) ** 2 + (enamel_y_mm - dentin_y_mm) ** 2)
    
    # 計算 ABLD
    ABLD = ((CEJ_ALC - 2) / (CEJ_APEX - 2)) * 100
    
    # 判定 predicted_stage
    if ABLD <= 0:
        stage = "0"
    elif ABLD <= 15:
        stage = "I"
    elif ABLD <= 33.3:
        stage = "II"
    else:
        stage = "III"
    
    return ABLD, stage

def process_and_save_predictions(predictions, correct_df):
    """處理並儲存預測結果"""
    df_true_cleaned = prepare_true_dataframe(correct_df)
    if not predictions:
        df_pred=pd.DataFrame()
        # col=pd.Series([np.nan] * len(df_true_cleaned['true_stage']))
        # df_pred['predicted_stage']=col
        return df_pred , df_true_cleaned
        #pd.Series(np.zeros(len(df_true_cleaned['true_stage'])))
    sorted_predictions = sorted(predictions, key=lambda x: x['mid'][0])

    df = pd.DataFrame(sorted_predictions)
    df = restructure_dataframe(df)
    df_combined = combine_and_clean_dataframe(df)
    df_cleaned = df_combined.dropna()
    if len(df_cleaned) != 0:
        percentage_predstage_tuple=df_cleaned.apply(calculate_predicted_stage, axis=1)
        #df_cleaned['percentage'], df_cleaned['predicted_stage'] = zip(*percentage_predstage_tuple) # 計算 stage
        df_cleaned_copy=df_cleaned.copy()
        df_cleaned_copy['percentage'], df_cleaned_copy['predicted_stage'] = zip(*percentage_predstage_tuple)

    

    #df_merged = merge_dataframes(df_cleaned, df_true_cleaned)
    # df_merged = df_merged.rename(columns={'牙齒ID（相對該張影像的順序ID即可、從左至右）':'tooth_id', 
    #                     "牙尖ID（從左側至右側，看是連線到哪一個牙尖端）":"dentin_id",
    #                     "珐瑯質跟象牙質交接點x":"enamel_x", "珐瑯質跟象牙質交接點y":"enamel_y",
    #                     "牙齦交接點x":"gum_x" , "牙齦交接點y":"gum_y",
    #                     "牙本體尖端點x":"dentin_x" , "牙本體尖端點y":"dentin_y" ,
    #                     "長度":"length","stage":"true_stage"
    #                     })
    return df_cleaned_copy, df_true_cleaned

# 設定主資料夾路徑
main_folder_path = './datasets/300'

# # 初始化空的 DataFrame 來儲存篩選後的結果
# filtered_data = pd.DataFrame()
# 定義分期對應
stage_mapping = {'0': 0, 'I': 1, 'II': 2, 'III': 3}
reverse_stage_mapping = {v: k for k, v in stage_mapping.items()}

tooth_col_mapping={
    '牙齒ID（相對該張影像的順序ID即可、從左至右）':'tooth_id',
    "牙尖ID（從左側至右側，看是連線到哪一個牙尖端）":"dentin_id",
    "珐瑯質跟象牙質交接點x":"enamel_x", 
    "珐瑯質跟象牙質交接點y":"enamel_y",
    "牙齦交接點x":"gum_x" ,
    "牙齦交接點y":"gum_y",
    "牙本體尖端點x":"dentin_x" , 
    "牙本體尖端點y":"dentin_y" ,
    "長度":"length"
}

# Function to apply value_mapping based on the type (Series or DataFrame)
def apply_value_mapping(df, value_mapping):
    if isinstance(df, pd.Series):
        # Apply value_mapping to each element of the Series
        return df.apply(lambda x: value_mapping.get(x, x))
    elif isinstance(df, pd.DataFrame):
        # Apply value_mapping to each element of the DataFrame
        return df.applymap(lambda x: value_mapping.get(x, x))
def test_detntalMeasure_performance(config=None):
    if config is not None:
        for key, value in config.items():
            globals()[key] = value        
    scale_x=30/960
    scale_y=40/1080

    # 遍歷所有子資料夾
    df_pred_list=[]
    df_true_list=[]
    for folder_name in os.listdir(main_folder_path):
        excel_path=os.path.join(main_folder_path, folder_name, f'analysis_{folder_name}.xlsx')
        raw_image_path=os.path.join(main_folder_path, folder_name, f'raw_{folder_name}.png')

        if not os.path.isfile(excel_path) or not os.path.isfile(raw_image_path):
            continue

        df = pd.read_excel(excel_path)
        df_true = df.rename(columns=tooth_col_mapping)

        image=cv2.imread(raw_image_path)

        estimation_results=dental_estimation(image, scale=(scale_x, scale_y), return_type='dict')
        estimation_results = sorted(estimation_results, key=lambda x: x['mid'][0])

        df_pred, df_true=process_and_save_predictions(estimation_results, df_true)
        df_merged=merge_dataframes(df_pred, df_true)

        if df_pred.empty:
            col=pd.Series([np.nan] * len(df_true['true_stage']))
            df_pred_list.append(col)
            df_true_list.append(df_true['true_stage'])        
        else:    
            df_pred_list.append(df_merged['predicted_stage'])
            df_true_list.append(df_merged['true_stage'])

    value_mapping = {
        "0": 0,
        "I": 1,
        "II": 2,
        "III": 3
    }

    # Concatenate lists of DataFrames into single DataFrames for df_pred and df_true
    df_pred = pd.concat(df_pred_list, ignore_index=True)
    df_true = pd.concat(df_true_list, ignore_index=True)

    # Fill NaN values with -1
    df_pred = df_pred.fillna(-1)
    df_true = df_true.fillna(-1)

    # Apply value_mapping to both df_pred and df_true
    df_pred = apply_value_mapping(df_pred, value_mapping)
    df_true = apply_value_mapping(df_true, value_mapping)

    cm = confusion_matrix(df_true, df_pred)
    classes = np.unique(df_true)
    y_true_bin = label_binarize(df_true, classes=classes)
    y_pred_bin = label_binarize(df_pred, classes=classes)
    roc_auc = roc_auc_score(y_true_bin, y_pred_bin, multi_class='ovr')

    print('confusion_matrix', cm)
    print('roc_auc', roc_auc)
    return roc_auc

if __name__ == "__main__":
    test_detntalMeasure_performance()