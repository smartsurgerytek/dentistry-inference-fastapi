import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch

def generate_filename_from_path(input_file_path):
    # 確保輸入的檔案路徑是絕對路徑
    input_file_path = os.path.abspath(input_file_path)
    
    # 從路徑中提取檔案名稱和副檔名
    base_name = os.path.basename(input_file_path)
    
    # 分離檔名和副檔名
    name, _ = os.path.splitext(base_name)
    
    return name

def plot_confusion_matrix(confusion_matrix, save_path, title_name, labels, plot_type='d'):

    if plot_type=='d':
        confusion_matrix=confusion_matrix.astype(int)
        

    ax=sns.heatmap(confusion_matrix,
                    annot=True,
                    fmt=plot_type,
                    cmap='Blues',
                    cbar=True,
                    xticklabels=labels,
                    yticklabels=labels)

    # 添加標題和軸標籤
    plt.title(title_name, fontsize=8)
    plt.xlabel('True Labels')
    plt.ylabel('Predicted Labels')
    for text in ax.texts:
        text.set_fontsize(6)
    plt.tight_layout()        
    plt.savefig(save_path)
    plt.clf()
    
def generate_f1_scores(confusion_matrix, labels):
    f1_scores = {}

    for i, label_name in enumerate(labels):
        TP = confusion_matrix[i, i]
        FP = confusion_matrix[:, i].sum() - TP
        FN = confusion_matrix[i, :].sum() - TP
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores[label_name] = f1
    
    return f1_scores

def generate_dsc_scores(confusion_matrix, labels):
# 計算每個標籤的 DSC
    dsc_scores = {}

    for i, label_name in enumerate(labels):
        TP = confusion_matrix[i, i]
        FP = confusion_matrix[:, i].sum() - TP
        FN = confusion_matrix[i, :].sum() - TP
        
        dsc = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
        dsc_scores[label_name] = dsc
    return dsc_scores
    
def generate_model_report(model, model_path, yaml_path, save_val_path, full_name, remove_indexes=[], device=[0]):

    if device=='cpu' or not torch.cuda.is_available():
        device = torch.device('cpu')        

    result=model.val(data=yaml_path, device=device)

    labels_dict=result.names
    labels=list(labels_dict.values())
    labels.append('background')    
    model_map_50_95=result.box.maps

    confusion_matrix=result.confusion_matrix.matrix


    
    if remove_indexes:
        confusion_matrix = np.delete(confusion_matrix, remove_indexes, axis=0)
        confusion_matrix = np.delete(confusion_matrix, remove_indexes, axis=1)
        labels=np.delete(labels, remove_indexes).tolist()
        model_map_50_95= np.delete(model_map_50_95, remove_indexes)

    average_map_50_95=sum(model_map_50_95)/len(model_map_50_95)
    average_map_50=result.box.map50
    print(f'average_map_50: {average_map_50}')
    f1_scores_dict=generate_f1_scores(confusion_matrix, labels)

    dsc_scores_dict=generate_dsc_scores(confusion_matrix, labels)

    confusion_matrix_recall=np.nan_to_num(confusion_matrix.copy().astype('float') / confusion_matrix.copy().astype('float').sum(axis=0))

    confusion_matrix_precision=np.nan_to_num(confusion_matrix.copy().astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis])


    
    plot_confusion_matrix(confusion_matrix=confusion_matrix, 
                          save_path=os.path.join(save_val_path,f'{full_name}_confusion_matrix.png'), 
                          title_name=f'{full_name} CM',
                          labels=labels,
                          )
    plot_confusion_matrix(confusion_matrix=confusion_matrix_precision, 
                          save_path=os.path.join(save_val_path,f'{full_name}_confusion_matrix_precision.png'), 
                          title_name=f'{full_name} precision',
                          labels=labels,
                          plot_type='.2f'
                          )
    plot_confusion_matrix(confusion_matrix=confusion_matrix_recall, 
                          save_path=os.path.join(save_val_path,f'{full_name}_confusion_matrix_recall.png'), 
                          title_name=f'{full_name} recall',
                          labels=labels,
                          plot_type='.2f'
                          )

    model_name=generate_filename_from_path(model_path)

    if 'background' in labels:# note that it is background instead of 'B'ackground
        labels=labels[:-1]

    if len(labels)!=len(model_map_50_95):
        raise ValueError('len labels is not equal to model_map_50_9')

    df = pd.DataFrame({
        'Label': labels,
        f'{model_name} map_50_95': model_map_50_95,
    })
    markdown_table = df.to_markdown(index=False)

    df2 = pd.DataFrame(list(f1_scores_dict.items()), columns=['Label', 'Score'])

    markdown_table2 = df2.to_markdown(index=False)

    df3 = pd.DataFrame(list(dsc_scores_dict.items()), columns=['Label', 'Score'])
    markdown_table3 = df3.to_markdown(index=False)

    now = datetime.now()
    current_time=now.strftime("%Y-%m-%d %H:%M:%S")
    
    with open(save_val_path+'/'+full_name+'_report.md', 'w', encoding='utf-8') as file:
        file.write(f'# Auto generate report {full_name} \n\n')
        file.write(f'At the current time: {current_time} \n\n')
        file.write(f'## Validations overview {full_name} \n\n')
        file.write(f'The average_map_50: {average_map_50}  \n\n')
        file.write(f'The average_map_50_95: {average_map_50_95}  \n\n')
        file.write('## Label Distribution Map 50_95\n\n')
        file.write(markdown_table)
        file.write('\n\n')
        file.write('## Label Distribution f1_scores\n\n')
        file.write(markdown_table2)
        file.write('\n\n')
        file.write('## Label Distribution DSC \n\n')
        file.write(markdown_table3)
        file.write('\n\n')                
        file.write('## Confusion matrix\n\n')
        file.write(f'## ![image]({full_name}_confusion_matrix.png) \n\n') 
        file.write('## Confusion matrix precision\n\n')   
        file.write(f'## ![image]({full_name}_confusion_matrix_precision.png) \n\n')  
        file.write('## Confusion matrix recall\n\n')   
        file.write(f'## ![image]({full_name}_confusion_matrix_recall.png) \n\n')  
    df.to_csv(save_val_path+'/'+f'{full_name}.csv', index=False)
    print(f"save report in {save_val_path}/{full_name}_report.md")

def test_model_performance():
    model_path='./models/dentistry_yolov11x-seg-all_4.42.pt'
    model=YOLO(model_path)
    yaml_path='./conf/dentistry.yaml'
    project_name='dentistry'
    model_name='yolo11x-seg'
    feature_name='randomVal640'
    current_version='4.42.0'
    full_name=project_name+'_'+model_name+'_'+feature_name+'_'+current_version

    generate_model_report(model=model, 
                          model_path=model_path,
                        yaml_path=yaml_path, 
                        save_val_path='./docs', 
                        full_name=full_name,
                        remove_indexes=[13]
                        )

if __name__=='__main__':
    test_model_performance()