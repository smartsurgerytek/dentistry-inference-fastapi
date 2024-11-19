# Periodontal disease
The project is focused on the analysis of periodontal disease. 
The aim is to identify the factors that contribute to the development of periodontal disease and to develop a predictive model that can be used to predict the likelihood of developing periodontal disease which says the "Ground rules".

## What is periodontal disease?
Periodontal disease is a group of diseases that affect the tissues of the gums. It is often called "gum disease" or "periodontitis". Periodontal disease is caused by bacteria that live in the gums. These bacteria can cause inflammation, bleeding, and other problems.


# Ground rules
According to 

```
Clinical application of the new classification of periodontal
diseases: Ground rules, clarifications and “gray zones”
```

Some of observation are given as follows and we hope to clarify and to verify by such the computer vision approaches. 

## Time-based stage (current running)

The **periapical size** is measured as 31mm x 41mm, and our primary goal is to determine the time-based stage of attachment loss (bone loss) by analyzing the periapical films.

### Normal Alveolar Bone Crest
The normal alveolar bone crest is typically located **1–2 mm** apical to the **cemento-enamel junction (CEJ)**. If bone loss has occurred, the alveolar bone crest is positioned more than **2 mm** apical to the CEJ. The **bone crest level** is defined as the point along the root where the intact lamina dura is present.

### Measuring Alveolar Bone Loss and Distance (ABLD)
The value of **Alveolar Bone Loss Distance (ABLD)** is determined using two key metrics:
1. **BL (Bone Loss)** – This is the length between the point **2 mm below the CEJ** and the **Alveolar Bone Crest (ALC)**.
2. **TR (Tooth Root Length)** – This is the distance from the CEJ to the **root apex (APEX)**.

The ABLD can be calculated using the following formula:

$$
\text{ABLD} = \left( \frac{\text{CEJ} - \text{ALC} - 2 \text{mm}}{\text{CEJ} - \text{APEX} - 2 \text{mm}} \right) \times 100\%
$$

### Classification Based on Bone Loss
- A **normal tooth** is characterized by the distance between the CEJ and the alveolar bone level (ABL) being **< 2 mm**, indicating no bone loss. In this case, the tooth is classified as **Stage 0** (no periodontal disease).
  
- A **distance of ≥ 2 mm** between the CEJ and the bone indicates the presence of periodontal disease, necessitating further classification into stages.

### Staging of Periodontal Disease
Based on the ABLD values and other clinical factors, the disease progression is classified into the following stages:

- **Stage I**: Mild bone loss  
- **Stage II**: Moderate bone loss  
- **Stage III**: Severe bone loss  
- **Stage IV**: Advanced bone loss (extensive damage)

The data will be categorized according to these stages to assess the severity of the periodontal disease.

**Zero**: the cemento-enamel junction (CEJ) and the alveolar bone level (bone) being < 2 mm

**I**: bone loss < 15%

**II**: bone loss ≥ 15% and <33%

**III** bone loss ≥ 33%


### The segmentation of dental components 
The dental components are segmented by the machine learning approach. 

By such the segmentation, the dental components are segmented into the following components: 

- **0**: Alveolar bone
- **1**: Caries
- **2**: Crown
- **3**: Dentin
- **4**: Enamel
- **5**: Implant
- **6**: Mandibular alveolar nerve
- **7**: Maxillary sinus
- **8**: Periapical lesion
- **9**: Post and core
- **10**: Pulp
- **11**: Restoration
- **12**: Root canal filling
- **13**: Background

### Post processing of segmentation

To enhance the segmentation, we will implement postprocessing techniques.

One notable issue is that the Pulp is occasionally segmented outside the Dentin region. To address this, we will detect and correct these instances by identifying the background and replacing it with Dentin, as detailed in the work: postprocessing_background_mask.ipynb.


## Exemplify the gray zones
Simple criteria may be not enough to verify the classification of periodontal diseases. For example, bone loss < 15% is sorted of initial periodontitis (I), is this correct? how is the bound can be determined in the clinical experience? or the better way is to using the bound width? such as 10%-20%? Also, what is the distribution of bone loss in the population? We can offer the better statistical analysis to help dentist have better decision. 

## FDI clustering
We aim to enhance the efficiency and convenience of dental workflows by implementing clustering for each FDI (Fédération Dentaire Internationale) position. By automating the organization and categorization of FDI positions, we hope to significantly reduce the time and effort required for administrative tasks, allowing dentists to focus more on patient care and less on documentation. To address the sorting problems, serval tricks can be discussed such as the multi-label regression or clustering.  

## Patient-based stage
My understanding is the observation is based on the patient's properties such as age, gender, smoking status, etc. It means the image is not the only factor to determine the periodontitis. The way to implement this is to use the patient's medical history and other factors (The score of periodontal from image presumably) is trees-model approach such as Random Forest, XGBoost, LGBM are desirable.


# DATA Aviability
check in our google cloud:
https://drive.google.com/drive/u/0/folders/1pVjfbgGWWcPv0x4HVd1HNvlm8Fwi5VNg

300.rar: contains dental position in remark.png and excel report.
raw_data_pytorch_4.34: contains raw data in our annotations labels and images.
split_data_pytorch_4.34: contains  training and validations in CoCo dataset format.