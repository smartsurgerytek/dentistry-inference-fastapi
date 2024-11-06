# periodontal disease
The project is focused on the analysis of periodontal disease. 
The aim is to identify the factors that contribute to the development of periodontal disease and to develop a predictive model that can be used to predict the likelihood of developing periodontal disease which says the "Ground rules".

## what is periodontal disease?
Periodontal disease is a group of diseases that affect the tissues of the gums. It is often called "gum disease" or "periodontitis". Periodontal disease is caused by bacteria that live in the gums. These bacteria can cause inflammation, bleeding, and other problems.


# ground rules
According to 

```
Clinical application of the new classification of periodontal
diseases: Ground rules, clarifications and “gray zones”
```

Some of observation are given as follows and we hope to clarify and to verify by such the computer vision approaches. 

## time-based stage (current running)

the preiapical size: 31mm x 41mm or 31mm x 41mm

Our main goal is to get the time-based stage by the attachment loss (bone loss) from the periapical films.

The distance between the cemento-enamel junction (CEJ) and the alveolar bone level (bone) being <2 mm indicates a normal tooth, and there is no need to classify periodontal disease so that is can be sorted of "zero" stage. 

A distance of ≥2.0 mm between the CEJ and the bone is considered indicative of periodontal disease, and further classification is required.
We will sort the data by phase of pathology into different stages like I, II, III(IV).

**Zero**: the cemento-enamel junction (CEJ) and the alveolar bone level (bone) being < 2 mm

**I**: bone loss < 15%

**II**: bone loss ≥ 15% and <33%

**III** bone loss ≥ 33%


### the segmentation of dental components 
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

### post processing of segmentation

To enhance the segmentation, we will implement postprocessing techniques.

One notable issue is that the Pulp is occasionally segmented outside the Dentin region. To address this, we will detect and correct these instances by identifying the background and replacing it with Dentin, as detailed in the work: postprocessing_background_mask.ipynb.


## Exemplify the gray zones
Simple criteria may be not enough to verify the classification of periodontal diseases. For example, bone loss < 15% is sorted of initial periodontitis (I), is this correct? how is the bound can be determined in the clinical experience? or the better way is to using the bound width? such as 10%-20%? Also, what is the distribution of bone loss in the population? We can offer the better statistical analysis to help dentist have better decision. 

## FDI clustering
We aim to enhance the efficiency and convenience of dental workflows by implementing clustering for each FDI (Fédération Dentaire Internationale) position. By automating the organization and categorization of FDI positions, we hope to significantly reduce the time and effort required for administrative tasks, allowing dentists to focus more on patient care and less on documentation. To address the sorting problems, serval tricks can be discussed such as the multi-label regression or clustering.  

## patient-based stage
My understanding is the observation is based on the patient's properties such as age, gender, smoking status, etc. It means the image is not the only factor to determine the periodontitis. The way to implement this is to use the patient's medical history and other factors (The score of periodontal from image presumably) is trees-model approach such as Random Forest, XGBoost, LGBM are desirable.


# DATA aviability
check in our google cloud:
https://drive.google.com/drive/u/0/folders/1pVjfbgGWWcPv0x4HVd1HNvlm8Fwi5VNg

300.rar: contains dental position in remark.png and excel report.
raw_data_pytorch_4.34: contains raw data in our annotations labels and images.
split_data_pytorch_4.34: contains  training and validations in CoCo dataset format.