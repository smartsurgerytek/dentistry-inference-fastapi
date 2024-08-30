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
We will sort the data by phase of pathology into different stages like I, II, III(IV).
So far our main goal is to get the time-based stage by the attachment loss (bone loss) from the periapical films.

### the segmentation of dental components

### post processing of segmentation


## Exemplify the gray zones
Simple criteria may be not enough to verify the classification of periodontal diseases. For example, bone loss < 15% is sorted of initial periodontitis (I), is this correct? how is the bound can be determined in the clinical experience? or the better way is to using the bound width? such as 10%-20%? Also, what is the distribution of bone loss in the population? We can offer the better statistical analysis to help dentist have better decision. 

## FDI Regression

## patient-based stage
My understanding is the observation is based on the patient's properties such as age, gender, smoking status, etc. It means the image is not the only factor to determine the periodontitis. The way to implement this is to use the patient's medical history and other factors (The score of periodontal from image presumably) is trees-model approach such as Random Forest, XGBoost, LGBM are desirable.



