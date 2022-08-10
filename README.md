# Immune cell morphology holotomography dataset for AI-driven assessment of sepsis patients
## Author: JongHyun Kim

This repository contains the training code for our paper entitled "Immune cell morphology holotomography dataset for AI-driven assessment of sepsis patients"

Jong Hyun Kim, BS\*; MinDong Sung, MD\* (\* Authors contributed equally).

and its journal under review by [Gigascience 2022](https://academic.oup.com/gigascience).

**Now with memory-efficient implementation!** Please check the [code](https://github.com/kimjh0107/2022_Gigascience/tree/main/src) for more infomation.

The code for labeling tool is built on [Cell Morphology Labeling Tool](https://github.com/DigitalHealthcareLab/22CellMorphologyLabelingTool) by MD Sung.

## Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model](#model)
4. [Results on CD8+ Tcell](#results-on-CD8+)
5. [Results on CD4+ Tcell](#results-on-CD4+)


## Intro
Sepsis is life-threatening organ dysfunction caused by a dysregulated host response to infection. However, there is no gold-standard diagnosis of sepsis due to heterogeneity of the disease. Advances in -omics science, data science, and machine learning have generated evidence in sepsis, furthermore, the field of critical care. However, there was a lack of studies to investigate the longitudinal changes in septic shock, and studies to investigate the immune cell morphology changes. While the advance of a 3-D microscope to investigate cell morphology can differentiate the subtypes of the immune cell, the immune cell morphology changes with patient status were less studied. Therefore, this study aims to investigate the dynamic changes in cell morphology in septic shock. This study constructs a dataset based on different timepoint of septic shock patients who were admitted to the emergency room. Peripheral blood mononuclear cell (PBMC) was isolated and CD4+ and CD8+ T cells were negatively sorted. The 3D tomogram images of these cells were taken, and we validated the dataset using an artificial intelligence (AI) learning model to highlight its possibilities. 

## Preprocess image
![tomocube_workflow1](https://user-images.githubusercontent.com/83206535/183031529-892dd178-e08b-4efe-99e1-3d40037091c5.png)

## Architecture 
![dense block1](https://user-images.githubusercontent.com/83206535/183028019-533bdfda-7379-45c9-a7e9-1f7feeddf4b9.png)

## Model result 
![cd8_ROC1](https://user-images.githubusercontent.com/83206535/183031818-eddfb5c6-9b69-4926-837e-c97c38b5a1a5.png)

# Blind test Result 
## (1) CD8 test 
