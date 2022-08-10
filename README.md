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


## Introduction
Sepsis is life-threatening organ dysfunction caused by a dysregulated host response to infection. However, there is no gold-standard diagnosis of sepsis due to heterogeneity of the disease. Advances in -omics science, data science, and machine learning have generated evidence in sepsis, furthermore, the field of critical care. However, there was a lack of studies to investigate the longitudinal changes in septic shock, and studies to investigate the immune cell morphology changes. While the advance of a 3-D microscope to investigate cell morphology can differentiate the subtypes of the immune cell, the immune cell morphology changes with patient status were less studied. Therefore, this study aims to investigate the dynamic changes in cell morphology in septic shock. This study constructs a dataset based on different timepoint of septic shock patients who were admitted to the emergency room. Peripheral blood mononuclear cell (PBMC) was isolated and CD4+ and CD8+ T cells were negatively sorted. The 3D tomogram images of these cells were taken, and we validated the dataset using an artificial intelligence (AI) learning model to highlight its possibilities. 

## Dataset
3D tomocube raw image data can be download from [here](https://drive.google.com/drive/u/0/folders/1qYqS0kBQL9gVg3qescvCZmOVddbVMuN6).

**Table. Data distribution of each cell types timepoint dataset**
Patient ID | Time Point| CD8+ Tcell | CD4+ Tcell | Total cell count |
-------|:-------:|:--------:|:--------:|:--------:|
1 |1 |50 |50 | 100|
1 |2 |59 |91 | 150|
2 |1 |202 | 163|365 |
2 |2 |195 |60| 255|
3 |1 | 265| 222| 487|
3 |2 | 208| 190| 398|
4 |1 |121 |120 | 241|
4 |2 | 133|221 |354 |
5 |1 | 203|147 |350 |
5 |2 | 175|225 |400 |
6 |1 | 21| 64|85 |
6 |2 | 48|103 |151 |
7 |1 | 105|200 |305 |
7 |2 | 100|60 |160 |
8 |1 | 102| 101|203 |
8 |2 | 101|202 |303 |
Total| | 2088| 2219|4307

## Model
The model for discriminating the septic shock timepoints was developed with 3D Artificial Neural Network (ANN), the Convolutional Neural Network (CNN) based image recognition model. 
![densebb]("https://github.com/kimjh0107/2022_Gigascience/issues/7#issue-1334178258")
**Figure. The model architecture to discriminate the septic shock time points with 3D immune cell images**

More details about model are in:

(1) [Rapid species identification of pathogenic bacteria from a minute quantity exploiting three-dimensional quantitative phase imaging and artificial neural network](https://www.nature.com/articles/s41377-022-00881-x) 

(2) [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)



## Architecture 
![dense block1](https://user-images.githubusercontent.com/83206535/183028019-533bdfda-7379-45c9-a7e9-1f7feeddf4b9.png)

## Model result 
![cd8_ROC1](https://user-images.githubusercontent.com/83206535/183031818-eddfb5c6-9b69-4926-837e-c97c38b5a1a5.png)

# Blind test Result 
## (1) CD8 test 
