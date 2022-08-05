# Immune cell morphology holotomography dataset for AI-driven assessment of sepsis patients
## Author: JongHyun Kim

This repository contains the training code for our paper entitled "Immune cell morphology holotomography dataset for AI-driven assessment of sepsis patients"

Under review by [Gigascience 2022 (IF:11.5)](https://academic.oup.com/gigascience).

## Abstract


## Preprocess image
![tomocube_workflow1](https://user-images.githubusercontent.com/83206535/183031529-892dd178-e08b-4efe-99e1-3d40037091c5.png)

## Architecture 
![dense block1](https://user-images.githubusercontent.com/83206535/183028019-533bdfda-7379-45c9-a7e9-1f7feeddf4b9.png)

## Model result 
![cd8_ROC1](https://user-images.githubusercontent.com/83206535/183031818-eddfb5c6-9b69-4926-837e-c97c38b5a1a5.png)

# Blind test Result 
## (1) CD8 test 

|blind_test|exclusion_Patient_ID|test_dataset|AUROC|AUPR|ACC|F1_score|loss
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|blind_test_{}|patient {}|CD8|||||
|""|""|CD8_blindtest|||||
|blind_test_{}|patient {}|CD8|||||
|""|""|CD8_blindtest|||||
|blind_test_{}|patient {}|CD8|||||
|""|""|CD8_blindtest|||||
|blind_test_{}|patient {}|CD8|||||
|""|""|CD8_blindtest|||||
|blind_test_{}|patient {}|CD8|||||
|""|""|CD8_blindtest|||||
|blind_test_{}|patient {}|CD8|||||
|""|""|CD8_blindtest|||||
|blind_test_{}|patient {}|CD8|||||
|""|""|CD8_blindtest|||||
|blind_test_{}|patient {}|CD8|||||
|""|""|CD8_blindtest|||||

## (2) CD4 test 

|blind_test|exclusion_Patient_ID|test_dataset|AUROC|AUPR|ACC|F1_score|loss
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|blind_test_{}|patient {}|CD4|||||
|""|""|CD4_blindtest|||||
|blind_test_{}|patient {}|CD4|||||
|""|""|CD4_blindtest|||||
|blind_test_{}|patient {}|CD4|||||
|""|""|CD4_blindtest|||||
|blind_test_{}|patient {}|CD4|||||
|""|""|CD4_blindtest|||||
|blind_test_{}|patient {}|CD4|||||
|""|""|CD4_blindtest|||||
|blind_test_{}|patient {}|CD4|||||
|""|""|CD4_blindtest|||||
|blind_test_{}|patient {}|CD4|||||
|""|""|CD4_blindtest|||||
|blind_test_{}|patient {}|CD4|||||
|""|""|CD4_blindtest|||||




