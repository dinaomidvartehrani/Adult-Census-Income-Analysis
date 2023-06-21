# COMP 6721 - Project : Adult Census Income Analysis
## Group 06 

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Description](#description)
- [Dataset](#dataset)
- [Models](#models)
- [Files and Reports](#Files and Reports)
- [Contact](#contact)

## Description

The goal is to predict income levels (<=50K or >50K) based on various factors, including age, education, occupation, marital status, etc. 
The challenges to be overcome include handling missing data, class imbalance, feature engineering, and interpreting the results. 
The expectations are to develop a model, identify influential factors, evaluate model performance, compare different methodologies, and generate actionable insights.


## Dataset
The data is the Adult Census Income dataset from Kaggle. It contains approximately 48,842 instances, each with 15 features and a target variable indicating income level. 
The features cover a broad spectrum of demographic and work-related information. here is the link to the Kaggle where you can find this dataset and download it. https://www.kaggle.com/datasets/uciml/adult-census-income?resource=download; however you could find the sample test dataset which is the csv file on our github. 


## Models
The proposed methodology includes data preprocessing (handling missing values, encoding categorical variables, normalizing numerical features), exploratory data analysis, 
model selection, model evaluation, cross-validation and hyperparameter tuning, and analysis of results.
The goal is not just to develop an effective model but also to understand the variables influencing income levels and provide insights for policy-making.


## Files and Reports
One-page proposal 
Progress Report 
Final Report 
Readme ( link to github and link be video ) 
Sample test dataset
One-page contribution
Presentation Slides
Final Presentation ( video ) 

git hub :: high level description / presentation of the project 
requirements to run your Python code 
instruction on how to train / validate your model 
instruction on how to run the pre-trained model on the provided sample test dataset 
your source code package in Scikit-learn and PyTorch 
description on how to obtain the dataset from an available download link 







DNN Model.ipynb
DNN.ipynb
LICENSE
Pre_processing.ipynb
Presentation-withsemi.pptx
Presentation.pptx
Progress_G06.pdf
Proposal_G06.pdf
Semi Supervised Learnng.ipynb
Supervised learning Classification with Decision Trees.ipynb
adult.csv.



## Contact

For any inquiries or questions regarding the project, please feel free to contact:

- Dina Omidvar: [dinaomidvar1377@gmail.com]
- Niloofar Tavakolian: [niltavakolian@gmail.com]
- Nastaran Naseri: [naseri.nastaran@hotmail.com]
- Seyedeh Mojdeh Haghighat Hosseini: [mozhde.2h@gmail.com]








# COMP 6721 - Project : Adult Census Income Analysis
## Group 06 

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Description](#description)
- [High-Level Description](#high-level-description)
- [Requirements](#requirements)
- [Training and Validation](#training-and-validation)
- [Running Pre-trained Model](#running-pre-trained-model)
- [Source Code](#source-code)
- [Dataset](#dataset)
- [References](#references)

## Description

This repository contains the code and resources for our group project in [Course Name]. The project focuses on income classification using a supervised approach with decision tree, semi-supervised learning, and deep neural network models. The objective is to classify income based on a dataset containing 15 features.

## High-Level Description

The project aims to classify income levels based on socio-economic factors. It utilizes a decision tree model, a semi-supervised learning algorithm, and a deep neural network model. The decision tree model makes decisions based on specific features, the semi-supervised learning algorithm leverages both labeled and unlabeled data, and the deep neural network model extracts complex patterns and representations from the data. Each model demonstrates significant accuracy in income classification.
### Description

The goal is to predict income levels (<=50K or >50K) based on various factors, including age, education, occupation, marital status, etc. 
The challenges to be overcome include handling missing data, class imbalance, feature engineering, and interpreting the results. 
The expectations are to develop a model, identify influential factors, evaluate model performance, compare different methodologies, and generate actionable insights.




### Models
The proposed methodology includes data preprocessing (handling missing values, encoding categorical variables, normalizing numerical features), exploratory data analysis, 
model selection, model evaluation, cross-validation and hyperparameter tuning, and analysis of results.
The goal is not just to develop an effective model but also to understand the variables influencing income levels and provide insights for policy-making.


## Requirements

To run the Python code in this project, the following libraries are required:
- Python 
- Scikit-learn 
- PyTorch 

The specific versions of these libraries are not mentioned as the project was implemented in Google Colab, which provides the necessary libraries and their compatible versions. The code should work without any issues on Google Colab with the default library versions provided.


## Running the Code in Google Colab
To run the code, follow these instructions:
1. Open the "Final_Code.ipynb" in Google Colab.
2. Click on the "Open in Colab" button to import the notebook into your Google Colab environment.
3. Upload the "adult.csv" dataset to your Google Colab environment.
4. Run the code cells in the notebook sequentially to execute the code step by step.

The project is divided into different sections, such as "Data Pre-Processing," "Supervised learning Classification with Decision Trees," "Semi-supervised learning Classification with Decision Trees," and "Supervised learning Classification with a deep learning model." You can access each section individually by opening the corresponding notebook file.

## Training and Validation

To train and validate the models, we have follow these steps that you could find them in "Final_Code.ipynb" :
1. Preprocess the dataset by handling missing values, categorical data encoding, and addressing class imbalance; related section in the file is "Data Pre-Processing".
2. Run the decision tree model by executing the appropriate Python script section ("Supervised learning Classification with Decision Trees"). Adjust the hyperparameters such as max_depth, min_samples_leaf, and min_samples_split for better performance.
3. Execute the semi-supervised learning algorithm by running the corresponding Python script section("Semi-supervised learning Classification with Decision Trees"). Fine-tune the hyperparameters, including confidence_threshold, to optimize the algorithm's performance.
4. Train the deep neural network model using the PyTorch framework. Run the Python script section ("Supervised learning Classification with a deep learning model"), ensuring that the dataset is properly preprocessed and the model architecture is defined.
5. Evaluate the trained models using appropriate metrics such as accuracy, precision, recall, and F1-score.

Note that we are not explaining how to run the pre-trained model on the provided sample test dataset, as we have not used the pre-trained model.

   

## Obtaining the Dataset

The dataset used in this project is the "Adult Census Income" dataset obtained from the UCI Machine Learning Repository. It contains information about individuals, including various demographic and socio-economic features. The dataset comprises 48,842 observations and includes features such as age, education, marital status, occupation, and more. The income variable is the target, indicating whether an individual's income exceeds a certain threshold. here is the link to the Kaggle where you can find this dataset and download it. https://www.kaggle.com/datasets/uciml/adult-census-income?resource=download; however you could find the sample test dataset which is the csv file on our github. 

To obtain the dataset, download it from the available download link provided by the UCI Machine Learning Repository. Preprocess the dataset as described in the "Final_Code.ipynb" at pre-processing section before using it for training and evaluation.



