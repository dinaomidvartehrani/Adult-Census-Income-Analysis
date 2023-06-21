# COMP 6721 - Project : Adult Census Income Analysis
## Group 06 

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Description](#description)
- [Data](#data)
  - [Data Preprocessing](#data-preprocessing)
  - [Features](#features)
- [Models](#models)
  - [Decision Tree Model](#decision-tree-model)
  - [Semi-Supervised Learning Model](#semi-supervised-learning-model)
  - [Deep Neural Network Model](#deep-neural-network-model)
- [Requirements](#requirements)
- [Running the Code in Google Colab](#running-the-code-in-google-colab)
- [Training and Validation](#training-and-validation)
- [Obtaining the Dataset](#obtaining-the-dataset)
- [Files and Reports](#files-and-reports)
- [Contact](#contact)



## Description

Welcome to the repository for our group project in COMP 6721. The objective of this project is to analyze and classify income levels based on a dataset of socio-economic factors. We have employed various machine learning models, including a decision tree model, a semi-supervised learning model, and a deep neural network model. This analysis can provide valuable insights into income classification and contribute to related fields such as finance, social sciences, and market research.

## Data

The dataset used in this project is the "Adult Census Income" dataset obtained from Kaggle, also we provide it in this repository "adult.csv". More on obtaining the dataset, [click here](#obtaining-the-dataset).


### Data Preprocessing

Before training the models, we performed data preprocessing to ensure the dataset's quality and suitability for analysis. The preprocessing steps included handling missing values, encoding categorical variables, and addressing class imbalance. By carefully addressing these issues, we aimed to enhance the accuracy and reliability of the models' predictions.

### Features

The dataset used in this project contains 15 features related to individuals' socio-economic factors. These features include age, education, marital status, occupation, and more. Each feature provides valuable information for income classification and contributes to the models' decision-making process.

## Models

In this project, we have implemented three different models to classify income levels based on the provided dataset. Each model offers unique characteristics and approaches to income classification.

### Decision Tree Model

The decision tree model utilizes a hierarchical structure to make decisions based on specific features. It recursively splits the data based on feature thresholds, resulting in a tree-like structure that provides interpretable rules for income classification. The decision tree model offers a balance between accuracy and interpretability, making it a valuable tool for understanding the factors influencing income levels.

### Semi-Supervised Learning Model

The semi-supervised learning model takes advantage of both labeled and unlabeled data to improve classification accuracy. Initially, the model is trained using the available labeled data. It then leverages the trained model to predict labels for the unlabeled data and assigns pseudo-labels based on these predictions. By iteratively updating the labeled dataset with high-confidence predictions, the model enhances its understanding of the data and improves classification performance.

### Deep Neural Network Model

The deep neural network (DNN) model utilizes multiple interconnected layers with nonlinear activation functions to extract complex patterns and representations from the data. The DNN model leverages its ability to learn hierarchical features and non-linear transformations to achieve high accuracy in income classification. Although the DNN model may have higher computational complexity, it provides good performance in capturing intricate relationships within the dataset.



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

The dataset used in this project is the "Adult Census Income" dataset obtained from Kaggle. It contains information about individuals, including various demographic and socio-economic features. The dataset comprises 48,842 observations and includes features such as age, education, marital status, occupation, and more. The income variable is the target, indicating whether an individual's income exceeds a certain threshold. here is the link to the Kaggle where you can find this dataset and download it. https://www.kaggle.com/datasets/uciml/adult-census-income?resource=download; however you could find the sample test dataset which is the csv file on our github. 

To obtain the dataset, download it from the available download link provided by Kaggle. Preprocess the dataset as described in the "Final_Code.ipynb" at pre-processing section before using it for training and evaluation.

## Files and Reports


## Contact

For any inquiries or questions regarding the project, please feel free to contact:

- Dina Omidvar: [dinaomidvar1377@gmail.com]
- Niloofar Tavakolian: [niltavakolian@gmail.com]
- Nastaran Naseri: [naseri.nastaran@hotmail.com]
- Seyedeh Mojdeh Haghighat Hosseini: [mozhde.2h@gmail.com]
