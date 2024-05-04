# ADHD Classification


A repository containing code for applying various machine learning algorithms to classify ADHD using the NSCH dataset.

## Introduction

This repository contains code for applying machine learning algorithms to classify ADHD using the NSCH (National Survey of Children's Health) dataset from the NSCH organization. The goal is to predict ADHD based on a set of manually selected features and feature extraction algorithms.

## Dataset

The dataset used in this project is the NSCH dataset provided by the NSCH organization. It contains data related to children's health, including ADHD diagnosis and various features.
The selected features and the result of the feature extraction algorithms can be found in `pre_date` folder.
## Features Selection

Before applying the machine learning algorithms, a manual feature selection process is performed to identify the most relevant features. Additionally, feature extraction algorithms such as Information Gain, Chi-Square, Fisher Score, and Correlation Coefficient are applied to further enhance the feature set.

## Machine Learning Algorithms

The following machine learning algorithms are applied to classify ADHD:

- RandomForestClassifier
- GradientBoostingClassifier
- AdaBoostClassifier
- XGBClassifier
- DecisionTreeClassifier
- SVM (Support Vector Machines)
- LogisticRegression
- GaussianNB
- KNeighborsClassifier

In addition to the individual algorithms, combinations and variants of these algorithms are also explored to find the most effective approach for ADHD classification.

## Usage

The main code for this project is available in the `ADHD/ADHD/ADHD_CODE.ipynb` Jupyter Notebook. You can open it using Jupyter or Google Colab to run the code and experiment with different configurations.
