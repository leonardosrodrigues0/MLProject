# Activity 1: Evaluate data requirements.

## Template 1: Work brief (250-300 words total)

Introduction:

The organisation is experiencing ongoing cyber security issues involving authentication, authorisation, identification, and IoT-related exposure. To strengthen monitoring of network activity, the Artificial Intelligence department has been tasked with developing an automated Intrusion Detection System (IDS) that uses machine learning to distinguish legitimate traffic from malicious behaviour. The project will use the NSL-KDD dataset as the input source and must follow organisational procedures for responsible use of information technology, lawful handling of data, and accurate submission of assessment documentation.

Objectives:

The main objective is to train and evaluate an ML-based IDS that can classify normal and attack traffic from previously unseen records. The model must support binary intrusion detection, achieve measurable predictive performance, and produce evidence of preprocessing, feature engineering, validation, testing, and evaluation. A further objective is to document all work in the required templates so the process is transparent, auditable, and aligned with organisational expectations.

Technologies to be used and their basic concepts:

Python will be used as the implementation language because it supports efficient data analysis and ML experimentation.
Pandas will manage tabular data preparation, cleaning, and transformation.
Scikit-learn will support supervised learning, preprocessing, train-test splitting, PCA-based feature engineering, and performance metrics.
NumPy will handle numerical operations, while Matplotlib can be used to visualise outputs and support reporting.
PCA is the selected feature engineering technique because it reduces dimensionality by transforming correlated variables into a smaller set of principal components while preserving useful variance.

Steps involved in designing IDS model:

Review the case study and dataset requirements;
Inspect and understand dataset attributes;
Convert categorical fields to numeric form;
Apply min-max normalisation;
Map detailed attack labels into binary normal or intrusion classes;
Randomly select at least 5,000 samples;
Split data into training, validation, and test subsets;
Apply PCA;
Train the chosen ML model;
Evaluate accuracy, precision, recall, and F-score;
Document the full procedure and results.

## Template 2: Data Requirements (250-300 words total)

ML requirements:

The work brief requires a machine learning IDS designed with a recognised workflow such as CRISP-DM. This means defining the business goal, understanding the data, preparing features, building a model, evaluating performance, and preparing the solution for documented deployment. The project requires prior knowledge of Python programming, core ML concepts, parameter tuning, binary classification, and matrix-based operations used in scaling and PCA.

Characteristics of NSL-KDD dataset:

The current `data/nsl_kdd_dataset.csv` contains 22,544 records and 42 columns. There are 41 predictors and one target column named `labels`. Three predictors are categorical (`protocol_type`, `service`, and `flag`) and the remaining predictors are numerical. The target includes `normal` plus multiple attack categories such as `neptune`, `satan`, `smurf`, and `portsweep`. The file contains no missing values and 31 duplicate rows. Because the organisation requires binary intrusion detection, the target must be transformed from multi-class labels into two classes: normal and attack.

Specifications of NSL-KDD dataset:

The dataset is large enough to satisfy the requirement to randomly select at least 5,000 samples. It includes connection-based traffic features such as byte counts, login behaviour, error rates, host statistics, and service patterns, which are appropriate for IDS modelling.

Procedure for data transformation:

Remove duplicate records if required; randomly sample the required dataset size; convert categorical fields using encoding; convert `labels` into binary classes; apply min-max normalisation to numerical predictors; split the data into 70:30 training and testing proportions; then apply PCA to produce a reduced feature subset.

List of default and non-default training parameters:

For a supervised baseline, `MLPClassifier` can be used. Default parameters include `activation='relu'`, `solver='adam'`, `alpha=0.0001`, and `learning_rate='constant'`. Non-default parameters proposed for this project are `hidden_layer_sizes=(64, 32)`, `max_iter=100`, `random_state=42`, and `early_stopping=True`.

# Activity 2: Arrange machine training datasets.

## Template 3: Training of IDS dataset (in 450-500 words)

- Training of IDS dataset
- Set training data parameters
- IDS dataset model size
- Algorithm for applying feature engineering process using PCA
- Final IDS procedure through Block Diagram

# Activity 3: Arrange validation datasets.

## Template 4: Supervisor feedback on recommendations (in 150-200 words)

- Functionality issues of ML parameters
- Refined ML parameters

## Template 5: Code and screenshots

- Software Code
- Screenshots

# Activity 4: Arrange test datasets.

## Template 6: Test code and functionality issue check

- Test code and functionality issue check
- Software test code
- Functionality issues identified
- Issues resolved

# Activity 5: Finalise ML evaluations.

## Template 7: Performance results

One value for each metric:

- Classification testing accuracy
- Training accuracy
- Precision
- Recall
- F-score
