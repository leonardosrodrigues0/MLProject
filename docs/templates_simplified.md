# Activity 1: Evaluate data requirements.

## Template 1: Work brief (250-300 words total)

Introduction:

The organisation is experiencing ongoing cyber security issues involving authentication (verifying identity), authorisation (checking permissions), and IoT-related exposure. To strengthen monitoring of network activity, the Artificial Intelligence department has been tasked with developing an automated Intrusion Detection System (IDS), which detects suspicious or malicious behaviour. This project uses machine learning (ML), meaning the system learns patterns from examples rather than relying only on fixed rules. Work must follow organisational policies for responsible IT use, lawful data handling, and accurate documentation.

Objectives:

The main objective is to train and evaluate an ML-based IDS that classifies traffic as either normal or attack. The model must work on previously unseen records, show measurable performance, and provide evidence of:

- Data preparation (cleaning, encoding, scaling)
- Feature selection (keeping relevant inputs and removing non-relevant ones)
- Validation and testing

Technologies to be used and their basic concepts:

- Python: programming language used to build the solution
- Pandas: loads and prepares tabular data
- NumPy: numerical operations
- Scikit-learn: preprocessing, model training, and evaluation

Steps involved in designing IDS model:

1. Review the case study and organisational requirements.
2. Inspect the dataset and confirm the target column.
3. Convert categorical fields (text categories) into numeric form using encoding.
4. Scale numeric predictors using min-max normalisation (maps values to a 0 to 1 range).
5. Convert detailed labels into binary classes: normal vs attack.
6. Use the full dataset as input.
7. Split data into training, validation, and test subsets.
8. Remove predictors judged not relevant for training (for example constant fields).
9. Train the chosen ML model.
10. Evaluate accuracy, precision, recall, and F-score.
11. Document the procedure, parameter choices, and results.

## Template 2: Data Requirements (250-300 words total)

ML requirements:

The work brief requires an IDS built using a recognised workflow such as CRISP-DM. In simple terms, it means: define the goal, understand the data, prepare features, train a model, evaluate results, and document how it would be used. The project requires knowledge of Python, supervised learning (learning from labelled examples), binary classification, and preprocessing steps such as encoding and scaling.

Characteristics of NSL-KDD dataset:

The dataset contains 22,544 records and 42 columns. There are 41 predictors and one target column named `labels`. Three predictors are categorical (`protocol_type`, `service`, and `flag`), and the rest are numerical. For binary IDS operation, the target is mapped to:

- Normal: `labels == normal`
- Attack: all other labels

Specifications of NSL-KDD dataset:

The dataset contains connection indicators such as byte counts, login behaviour, error rates, host statistics, and service patterns. These attributes are appropriate for IDS modelling because they represent network connection behaviour.

Procedure for data transformation:

1. Remove duplicates if required.
2. Encode categorical predictors into numeric columns (one-hot encoding turns each category into a 0 or 1 indicator).
3. Convert `labels` into binary classes (normal vs attack).
4. Scale numerical predictors using min-max normalisation or standardisation (scales values to a comparable range).
5. Split the full dataset into training, validation, and testing subsets.
6. Drop columns judged not relevant for training and document the final retained feature set.

List of default and non-default training parameters:

For a supervised baseline, `MLPClassifier` can be used. This is a neural network model. Non-default parameters used in this project include:

- `hidden_layer_sizes=(64, 32)`
- `max_iter=300`
- `alpha=0.00005` (regularisation strength)
- `random_state=42`

# Activity 2: Arrange machine training datasets.

## Template 3: Training of IDS dataset (in 450-500 words)

Training of IDS dataset:

The IDS model is trained using the NSL-KDD dataset. The dataset includes categorical and numerical predictors that describe network connection behaviour. For an IDS, the target labels are simplified into two classes: normal traffic and attack traffic. This creates a binary classification task, where the model learns to separate normal from malicious behaviour. The full dataset is used as the input source. Data preparation and feature decisions are applied consistently so the training process is repeatable and auditable.

Set training data parameters:

The dataset is split into three subsets using stratification, which means each split keeps a similar normal vs attack proportion:

- Training set: used to fit the preprocessing steps and train the model
- Validation set: used to compare settings and detect overfitting (when a model memorises training patterns and performs worse on new data)
- Test set: used for final evaluation only and not used to tune settings

The baseline model is a multi-layer perceptron (MLP), which is a feed-forward neural network. The refined training settings used are:

- Hidden layers: 64 and 32 neurons
- Maximum iterations: 300 (number of training epochs)
- Regularisation: `alpha=0.00005` to reduce overfitting
- Fixed random seed for reproducibility

IDS dataset model size:

Model size depends on the number of input features after preprocessing and the chosen network topology. Categorical predictors are one-hot encoded, which increases the number of input columns because each category becomes its own indicator feature. With the current preprocessing pipeline (one-hot encoding for categoricals and scaling for numerics), the final input vector contains 116 features. The network topology is:

- Input layer: 116 features
- Hidden layer 1: 64 units
- Hidden layer 2: 32 units
- Output layer: 1 unit (binary decision)

This corresponds to approximately 9,601 trainable parameters (weights and biases). A parameter is a value the model learns during training.

Algorithm for feature selection and preprocessing (column filtering + normalisation):

1. Load the dataset and confirm the target column exists.
2. Optionally remove duplicate records.
3. Map the target into binary classes (normal vs attack).
4. Drop constant columns and any other columns judged not relevant or likely to cause leakage.
5. Split the data into training, validation, and test sets using stratification.
6. Fit preprocessing on the training set only:
   - One-hot encode categorical columns.
   - Apply min-max normalisation to numerical columns.
7. Apply the same fitted preprocessing to validation and test data, so the feature columns stay consistent.
8. Train the MLP model on the training set.
9. Evaluate performance on validation and test using accuracy, precision, recall, and F-score (F-score balances precision and recall).

Final IDS procedure through Block Diagram:

# Activity 3: Arrange validation datasets.

## Template 4: Supervisor feedback on recommendations (in 150-200 words)

Functionality issues of ML parameters:

The initial neural network settings (low iteration count) produced a convergence warning, meaning the optimiser stopped before fully settling. This can make results less stable and harder to justify, especially if small changes in random seed or split lead to noticeably different performance. Another issue was the risk of duplicated examples after mapping many attack labels into a single attack class. If duplicates remain, they can inflate evaluation results because near-identical examples may appear across splits.

Refined ML parameters:

The parameters were refined to improve training stability and reduce overfitting. The refined settings are:

- `hidden_layer_sizes=(64, 32)` for a compact but expressive model
- `max_iter=300` to allow more optimisation steps
- `alpha=0.00005` to add mild regularisation
- A fixed random seed for reproducibility

Data handling was also refined by removing duplicates after the binary label mapping step so that identical feature rows do not appear multiple times with the same binary target.

## Template 5: Code and screenshots

- Software code and screenshots are not included in this version.

# Activity 4: Arrange test datasets.

## Template 6: Test code and functionality issue check

Test code and functionality issue check:

Testing checks whether the refined preprocessing and model settings generalise to previously unseen data. The test set is held out from training and is used only for final evaluation. The same preprocessing steps must be used for all splits, and the preprocessing must be fitted only on the training data to avoid leaking information from validation or test into training.

Software test code:

The test process uses a single training and evaluation script that:

- Splits the full dataset into train, validation, and test
- Fits preprocessing on the training set
- Trains the model on the training set
- Reports metrics on validation and test

Functionality issues identified:

1. Duplicate rows after binary label mapping can inflate metrics and reduce the credibility of test results.
2. Rare categories in categorical columns can appear in validation or test but not in training, which can cause mismatched feature vectors if encoding is not handled correctly.

Issues resolved:

1. Remove duplicates after mapping labels into binary classes, so the same example is not repeated.
2. Use a single preprocessing pipeline with one-hot encoding that ignores unknown categories, so validation and test can be transformed safely even if they contain rare values.

# Activity 5: Finalise ML evaluations.

## Template 7: Performance results

One value for each metric:

- Classification testing accuracy: 0.9840
- Training accuracy: 0.9928
- Precision: 0.9817
- Recall: 0.9864
- F-score: 0.9841
