# Activity 1: Evaluate data requirements.

## Template 1: Work brief (250-300 words total)

Introduction:

The organisation is experiencing ongoing cyber security issues involving authentication, authorisation, identification, and IoT-related exposure. To strengthen monitoring of network activity, the Artificial Intelligence department has been tasked with developing an automated Intrusion Detection System (IDS) that uses machine learning to distinguish legitimate traffic from malicious behaviour. The project will use the NSL-KDD dataset as the input source and must follow organisational procedures for responsible use of information technology, lawful handling of data, and accurate submission of assessment documentation.

Objectives:

The main objective is to train and evaluate an ML-based IDS that can classify normal and attack traffic from previously unseen records. The model must support binary intrusion detection, achieve measurable predictive performance, and produce evidence of preprocessing, feature engineering, validation, testing, and evaluation. A further objective is to document all work in the required templates so the process is transparent, auditable, and aligned with organisational expectations.

Technologies to be used and their basic concepts:

Python will be used as the implementation language because it supports efficient data analysis and ML experimentation.
Pandas will manage tabular data preparation, cleaning, and transformation.
Scikit-learn will support supervised learning, preprocessing, train-test splitting, feature selection via column filtering, and performance metrics.
NumPy will handle numerical operations, while Matplotlib can be used to visualise outputs and support reporting.
Feature engineering will focus on data preparation: removing irrelevant or leakage-prone columns, encoding categorical variables, scaling numerical predictors, and applying additional cleaning techniques (for example, duplicate handling and basic outlier checks) to produce stable model inputs.

Steps involved in designing IDS model:

Review the case study and dataset requirements;
Inspect and understand dataset attributes;
Convert categorical fields to numeric form;
Apply min-max normalisation;
Map detailed attack labels into binary normal or intrusion classes;
Use the full dataset as input;
Split data into training, validation, and test subsets;
Remove non-relevant or redundant predictors (feature filtering);
Train the chosen ML model;
Evaluate accuracy, precision, recall, and F-score;
Document the full procedure and results.

## Template 2: Data Requirements (250-300 words total)

ML requirements:

The work brief requires a machine learning IDS designed with a recognised workflow such as CRISP-DM. This means defining the business goal, understanding the data, preparing features, building a model, evaluating performance, and preparing the solution for documented deployment. The project requires prior knowledge of Python programming, core ML concepts, parameter tuning, binary classification, and matrix-based operations used in encoding and scaling.

Characteristics of NSL-KDD dataset:

The current `data/nsl_kdd_dataset.csv` contains 22,544 records and 42 columns. There are 41 predictors and one target column named `labels`. Three predictors are categorical (`protocol_type`, `service`, and `flag`) and the remaining predictors are numerical. The target includes `normal` plus multiple attack categories such as `neptune`, `satan`, `smurf`, and `portsweep`. The file contains no missing values and 31 duplicate rows. Because the organisation requires binary intrusion detection, the target must be transformed from multi-class labels into two classes: normal and attack.

Specifications of NSL-KDD dataset:

The dataset is large enough to support training on the full available records and splitting into separate training, validation, and test subsets. It includes connection-based traffic features such as byte counts, login behaviour, error rates, host statistics, and service patterns, which are appropriate for IDS modelling.

Procedure for data transformation:

Remove duplicate records if required; convert categorical fields using encoding; convert `labels` into binary classes; apply min-max normalisation (or standardisation) to numerical predictors; split the full dataset into training, validation, and testing subsets; then drop columns judged not relevant for training (for example identifiers, constant/near-constant fields, or leakage-prone attributes) and document the final retained feature set.

List of default and non-default training parameters:

For a supervised baseline, `MLPClassifier` can be used. Default parameters include `activation='relu'`, the default solver (adam), `alpha=0.0001`, and `learning_rate_init=0.001`. Non-default parameters proposed for this project are `hidden_layer_sizes=(64, 32)`, `max_iter=100`, `random_state=42`, and `early_stopping=True`.

# Activity 2: Arrange machine training datasets.

## Template 3: Training of IDS dataset (in 450-500 words)

Training of IDS dataset:

The NSL-KDD dataset (`data/nsl_kdd_dataset.csv`) is used as the sole input source for model training. The dataset contains 22,544 network connection records with 41 predictor columns and one target column (`labels`). The predictors include three categorical attributes (`protocol_type`, `service`, and `flag`) and a set of numerical attributes that capture connection duration, byte counts, login behaviour, error rates, and host/service statistics. For the IDS use-case, the target is converted to binary classes: `normal` is treated as legitimate traffic and every other label is treated as an intrusion (attack). To keep the process auditable and repeatable, the entire dataset is considered (no down-sampling step) and preprocessing decisions are applied consistently across all splits.

Set training data parameters:

The full dataset is split into training, validation, and test subsets using a stratified split to preserve the normal/attack class proportion across all subsets. The default split used in Activity 2 is 70% training, 15% validation, and 15% test, controlled by a fixed `random_state=42` for reproducibility. The baseline learning algorithm is a multi-layer perceptron classifier (`MLPClassifier`) with `max_iter=100` for this initial training run. The purpose of this activity is to establish a working training pipeline (data preparation plus a baseline model) that can later be refined in Activities 3 and 4 by adjusting parameters and re-evaluating performance.

IDS dataset model size:

Model size is determined by (1) the number of input features after encoding and scaling, and (2) the chosen network topology. Categorical predictors are one-hot encoded, which expands the raw feature space into multiple indicator columns. With the current dataset and preprocessing pipeline, the encoded feature vector contains 116 input features. The baseline topology used is `116 -> 64 -> 32 -> 1`, meaning two hidden layers with 64 and 32 neurons and a single output unit for binary classification. This corresponds to approximately 9,601 trainable parameters (weights and biases).

Algorithm for feature selection and preprocessing (column filtering + normalisation):

1. Load the CSV and confirm `labels` exists as the target column.
2. Optionally remove duplicate records.
3. Apply column filtering on the raw table before encoding: automatically drop constant columns (for example, `num_outbound_cmds` is constant in this dataset) and optionally drop any additional columns judged not relevant for training (such as leakage-prone or redundant predictors).
4. Convert the target into binary classes: `y = 1` for attacks (`labels != 'normal'`) and `y = 0` for normal traffic.
5. Split the full dataset into train/validation/test using stratification.
6. Build a preprocessing pipeline where categorical columns are one-hot encoded (`handle_unknown='ignore'`) and numeric columns are scaled using min-max normalisation.
7. Train the baseline `MLPClassifier` on the training split, use the validation split for tuning decisions in later activities, and reserve the test split for final evaluation.

Final IDS procedure through Block Diagram:

# Activity 3: Arrange validation datasets.

## Template 4: Supervisor feedback on recommendations (in 150-200 words)

Functionality issues of ML parameters:

During baseline training, the `MLPClassifier` raised a convergence warning when `max_iter=100`, indicating the optimiser had not reached a stable minimum. This is a functionality issue because it can lead to unstable performance across runs and makes it harder to justify results (the model may stop improving early). A smaller but expected issue was a modest generalisation gap between training and validation performance, which suggests the need to balance training time and regularisation. No class-imbalance issues were observed after converting `labels` into binary classes, as the dataset is approximately 50/50 normal vs attack after duplicate removal.

Refined ML parameters:

To address convergence and stabilise validation performance, the model parameters were refined by increasing training iterations and tuning regularisation. The refined configuration is `hidden_layer_sizes=(64, 32)`, `max_iter=300`, `alpha=0.00005`, and `random_state=42` to keep results reproducible. With the same preprocessing pipeline (one-hot encoding + min-max scaling and constant-column dropping), the refined settings remove the convergence warning in typical runs and slightly improve validation F1 compared to the baseline while maintaining similar test performance.

## Template 5: Code and screenshots

- Software Code
- Screenshots

# Activity 4: Arrange test datasets.

## Template 6: Test code and functionality issue check

Test code and functionality issue check:

Testing uses the same preprocessing pipeline and refined parameters from Activity 3, but evaluates performance only on the held-out test split created from the full dataset. The goal is to confirm that the selected preprocessing (column filtering, encoding, and min-max normalisation) and model settings generalise to previously unseen records, and that there is no obvious data leakage between splits. The test procedure includes sanity checks for contamination (row overlap between train/validation/test) and a dummy baseline (majority-class predictor) so that unexpectedly high results can be interpreted correctly.

Software test code:

The test run is executed using the shared training/evaluation script `scripts/train_ids.py` with refined settings:
`python3 scripts/train_ids.py --drop-duplicates --drop-constant-cols --drop-duplicates-after-labelmap --sanity-checks --max-iter 300 --alpha 0.00005`
This command prints dataset split sizes, feature counts after encoding/scaling, sanity-check statistics, and the Train/Val/Test accuracy, precision, recall, and F1 metrics.

Functionality issues identified:

1. Duplicate handling after label mapping: although exact duplicate rows can be removed from the raw CSV, converting multi-class attack labels into a single binary attack class can create new duplicates (identical features with the same binary target). This can cause exact row overlap between training and validation/test splits if not addressed.
2. Misleading “too-good” early results: even with very small `max_iter`, `MLPClassifier` still performs a full epoch of updates, and NSL-KDD binary normal vs attack can be separable. Without a baseline comparison and split-overlap check, early results may look suspiciously strong.

Issues resolved:

1. Enabled `--drop-duplicates-after-labelmap` to remove duplicates created by the binary label mapping step (dedupe on features + binary target). With sanity checks enabled, exact row overlap across splits is reduced to zero.
2. Added `--sanity-checks` output (row-overlap counts and a majority-class dummy baseline) to make evaluation results auditable and to help detect leakage/contamination during testing.

# Activity 5: Finalise ML evaluations.

## Template 7: Performance results

One value for each metric:

- Classification testing accuracy: 0.9840
- Training accuracy: 0.9928
- Precision: 0.9817
- Recall: 0.9864
- F-score: 0.9841
