# Automated Machine Learning Experiment

## 5 Algorithms × 3 Test Sizes × 5 PCA Components = 75 Experiments

This project performs an automated machine learning experiment on network traffic data.

The objective is to compare **five classification algorithms** under different **train-test split sizes** and **PCA dimensionality-reduction settings**.

The complete experiment generates:

* **75 machine learning experiments**
* **75 JPG result reports**
* **1 CSV file containing all experiment metrics**
* Confusion matrices
* Accuracy
* Precision
* Recall
* F1-score
* Classification reports

---

## Project Overview

The project combines three network traffic datasets and uses the last column as the target variable.

The workflow is:

```text
Dataset
   ↓
Combine 3 CSV files
   ↓
Clean column names
   ↓
Convert features to numeric
   ↓
Replace infinite values
   ↓
Handle missing values
   ↓
Encode target labels
   ↓
Stratified sampling
   ↓
Train/Test Split
   ↓
PCA
   ↓
5 Machine Learning Algorithms
   ↓
Prediction
   ↓
Evaluation
   ↓
JPG Report + CSV Results
```

---

## Algorithms Used

The project compares the following five classification algorithms.

### 1. CatBoost

CatBoost is a gradient-boosting algorithm based on decision trees.

```python
CatBoostClassifier(
    iterations=100,
    depth=6,
    learning_rate=0.1,
    verbose=0,
    random_state=0
)
```

### 2. AdaBoost

AdaBoost is an ensemble learning algorithm that combines multiple weak learners to create a stronger classifier.

```python
AdaBoostClassifier(
    random_state=0
)
```

### 3. K-Nearest Neighbors

KNN classifies a sample based on the classes of its nearest neighboring samples.

```python
KNeighborsClassifier(
    n_neighbors=5,
    n_jobs=-1
)
```

### 4. Gaussian Naive Bayes

Gaussian Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem.

```python
GaussianNB()
```

### 5. Support Vector Machine

The project uses a linear Support Vector Machine through `LinearSVC`.

```python
LinearSVC(
    C=1.0,
    max_iter=5000,
    random_state=0
)
```

`LinearSVC` is used instead of the general `SVC` implementation because the original dataset contains a very large number of samples.

---

# Dataset

The project combines three CSV datasets:

```text
Tuesday-WorkingHours.pcap_ISCX.csv
Wednesday-workingHours.pcap_ISCX.csv
Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
```

The datasets contain network traffic features and a target class representing different types of network activity.

The combined dataset used during development contains approximately:

```text
1.3 million rows
78 features
11 target classes
```

### Target Classes

The target classes include network traffic categories such as:

```text
BENIGN
DoS GoldenEye
DoS Hulk
DoS Slowhttptest
DoS slowloris
FTP-Patator
Heartbleed
SSH-Patator
Web Attack – Brute Force
Web Attack – Sql Injection
Web Attack – XSS
```

> Dataset files are not included in this repository if they are too large for GitHub. Place the required CSV files in the project directory before running the program.

---

# Data Preprocessing

Several preprocessing steps are performed automatically.

## 1. Combine datasets

The three CSV files are combined using:

```python
pd.concat()
```

---

## 2. Clean column names

Whitespace is removed from column names:

```python
dataset.columns = dataset.columns.str.strip()
```

---

## 3. Separate features and target

The last column is used as the target:

```python
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
```

---

## 4. Convert features to numeric

Non-numeric values are converted to missing values:

```python
X = X.apply(
    pd.to_numeric,
    errors="coerce"
)
```

---

## 5. Handle infinite values

Positive and negative infinity values are replaced with `NaN`.

```python
X.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)
```

---

## 6. Handle missing values

Missing values are replaced using the mean of each feature.

```python
SimpleImputer(
    strategy="mean"
)
```

---

## 7. Encode target labels

The categorical target labels are converted into numerical values using:

```python
LabelEncoder()
```

For example:

```text
BENIGN → 0
DoS GoldenEye → 1
DoS Hulk → 2
...
```

---

# Stratified Sampling

The original combined dataset contains approximately 1.3 million samples.

Running 75 experiments directly on the entire dataset can require a very large amount of computational time, especially for algorithms such as KNN and SVM.

Therefore, the program uses a stratified sample:

```python
MAX_EXPERIMENT_ROWS = 20000
```

The sample is selected while preserving the approximate class distribution.

```python
train_test_split(
    np.arange(X.shape[0]),
    train_size=MAX_EXPERIMENT_ROWS,
    random_state=0,
    stratify=y
)
```

This means the experiment uses up to **20,000 samples** while maintaining class representation.

---

# PCA

Principal Component Analysis (PCA) is used for dimensionality reduction.

The experiment tests five different PCA configurations:

```python
PCA_COMPONENTS = [2, 4, 6, 8, 10]
```

Therefore, each algorithm is tested with:

```text
2 PCA components
4 PCA components
6 PCA components
8 PCA components
10 PCA components
```

PCA is fitted only on the training data and then applied to the test data.

```python
X_train = pca.fit_transform(X_train_original)

X_test = pca.transform(X_test_original)
```

This avoids fitting PCA using the test data.

---

# Test Sizes

Three different test sizes are evaluated:

```python
TEST_SIZES = [0.2, 0.4, 0.6]
```

This corresponds to:

| Test Size | Training Data | Testing Data |
| --------- | ------------: | -----------: |
| 20%       |           80% |          20% |
| 40%       |           60% |          40% |
| 60%       |           40% |          60% |

All splits use stratification:

```python
stratify=y
```

---

# Experiment Structure

There are:

```text
5 Algorithms
×
3 Test Sizes
×
5 PCA Configurations
=
75 Experiments
```

### Calculation

```text
5 × 3 × 5 = 75
```

The program automatically runs all combinations.

---

# Evaluation Metrics

Each experiment calculates four main metrics.

## Accuracy

Accuracy measures the proportion of correctly classified samples.

```text
Accuracy =
Correct Predictions / Total Predictions
```

---

## Precision

Precision measures how many samples predicted as a particular class were actually members of that class.

---

## Recall

Recall measures how many samples belonging to a class were correctly identified.

---

## F1 Score

F1-score combines precision and recall.

```text
F1 = 2 × (Precision × Recall)
     --------------------------
       Precision + Recall
```

The project uses **weighted averaging** for precision, recall, and F1-score.

---

# Confusion Matrix

A confusion matrix is generated for every experiment.

It shows:

```text
Actual Class
     ↓
Predicted Class
```

The confusion matrix helps identify which network traffic classes are being correctly or incorrectly classified.

---

# Output Files

All generated results are stored in:

```text
75_ML_Results/
```

Each experiment produces one JPG report.

Example:

```text
01_CatBoost_Test_20_PCA_2.jpg
02_AdaBoost_Test_20_PCA_2.jpg
03_KNearest_Neighbors_Test_20_PCA_2.jpg
04_Naive_Bayes_Test_20_PCA_2.jpg
05_Support_Vector_Machine_Test_20_PCA_2.jpg
```

The exact filenames are generated automatically by the program.

Each JPG contains:

1. Confusion matrix
2. Algorithm name
3. Test size
4. PCA components
5. Accuracy
6. Precision
7. Recall
8. F1-score
9. Classification report

---

# CSV Results

The program also generates:

```text
ALL_75_RESULTS.csv
```

The CSV contains the results of all 75 experiments.

Example columns:

```text
Experiment
Algorithm
Test Size
PCA Components
Accuracy
Precision
Recall
F1 Score
```

This CSV can be used for further analysis in:

* Excel
* Pandas
* Power BI
* Tableau
* Python visualization

---

# Project Structure

Recommended GitHub repository structure:

```text
Automated-ML-Experiment/
│
├── main.py
│
├── README.md
│
├── requirements.txt
│
├── 75_ML_Results/
│   ├── 01_CatBoost_Test_20_PCA_2.jpg
│   ├── 02_AdaBoost_Test_20_PCA_2.jpg
│   ├── ...
│   ├── 75_*.jpg
│   └── ALL_75_RESULTS.csv
│
├── .gitignore
│
└── datasets/
    ├── Tuesday-WorkingHours.pcap_ISCX.csv
    ├── Wednesday-workingHours.pcap_ISCX.csv
    └── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
```

For large datasets, it is recommended to exclude the CSV files from GitHub using `.gitignore`.

---

# Installation

## 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Move into the project directory:

```bash
cd Automated-ML-Experiment
```

---

## 2. Create a virtual environment

You can use either Anaconda or Python's built-in virtual environment.

### Using Python

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

---

# Install Required Libraries

Install the required Python packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn catboost
```

Or install everything from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

# Running the Project

Make sure the three dataset files are located in the same directory as the Python program.

Then run:

```bash
python main.py
```

The program will automatically:

```text
Load datasets
      ↓
Combine datasets
      ↓
Clean data
      ↓
Sample 20,000 rows
      ↓
Create train/test splits
      ↓
Apply PCA
      ↓
Train 5 algorithms
      ↓
Calculate metrics
      ↓
Generate JPG
      ↓
Repeat 75 times
      ↓
Create CSV
```

---

# Expected Console Output

The program displays the progress of every experiment.

Example:

```text
======================================================================
EXPERIMENT 1/75
Algorithm : CatBoost
Test size : 0.2
PCA       : 2
======================================================================

Training model...
Training completed.
Making predictions...

Accuracy : 0.XXXX
Precision: 0.XXXX
Recall   : 0.XXXX
F1 Score : 0.XXXX

JPG saved:
75_ML_Results/01_CatBoost_Test_20_PCA_2.jpg
```

The experiment counter continues until:

```text
EXPERIMENT 75/75
```

---

# Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* CatBoost

---

# Machine Learning Concepts Demonstrated

This project demonstrates several important machine learning concepts:

* Data preprocessing
* Missing-value imputation
* Label encoding
* Stratified sampling
* Train-test splitting
* Dimensionality reduction
* Principal Component Analysis
* Ensemble learning
* Instance-based learning
* Probabilistic classification
* Support Vector Machines
* Multi-class classification
* Confusion matrices
* Classification reports
* Accuracy
* Precision
* Recall
* F1-score
* Automated experimentation
* Result generation
* CSV-based experiment tracking

---

# Important Note About the Results

The experiment uses a **stratified sample of up to 20,000 rows** from the combined dataset to make the 75-model experiment computationally manageable.

Therefore, the results represent the performance of the models on this experimental sample and should not automatically be interpreted as results obtained from the entire original dataset.

The random state is fixed:

```python
RANDOM_STATE = 0
```

This makes the sampling and train-test splits reproducible.

---

# Future Improvements

Possible improvements include:

* Testing additional classification algorithms
* Hyperparameter tuning
* Cross-validation
* Feature selection
* Comparing different PCA configurations
* Visualization of algorithm performance
* ROC-AUC analysis
* Precision-recall curves
* Automated result ranking
* Interactive dashboards
* Experiment tracking with MLflow

---

# Author

**Rahul Kapar**

B.Tech Computer Science Engineering
Artificial Intelligence & Machine Learning

---

# License

This project is intended for educational and research purposes.
