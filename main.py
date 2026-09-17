# ============================================================
# AUTOMATED ML EXPERIMENT
# 5 Algorithms × 3 Test Sizes × 5 PCA Components = 75 JPGs
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# ============================================================
# ALGORITHMS
# ============================================================

from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. SETTINGS
# ============================================================

# PCA values
PCA_COMPONENTS = [2, 4, 6, 8, 10]

# Test sizes
TEST_SIZES = [0.2, 0.4, 0.6]

# Maximum number of rows used for experiments
# This makes the 75 experiments much faster.
MAX_EXPERIMENT_ROWS = 20000

# Random state
RANDOM_STATE = 0

# Output folder
OUTPUT_FOLDER = "75_ML_Results"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# 2. LOAD ALL THREE DATASETS
# ============================================================

print("=" * 70)
print("LOADING DATASETS")
print("=" * 70)

DATASET_FOLDER = "datasets"

df1 = pd.read_csv(
    os.path.join(
        DATASET_FOLDER,
        "Tuesday-WorkingHours.pcap_ISCX.csv"
    ),
    low_memory=True
)

df2 = pd.read_csv(
    os.path.join(
        DATASET_FOLDER,
        "Wednesday-workingHours.pcap_ISCX.csv"
    ),
    low_memory=True
)

df3 = pd.read_csv(
    os.path.join(
        DATASET_FOLDER,
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
    ),
    low_memory=True
)



# ============================================================
# 3. COMBINE DATASETS
# ============================================================

dataset = pd.concat(
    [
        df1,
        df2,
        df3
    ],
    ignore_index=True
)

print(
    "\nCombined dataset shape:"
)

print(
    dataset.shape
)


# ============================================================
# 4. CLEAN COLUMN NAMES
# ============================================================

dataset.columns = (
    dataset.columns
    .str.strip()
)


# ============================================================
# 5. SEPARATE X AND Y
# ============================================================

X = dataset.iloc[:, :-1]

y = dataset.iloc[:, -1]


# ============================================================
# 6. CONVERT X TO NUMERIC
# ============================================================

print(
    "\nConverting features to numeric..."
)

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)


# ============================================================
# 7. REPLACE INF WITH NaN
# ============================================================

X.replace(
    [
        np.inf,
        -np.inf
    ],
    np.nan,
    inplace=True
)


# ============================================================
# 8. IMPUTE MISSING VALUES
# ============================================================

print(
    "Imputing missing values..."
)

imputer = SimpleImputer(
    missing_values=np.nan,
    strategy="mean"
)

X = imputer.fit_transform(
    X
)


# ============================================================
# 9. ENCODE TARGET
# ============================================================

print(
    "Encoding target labels..."
)

labelencoder_y = LabelEncoder()

y = labelencoder_y.fit_transform(
    y
)

class_names = labelencoder_y.classes_


print(
    "\nClasses:"
)

for i, name in enumerate(class_names):

    print(
        i,
        "=",
        name
    )


# ============================================================
# 10. MAKE SURE DATA IS NUMERIC
# ============================================================

X = np.asarray(
    X,
    dtype=np.float64
)

y = np.asarray(
    y
)


# ============================================================
# 11. STRATIFIED SAMPLING
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "CREATING EXPERIMENT SAMPLE"
)

print(
    "=" * 70
)

print(
    "Original samples:",
    X.shape[0]
)

print(
    "Maximum experiment samples:",
    MAX_EXPERIMENT_ROWS
)


if X.shape[0] > MAX_EXPERIMENT_ROWS:

    sample_indices, _ = train_test_split(

        np.arange(
            X.shape[0]
        ),

        train_size=MAX_EXPERIMENT_ROWS,

        random_state=RANDOM_STATE,

        stratify=y

    )

    X = X[
        sample_indices
    ]

    y = y[
        sample_indices
    ]

    print(
        "\nStratified sampling completed."
    )

else:

    print(
        "\nSampling not required."
    )


print(
    "Samples used for experiments:",
    X.shape[0]
)


# ============================================================
# 12. INFORMATION
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "DATA INFORMATION"
)

print(
    "=" * 70
)

print(
    "Total samples :",
    X.shape[0]
)

print(
    "Total features:",
    X.shape[1]
)

print(
    "Total classes :",
    len(class_names)
)

print(
    "\nPCA components:",
    PCA_COMPONENTS
)

print(
    "Test sizes:",
    TEST_SIZES
)

print(
    "\nTotal experiments:",
    len(PCA_COMPONENTS)
    * len(TEST_SIZES)
    * 5
)


# ============================================================
# 13. EXPERIMENT COUNTER
# ============================================================

experiment_number = 0

results = []


# ============================================================
# 14. MAIN AUTOMATION LOOP
# ============================================================

for test_size in TEST_SIZES:

    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    print(
        "\n"
    )

    print(
        "=" * 70
    )

    print(
        "TEST SIZE =",
        test_size
    )

    print(
        "=" * 70
    )


    X_train_original, X_test_original, y_train, y_test = (
        train_test_split(

            X,

            y,

            test_size=test_size,

            random_state=RANDOM_STATE,

            stratify=y

        )
    )


    # ========================================================
    # PCA LOOP
    # ========================================================

    for n_components in PCA_COMPONENTS:

        print(
            "\n"
        )

        print(
            "-" * 70
        )

        print(
            "PCA COMPONENTS =",
            n_components
        )

        print(
            "-" * 70
        )


        # ====================================================
        # PCA
        # ====================================================

        pca = PCA(
            n_components=n_components
        )


        # Fit PCA only on training data
        X_train = pca.fit_transform(
            X_train_original
        )


        # Transform test data
        X_test = pca.transform(
            X_test_original
        )


        # ====================================================
        # 5 ALGORITHMS
        # ====================================================

        algorithms = [

            # ------------------------------------------------
            # 11. CATBOOST
            # ------------------------------------------------
            (
                "CatBoost",

                CatBoostClassifier(

                    iterations=100,

                    depth=6,

                    learning_rate=0.1,

                    verbose=0,

                    random_state=RANDOM_STATE

                )
            ),


            # ------------------------------------------------
            # 12. ADABOOST
            # ------------------------------------------------
            (
                "AdaBoost",

                AdaBoostClassifier(

                    random_state=RANDOM_STATE

                )
            ),


            # ------------------------------------------------
            # 13. K-NEAREST NEIGHBORS
            # ------------------------------------------------
            (
                "K-Nearest Neighbors",

                KNeighborsClassifier(

                    n_neighbors=5,

                    n_jobs=-1

                )
            ),


            # ------------------------------------------------
            # 14. NAIVE BAYES
            # ------------------------------------------------
            (
                "Naive Bayes",

                GaussianNB()

            ),


            # ------------------------------------------------
            # 15. SUPPORT VECTOR MACHINE
            # ------------------------------------------------
            (
                "Support Vector Machine",

                LinearSVC(

                    C=1.0,

                    max_iter=5000,

                    random_state=RANDOM_STATE

                )
            )

        ]


        # ====================================================
        # ALGORITHM LOOP
        # ====================================================

        for algorithm_name, model in algorithms:

            experiment_number += 1


            print(
                "\n"
            )

            print(
                "=" * 70
            )

            print(
                f"EXPERIMENT {experiment_number}/75"
            )

            print(
                "Algorithm :",
                algorithm_name
            )

            print(
                "Test size :",
                test_size
            )

            print(
                "PCA       :",
                n_components
            )

            print(
                "=" * 70
            )


            # =================================================
            # TRAIN MODEL
            # =================================================

            print(
                "\nTraining model..."
            )

            model.fit(
                X_train,
                y_train
            )


            print(
                "Training completed."
            )


            # =================================================
            # PREDICTION
            # =================================================

            print(
                "Making predictions..."
            )

            y_pred = model.predict(
                X_test
            )

            y_pred = np.asarray(
                y_pred
            ).astype(int)


            # =================================================
            # CONFUSION MATRIX
            # =================================================

            cm = confusion_matrix(

                y_test,

                y_pred,

                labels=np.arange(
                    len(class_names)
                )

            )


            # =================================================
            # ACCURACY
            # =================================================

            accuracy = accuracy_score(

                y_test,

                y_pred

            )


            # =================================================
            # PRECISION
            # =================================================

            precision = precision_score(

                y_test,

                y_pred,

                average="weighted",

                zero_division=0

            )


            # =================================================
            # RECALL
            # =================================================

            recall = recall_score(

                y_test,

                y_pred,

                average="weighted",

                zero_division=0

            )


            # =================================================
            # F1 SCORE
            # =================================================

            f1 = f1_score(

                y_test,

                y_pred,

                average="weighted",

                zero_division=0

            )


            # =================================================
            # CLASSIFICATION REPORT
            # =================================================

            report = classification_report(

                y_test,

                y_pred,

                labels=np.arange(
                    len(class_names)
                ),

                target_names=class_names,

                zero_division=0

            )


            # =================================================
            # PRINT RESULTS
            # =================================================

            print(
                "\nConfusion Matrix:\n"
            )

            print(
                cm
            )


            print(
                "\nAccuracy : {:.4f}".format(
                    accuracy
                )
            )

            print(
                "Precision: {:.4f}".format(
                    precision
                )
            )

            print(
                "Recall   : {:.4f}".format(
                    recall
                )
            )

            print(
                "F1 Score : {:.4f}".format(
                    f1
                )
            )


            print(
                "\nClassification Report:\n"
            )

            print(
                report
            )


            # =================================================
            # SAVE RESULTS TO LIST
            # =================================================

            results.append({

                "Experiment":
                    experiment_number,

                "Algorithm":
                    algorithm_name,

                "Test Size":
                    test_size,

                "PCA Components":
                    n_components,

                "Accuracy":
                    accuracy,

                "Precision":
                    precision,

                "Recall":
                    recall,

                "F1 Score":
                    f1

            })


            # =================================================
            # CREATE JPG
            # =================================================

            safe_algorithm_name = (

                algorithm_name

                .replace(
                    " ",
                    "_"
                )

                .replace(
                    "-",
                    ""
                )

            )


            filename = (

                f"{experiment_number:02d}_"

                f"{safe_algorithm_name}_"

                f"Test_{int(test_size * 100)}_"

                f"PCA_{n_components}.jpg"

            )


            filepath = os.path.join(

                OUTPUT_FOLDER,

                filename

            )


            # =================================================
            # CREATE FIGURE
            # =================================================

            fig = plt.figure(

                figsize=(18, 13)

            )


            # =================================================
            # CONFUSION MATRIX
            # =================================================

            ax1 = plt.subplot2grid(

                (2, 2),

                (0, 0)

            )


            sns.heatmap(

                cm,

                annot=True,

                fmt="d",

                cmap="Blues",

                xticklabels=class_names,

                yticklabels=class_names,

                ax=ax1

            )


            ax1.set_title(

                "Confusion Matrix",

                fontsize=15,

                fontweight="bold"

            )


            ax1.set_xlabel(

                "Predicted Label"

            )


            ax1.set_ylabel(

                "Actual Label"

            )


            # =================================================
            # METRICS BOX
            # =================================================

            ax2 = plt.subplot2grid(

                (2, 2),

                (0, 1)

            )


            ax2.axis(
                "off"
            )


            metrics_text = f"""

ALGORITHM
{algorithm_name}

MODEL TYPE
Classification

TEST SIZE
{test_size}

PCA COMPONENTS
{n_components}

--------------------------------------

ACCURACY
{accuracy:.4f}

PRECISION
{precision:.4f}

RECALL
{recall:.4f}

F1 SCORE
{f1:.4f}

"""


            ax2.text(

                0.05,

                0.95,

                metrics_text,

                fontsize=13,

                verticalalignment="top",

                family="monospace"

            )


            # =================================================
            # CLASSIFICATION REPORT
            # =================================================

            ax3 = plt.subplot2grid(

                (2, 2),

                (1, 0),

                colspan=2

            )


            ax3.axis(
                "off"
            )


            ax3.text(

                0.01,

                0.98,

                "CLASSIFICATION REPORT\n\n"

                + report,

                fontsize=9,

                verticalalignment="top",

                family="monospace"

            )


            # =================================================
            # MAIN TITLE
            # =================================================

            fig.suptitle(

                f"{algorithm_name}   |   "

                f"Test Size = {test_size}   |   "

                f"PCA Components = {n_components}",

                fontsize=17,

                fontweight="bold"

            )


            # =================================================
            # SAVE JPG
            # =================================================

            plt.tight_layout(

                rect=[0, 0, 1, 0.95]

            )


            plt.savefig(

                filepath,

                dpi=150,

                format="jpg",

                bbox_inches="tight"

            )


            plt.close()


            print(
                "\nJPG saved:"
            )

            print(
                filepath
            )


# ============================================================
# 15. SAVE ALL 75 RESULTS TO CSV
# ============================================================

results_df = pd.DataFrame(
    results
)


csv_path = os.path.join(

    OUTPUT_FOLDER,

    "ALL_75_RESULTS.csv"

)


results_df.to_csv(

    csv_path,

    index=False

)


# ============================================================
# 16. FINAL MESSAGE
# ============================================================

print(
    "\n\n"
)

print(
    "=" * 70
)

print(
    "              ALL EXPERIMENTS COMPLETED"
)

print(
    "=" * 70
)


print(
    "\nTotal experiments:",
    len(results)
)


print(
    "\nExpected JPG files: 75"
)


actual_jpg_files = len(

    [

        f

        for f in os.listdir(
            OUTPUT_FOLDER
        )

        if f.lower().endswith(
            ".jpg"
        )

    ]

)


print(
    "Actual JPG files:",
    actual_jpg_files
)


print(
    "\nResults folder:"
)


print(
    os.path.abspath(
        OUTPUT_FOLDER
    )
)


print(
    "\nCSV file:"
)


print(
    os.path.abspath(
        csv_path
    )
)


print(
    "\nRun Successfully!."
)