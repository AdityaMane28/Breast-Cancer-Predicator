# Breast Cancer Classification with Logistic Regression

This project uses the Breast Cancer Wisconsin data in `BreastCancer.csv` to explore binary classification with logistic regression. The notebooks include exploratory analysis, preprocessing, model evaluation, and a Jupyter widget interface.

> **Educational use only.** This project is not a medical device and must not be used to diagnose, screen, or guide treatment. Its output is not a substitute for advice from qualified healthcare professionals.

## Project files

- [`Breast Cancer.ipynb`](Breast%20Cancer.ipynb) — exploratory data analysis, preprocessing, training, and evaluation.
- [`Breast Cancer Model.ipynb`](Breast%20Cancer%20Model.ipynb) — interactive Jupyter widget for looking up a record by ID or entering feature values.
- [`BreastCancer.csv`](BreastCancer.csv) — dataset used for training.
- [`breast_cancer_model.pkl`](breast_cancer_model.pkl) and [`scaler.pkl`](scaler.pkl) — serialized logistic regression model and scaler.

## Dataset

The CSV contains 699 records and 11 columns: an `Id`, nine cell-characteristic measurements, and the binary `Class` target. The model uses all columns except `Class`, including `Id`:

`Id`, `Cl.thickness`, `Cell.size`, `Cell.shape`, `Marg.adhesion`, `Epith.c.size`, `Bare.nuclei`, `Bl.cromatin`, `Normal.nucleoli`, and `Mitoses`.

The raw data has 16 missing values in `Bare.nuclei`. The training notebook fills them with the column mean and removes 8 duplicate rows, leaving 691 records. The widget labels class 0 as benign and class 1 as malignant; verify the label coding against the dataset before interpreting results.

## Modeling workflow

1. Explore class counts, distributions, feature relationships, outliers, and correlations.
2. Fill missing `Bare.nuclei` values with the mean and remove duplicate rows.
3. Split into training and test sets using an 80/20 stratified split with `random_state=42`.
4. Fit `StandardScaler` on the training data and scale both sets.
5. Train `LogisticRegression(max_iter=20, random_state=42)`.
6. In the evaluation notebook, classify probabilities at a 0.30 threshold and calculate ROC-AUC from the probabilities.

The notebook also creates SMOTE-resampled data, but the resampled data is not used to fit the model. The recorded scores use the original stratified split.

## Recorded results

The individual metric cells in the saved training notebook report:

| Metric | Score |
|---|---:|
| Accuracy | 0.9712 |
| Precision (class 1) | 0.9783 |
| Recall (class 1) | 0.9375 |
| F1-score (class 1) | 0.9574 |
| ROC-AUC | 0.9970 |

The saved confusion matrix has actual classes as rows and predicted classes as columns in `[0, 1]` order:

```text
[[90, 1],
 [ 3, 45]]
```

The saved `classification_report` output does not match these individual metrics and the confusion matrix. Rerun the notebook from the beginning before relying on the displayed results.

## Setup and run

Use Python 3 and install the required packages:

```bash
python -m pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn ipywidgets notebook
```

From the project directory, start Jupyter:

```bash
jupyter notebook
```

Open `Breast Cancer.ipynb` and run its cells in order. Keep `BreastCancer.csv` in the same directory as the notebook.

## Widget notebook notes

The widget fields now follow the same feature order used to train the model. Before running `Breast Cancer Model.ipynb`, make these adjustments for the files in this repository:

- Its CSV load statement currently looks for `breast_cancer.csv`; change it to `BreastCancer.csv` or rename the CSV to match.
- The widget reads the CSV directly and does not fill missing `Bare.nuclei` values. A record with a missing value can fail when the notebook converts that value to an integer; apply the same imputation used in the training notebook.
- The evaluation notebook uses a 0.30 probability threshold, while the widget calls `model.predict()` with scikit-learn's default 0.50 threshold. Predictions from the widget can therefore differ from the reported evaluation for borderline probabilities.