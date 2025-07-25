# Income Prediction with RandomForest and Hyperparameter Tuning

## Overview

This project aims to predict whether an individual's annual income exceeds $50,000 based on a set of demographic and other census data. The project involves a comprehensive Exploratory Data Analysis (EDA), data preprocessing, feature engineering, and the implementation of a RandomForest Classifier. To optimize the model's performance, feature selection based on importance was conducted, and hyperparameters were tuned using `RandomizedSearchCV`.

[Kaggle Link](https://www.kaggle.com/code/emirhanhasrc/eda-randomforest-classifier-randomsearch)

## Dataset

The dataset used is the "Adult" dataset from the UCI Machine Learning Repository, containing 14 features and one target variable (`income`).

### Data Dictionary

| # | Column | Non-Null Count | Dtype | Description |
|---|---|---|---|---|
| 0 | `age` | 32561 | int64 | Age of the individual |
| 1 | `workclass` | 32561 | object | Type of employer |
| 2 | `fnlwgt` | 32561 | int64 | Final weight, an estimate of the number of people the observation represents |
| 3 | `education` | 32561 | object | Highest level of education |
| 4 | `education-num` | 32561 | int64 | Numerical representation of education level |
| 5 | `marital-status` | 32561 | object | Marital status |
| 6 | `occupation` | 32561 | object | Occupation |
| 7 | `relationship` | 32561 | object | Relationship status |
| 8 | `race` | 32561 | object | Race |
| 9 | `sex` | 32561 | object | Gender |
| 10 | `capital-gain` | 32561 | int64 | Capital gains |
| 11 | `capital-loss` | 32561 | int64 | Capital losses |
| 12 | `hours-per-week`| 32561 | int64 | Number of hours worked per week |
| 13 | `native-country`| 32561 | object | Native country |
| 14 | `income` | 32561 | object | Target variable: <=50K or >50K |

## Data Preprocessing and Cleaning

1.  **Cleaning Object Columns:** The categorical (object) columns contained extra leading and trailing whitespaces (e.g., `" >50K"` instead of `">50K"`). These were stripped to ensure data consistency.
2.  **Handling Missing Values:** Some features had missing values represented by a `"?"`. These were replaced with `NaN` and then imputed using the mode (most frequent value) of the respective columns from the training data:
    *   `workclass`
    *   `occupation`
    *   `native-country`

## Exploratory Data Analysis (EDA)

EDA was performed to uncover insights and understand the relationships between different features and the target variable (`income`). Key visualizations included:
*   **Count Plots:** To visualize the income distribution across `sex` and `race`.
*   **Pie Charts:** To show the percentage distribution of `race`, `sex`, and the target `income` variable.
*   **Box Plot:** To examine the distribution of `age` for each `sex`.
*   **Correlation Heatmap:** To understand the linear relationships between numerical features.
*   **Pairplot:** To visualize pairwise relationships between features, colored by the `income` level.

## Feature Engineering

Categorical features were transformed into a numerical format suitable for the machine learning model.
1.  **Mean Encoding:** The `native-country` feature was encoded using mean encoding. The mean of the target variable (`income >50K` encoded as 1) was calculated for each country, and this mean value replaced the category name. This captures the likelihood of earning >$50K based on the country.
2.  **One-Hot Encoding:** The following categorical features were converted into numerical format using one-hot encoding:
    *   `workclass`
    *   `education`
    *   `marital-status`
    *   `occupation`
    *   `relationship`
    *   `race`
    *   `sex`

## Modeling and Evaluation

The dataset was split into training (70%) and testing (30%) sets (`random_state=8`).

### Initial Model

A baseline `RandomForestClassifier` was trained with `n_estimators=10` and `random_state=15`.

### Feature Selection

After an initial training, feature importances were analyzed. The following features were found to have minimal impact on the model's predictive power and were subsequently removed to reduce model complexity:

```
cat__race_ Other                          0.000975
cat__education_ 12th                      0.000773
cat__education_ 5th-6th                   0.000621
cat__marital-status_ Married-AF-spouse    0.000314
cat__education_ 1st-4th                   0.000227
cat__occupation_ Priv-house-serv          0.000081
cat__workclass_ Without-pay               0.000051
cat__education_ Preschool                 0.000045
cat__occupation_ Armed-Forces             0.000014
cat__workclass_ Never-worked              0.000003```

### Model After Feature Selection

The `RandomForestClassifier` was re-trained on the reduced feature set with `n_estimators=100`. This model achieved an accuracy of **85.81%**.

**Classification Report:**
```
              precision    recall  f1-score   support

       <=50K       0.89      0.93      0.91      7422
        >50K       0.75      0.62      0.68      2347

    accuracy                           0.86      9769
   macro avg       0.82      0.78      0.79      9769
weighted avg       0.85      0.86      0.85      9769
```

### Hyperparameter Tuning with RandomizedSearchCV

To further improve the model, `RandomizedSearchCV` was used to find the optimal set of hyperparameters. The search was performed over the following parameter grid:

```python
rf_params = {"max_depth": [5, 8, 15, None, 10],
             "max_features": [5, 7, "auto", 8],
             "min_samples_split": [2, 8, 15, 20],
             "n_estimators": [100, 200, 500, 1000]}
```

The best parameters found were:
```python
{'n_estimators': 1000,
 'min_samples_split': 20,
 'max_features': 8,
 'max_depth': 15}
```

## Final Results

The final model, trained with the optimized hyperparameters, achieved a final accuracy of **86.34%**.

**Final Classification Report:**
```
              precision    recall  f1-score   support

       <=50K       0.88      0.95      0.91      7422
        >50K       0.80      0.57      0.67      2347

    accuracy                           0.86      9769
   macro avg       0.84      0.76      0.79      9769
weighted avg       0.86      0.86      0.86      9769
```
The hyperparameter tuning resulted in a slight increase in overall accuracy and a noticeable improvement in precision for the `>50K` class.
