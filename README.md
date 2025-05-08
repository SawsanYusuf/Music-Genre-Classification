# Music Genre Classification
![](https://github.com/SawsanYusuf/Music-Genre-Classification/blob/main/images/marius-masalar-rPOmLGwai2w-unsplash.jpg)

## Table of Contents

1.  [Overview](#overview)
2.  [Reading Data](#reading-data)
3.  [Data Preparation](#data-preparation)
    * [Cleaning Data](#cleaning-data)
    * [Missing Values](#missing-values)
    * [Making New Features](#making-new-features)
4.  [Data Preprocessing (Continued)](#data-preprocessing-continued)
    * [Removing Outliers](#removing-outliers)
    * [Applying Log Transformation](#applying-log-transformation)
    * [Polynomial Features](#polynomial-features)
    * [Scaling Features](#scaling-features)
    * [Making Preprocessing Pipeline](#making-preprocessing-pipeline)
5.  [Machine Learning Models](#machine-learning-models)
    * [Traditional Models](#traditional-models)
    * [Ensemble and Boostings Models](#ensemble-and-boostings-models)
    * [Fine-Tuning Best Models](#fine-tuning-best-models)
        * [Logistic Regression](#logistic-regression)
        * [CatBoost Modeling](#catboost-modeling)
        * [Stacking Classifier](#stacking-classifier)
6.  [Making Submission](#making-submission)
7.  [Conclusion](#conclusion)

## Overview

This project focuses on music genre classification using machine learning techniques. The dataset comprises nearly 14,000 tracks, spanning a diverse range of musical genres. The goal is to train a model ensemble capable of accurately classifying each track into its respective genre, from Pop to Jazz, Rock to Classical, and others. This project explores the application of data science to the analysis and categorization of music, leveraging the patterns and features within audio data to understand and classify the rich tapestry of musical expression.

## Reading Data

This section of the notebook focuses on loading the music genre classification dataset into a pandas DataFrame for subsequent analysis.

The following libraries are imported:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
```
* **pandas:** Used for data manipulation and creating DataFrames.
* **numpy:** Used for numerical operations.
* **matplotlib.pyplot:** Used for basic plotting.
* **seaborn:** Used for enhanced statistical visualizations.
* **plotly.express:** Used for interactive visualizations.
* **sklearn.preprocessing.LabelEncoder:** Used for encoding categorical features (though not directly used in this "Reading Data" section, it's imported, suggesting it will be used later).

The path to the dataset CSV file is defined:
```
csv_file_path = '/content/shai-music-genre-classification-2/train (1).csv'
```
The dataset is then loaded into a pandas DataFrame named `df` using the `pd.read_csv()` function:
```
df = pd.read_csv(csv_file_path)
```
Finally, the notebook displays the first few rows of the DataFrame using `df.head()` to provide an initial look at the data, and `df.info()` is used to get a concise summary of the DataFrame.

## Data Preparation

This section focuses on preparing the music genre classification dataset for machine learning modeling. It involves cleaning the data, handling missing values, and engineering new features that might be informative for genre prediction.

### Cleaning Data

This subsection addresses potential inconsistencies and redundancies in the dataset.

First, the notebook checks the number of unique values in each column to identify high-cardinality features:
```
print(pd.DataFrame({'Data Type': df.dtypes,
                   'Unique Values': df.nunique(),
                   'Null Values': df.isnull().sum(),
                   '% Null Values': (df.isnull().sum() / len(df)) * 100}).sort_values(by='Null Values', ascending=False).style.background_gradient(cmap='YlOrRd'))
```
This code provides a summary of each column, including its data type, the number of unique values, the number of missing values, and the percentage of missing values.

The notebook then decides to drop the 'Id', 'Artist Name', and 'Track Name' columns due to their high cardinality and the likelihood that they won't generalize well for genre classification:
```
df.drop(columns=['Id', 'Artist Name', 'Track Name'], inplace=True)
```
Next, the notebook identifies categorical features that will require type conversion or encoding later:
```
cat_features = ['time_signature', 'mode', 'key', 'Class']
```
It then checks for duplicate rows in the DataFrame:
```
print(f"Number of duplicate rows: {df.duplicated().sum()}")
```
And drops any duplicate rows found:
```
df.drop_duplicates(inplace=True)
print(f"Shape of DataFrame after dropping duplicates: {df.shape}")
```
### Missing Values

This subsection deals with handling the missing values identified in the previous step.

A heatmap is generated to visualize the pattern of missing values across different columns:
```
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()
```
The notebook then focuses on imputing the missing values in the 'Popularity', 'key', and 'instrumentalness' columns. Histograms and box plots are displayed for these columns to understand their distributions before imputation.

Based on the distributions (potential outliers), the median is chosen as the imputation strategy for the missing values in 'Popularity', 'key', and 'instrumentalness'. The `SimpleImputer` from scikit-learn is used for this purpose:
```
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
df[nullCols] = imputer.fit_transform(df[nullCols])

print("Number of null values after imputation:")
print(df.isnull().sum())
```
### Making New Features

This subsection focuses on creating new features from existing ones that might provide additional information for the model.

The 'duration_in min/ms' column, which seems to contain duration in both minutes and milliseconds, is converted to a consistent unit (minutes) and a new feature 'duration_in min' is created:
```
df['duration_in min'] = np.where(df['duration_in min/ms'] > 1000, df['duration_in min/ms'] / 60000, df['duration_in min/ms'])
df.drop(columns=['duration_in min/ms'], inplace=True)
print(df[['duration_in min']].head())
```
Binning is applied to the 'Popularity', 'speechiness', 'valence', 'instrumentalness', and 'liveness' features to create categorical features:
```
popularity_bins = [0, 25, 50, 75, 100]
popularity_labels = ['Low', 'Medium', 'High', 'Very High']
df['Popularity_Bins'] = pd.cut(df['Popularity'], bins=popularity_bins, labels=popularity_labels, right=True, include_lowest=True)

speech_bins = [0, 0.33, 0.66, 1]
speech_labels = ['Non-Speech', 'Speech-Heavy', 'Mostly Speech']
df['speech_labels'] = pd.cut(df['speechiness'], bins=speech_bins, labels=speech_labels, right=True, include_lowest=True)

mood_bins = [0, 0.25, 0.5, 0.75, 1]
mood_labels = ['Negative', 'Neutral', 'Slightly Positive', 'Positive']
df['mood'] = pd.cut(df['valence'], bins=mood_bins, labels=mood_labels, right=True, include_lowest=True)

instrumental_bins = [-1, 0.5, 1]
instrumental_labels = ['Vocals Present', 'Instrumental']
df['instrumental'] = np.where(df['instrumentalness'] <= 0.5, instrumental_labels[0], instrumental_labels[1])

live_bins = [-1, 0.8, 1]
live_labels = ['Not Live', 'Live']
df['live'] = np.where(df['liveness'] <= 0.8, live_labels[0], live_labels[1])

print(df[['Popularity', 'Popularity_Bins']].head())
print(df[['speechiness', 'speech_labels']].head())
print(df[['valence', 'mood']].head())
print(df[['instrumentalness', 'instrumental']].head())
print(df[['liveness', 'live']].head())
```
New features are engineered based on the 'Artist Name' and 'Track Name' (before they were dropped) by calculating their lengths and the number of words:
```
# df['ArtistName_Lenght'] = df['Artist Name'].apply(len)
# df['TrackName_Lenght'] = df['Track Name'].apply(len)
# df['ArtistName_Words'] = df['Artist Name'].apply(lambda x: len(str(x).split()))
# df['TrackName_Words'] = df['Track Name'].apply(lambda x: len(str(x).split()))
```
### Removing Outliers

This subsection aims to identify and potentially remove or mitigate the impact of outliers in the numerical features of the dataset.

First, descriptive statistics are calculated for the DataFrame to get an overview of the range and distribution of numerical features:
```
print(df.describe())
```
The notebook then calculates the skewness and kurtosis of specific numerical features ('loudness', 'speechiness', 'liveness', 'duration_in min', 'ArtistName_Lenght', 'TrackName_Lenght'):
```
outliers_cols = ['loudness', 'speechiness', 'liveness', 'duration_in min', 'ArtistName_Lenght', 'TrackName_Lenght']
print(df[outliers_cols].skew())
print(df[outliers_cols].kurt())
```
Skewness measures the asymmetry of the probability distribution, and kurtosis measures the "tailedness" of the distribution.

Box plots and histograms are generated for the same set of features to visually inspect the presence and distribution of outliers:
```
plt.figure(figsize=(20, 8))
for i, col in enumerate(outliers_cols):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 8))
for i, col in enumerate(outliers_cols):
    plt.subplot(2, 3, i + 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()
```
The notebook then attempts to remove outliers using the Interquartile Range (IQR) method for a subset of the features:
```
def remove_outliers_iqr(df, cols):
    df_cleaned = df.copy()
    for col in cols:
        Q1 = df_cleaned[col].quantile(0.01)
        Q3 = df_cleaned[col].quantile(0.99)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        mask = (df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)
        df_cleaned = df_cleaned[mask]
    return df_cleaned

outlier_removal_cols = ['loudness', 'speechiness', 'liveness', 'duration_in min']
clean_df_iqr = remove_outliers_iqr(df, outlier_removal_cols)
print(f"Shape of DataFrame before IQR outlier removal: {df.shape}")
print(f"Shape of DataFrame after IQR outlier removal: {clean_df_iqr.shape}")
print(clean_df_iqr.describe())
```
### Applying Log Transformation

This subsection addresses the skewness observed in some of the numerical features by applying a log transformation.

The skewness of a set of 'skewed_features' is calculated:
```
skewed_features = ['speechiness', 'acousticness', 'instrumentalness', 'liveness', 'ArtistName_Lenght', 'TrackName_Lenght', 'ArtistName_Words', 'TrackName_Words']
print(df[skewed_features].skew())
print(df[skewed_features].kurt())
```
Histograms of the original skewed features are plotted:
```
plt.figure(figsize=(25, 8))
for i, col in enumerate(skewed_features):
    plt.subplot(2, 4, i + 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col} (Original)')
plt.tight_layout()
plt.show()
```
A log transformation (using `np.log1p` to handle zero values) is applied to these skewed features, and the transformed data is stored in a new DataFrame `trans_df`:
```
trans_df = df.copy()
for col in skewed_features:
    trans_df[col] = np.log1p(trans_df[col])
print(trans_df[skewed_features].head())
```
Histograms of the log-transformed features are then plotted to visualize the change in their distributions:
```
plt.figure(figsize=(25, 8))
for i, col in enumerate(skewed_features):
    plt.subplot(2, 4, i + 1)
    sns.histplot(trans_df[col], kde=True)
    plt.title(f'Distribution of {col} (Log Transformed)')
plt.tight_layout()
plt.show()
```
Finally, the skewness and kurtosis of the log-transformed features are calculated to check if the transformation helped in reducing the skewness:
```
print(trans_df[skewed_features].skew())
print(trans_df[skewed_features].kurt())
```
### Polynomial Features

This subsection explores the creation of new features by raising existing features to certain powers and including interaction terms.

The notebook uses `PolynomialFeatures` from scikit-learn to generate polynomial features of degree 2:
```
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
poly_data = poly.fit_transform(df.drop('Class', axis=1))
poly_df = pd.DataFrame(poly_data, columns=poly.get_feature_names_out(df.drop('Class', axis=1).columns))
print(poly_df.head())
print(poly_df.describe())
```
### Scaling Features

This subsection focuses on scaling the numerical features to have a similar range.

The notebook uses `MinMaxScaler` and `StandardScaler` from scikit-learn for feature scaling:
```
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# MinMaxScaler
scaler_minmax = MinMaxScaler()
df_scaled_minmax = pd.DataFrame(scaler_minmax.fit_transform(df.drop('Class', axis=1)), columns=df.drop('Class', axis=1).columns)
print("DataFrame scaled with MinMaxScaler:")
print(df_scaled_minmax.head())
print(df_scaled_minmax.describe())

# StandardScaler
scaler_standard = StandardScaler()
df_scaled_standard = pd.DataFrame(scaler_standard.fit_transform(df.drop('Class', axis=1)), columns=df.drop('Class', axis=1).columns)
print("\nDataFrame scaled with StandardScaler:")
print(df_scaled_standard.head())
print(df_scaled_standard.describe())
```
### Making Preprocessing Pipeline

This subsection creates a comprehensive preprocessing pipeline using scikit-learn's `ColumnTransformer` and `Pipeline`.
```
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Identify categorical and numerical features
categorical_features = ['mode', 'time_signature', 'Popularity_Bins', 'speech_labels', 'mood', 'instrumental', 'live']
numerical_features = df.drop(columns=['Class'] + categorical_features).columns.tolist()

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create the preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'  # Keep other columns (if any) unchanged
)

# Create the full preprocessing pipeline
preprocessing_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Fit and transform the training data
org_df_processed = preprocessing_pipeline.fit_transform(df.drop('Class', axis=1))

# Get feature names after preprocessing
feature_names = preprocessing_pipeline.named_steps['preprocessor'].get_feature_names_out()
org_df_processed = pd.DataFrame(org_df_processed, columns=feature_names)

print("Processed DataFrame shape:", org_df_processed.shape)
print(org_df_processed.head())
```
## Machine Learning Models

This section explores various machine learning models for the music genre classification task. It starts by training and evaluating traditional classification algorithms and then moves on to ensemble and boosting techniques.

### Traditional Models

This subsection trains and evaluates several standard classification models from scikit-learn:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Linear Support Vector Classifier (LinearSVC)
* Support Vector Classifier (SVC)
* Multilayer Perceptron (MLPClassifier)

For each model, the notebook performs training, prediction, and evaluation using metrics like accuracy, balanced accuracy, F1-score, classification report, and confusion matrices.

For example, for Logistic Regression:
```
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=42, max_iter=2000)
lr.fit(X_train, y_train)
lrPreds = lr.predict(X_test)
make_classification_

plots(lrPreds, 'Logistic Regression')
test_performance(lr, X_test, y_test)
```

### Ensemble and Boostings Models

This subsection explores ensemble learning methods:

* Random Forest Classifier
* Balanced Random Forest Classifier
* Gradient Boosting Classifier
* HistGradientBoostingClassifier
* XGBoost Classifier
* LightGBM Classifier
* CatBoost Classifier

Each model is trained and evaluated similarly to the traditional models.

For example, for Random Forest Classifier:

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rfPreds = rf.predict(X_test)
make_classification_plots(rfPreds, 'Random Forest')
test_performance(rf, X_test, y_test)
```

### Fine-Tuning Best Models

This subsection aims to improve the performance of some of the top-performing models by searching for the optimal combination of their hyperparameters.

#### Logistic Regression

Hyperparameter tuning for Logistic Regression using `GridSearchCV`:

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold

param_grid_lr = {'C': np.logspace(-4, 4, 20),
                   'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']}

grid_search_lr = GridSearchCV(LogisticRegression(random_state=42, max_iter=500),
                               param_grid_lr,
                               scoring='balanced_accuracy',
                               cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                               n_jobs=-1)
grid_search_lr.fit(X_train, y_train)
print("Best parameters for Logistic Regression:", grid_search_lr.best_params_)
print("Best balanced accuracy for Logistic Regression:", grid_search_lr.best_score_)
```

#### CatBoost Modeling

Hyperparameter tuning for CatBoost using `GridSearchCV`:

```python
from catboost import CatBoostClassifier

parameters_cat = {'iterations': [500, 1000, 1500],
                  'learning_rate': [0.01, 0.05, 0.1],
                  'depth': [4, 6, 8],
                  'l2_leaf_reg': [1, 3, 5]}

cat_classifier = CatBoostClassifier(random_state=42, verbose=0)

cat_grid_search = GridSearchCV(estimator=cat_classifier,
                               param_grid=parameters_cat,
                               scoring='balanced_accuracy',
                               cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                               n_jobs=-1)

cat_grid_search.fit(X_train, y_train)
print("CatBoost Best Parameters:", cat_grid_search.best_params_)
print("CatBoost Best Score:", cat_grid_search.best_score_)
```

#### Stacking Classifier

Combining multiple models using a Stacking Classifier:

```python
from sklearn.ensemble import StackingClassifier

estimators = [
    ('lr', LogisticRegression(random_state=42, solver='newton-cg', C=10.0, max_iter=500)),
    ('mlp', MLPClassifier(random_state=42, hidden_layer_sizes=(512,), max_iter=100, early_stopping=True, solver='adam', learning_rate='adaptive')),
    ('gb', GradientBoostingClassifier(random_state=42, learning_rate=0.1, max_depth=5, n_estimators=100)),
    ('lgbm', LGBMClassifier(random_state=42, learning_rate=0.1, n_estimators=200, num_leaves=31)),
    ('cat', CatBoostClassifier(random_state=42, iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3, verbose=0))
]

stack_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(random_state=42, solver='liblinear'))
stack_model.fit(X_train, y_train)
stackPreds = stack_model.predict(X_test)
make_classification_plots(stackPreds, 'StackingClassifier')
test_performance(stack_model, X_test, y_test)
```

## Making Submission

This section outlines the steps taken to prepare the predictions for submission.

The notebook focuses on using the best-performing model (CatBoost Classifier) after hyperparameter tuning.

The CatBoost Classifier is trained on the entire training dataset using the best hyperparameters:

```python
best_cat_params = {'depth': 6, 'iterations': 1000, 'l2_leaf_reg': 3, 'learning_rate': 0.05}
final_cat_model = CatBoostClassifier(random_state=42, **best_cat_params, verbose=1)
final_cat_model.fit(X_train, y_train)
```

Predictions are made on the test dataset:

```python
catPreds_final = final_cat_model.predict(X_test)
```

The predictions are then formatted into a submission file.

## Conclusion

This notebook provides a comprehensive workflow for music genre classification, covering data loading, preprocessing, model training, hyperparameter tuning, and submission preparation. The project explores various machine learning models, including traditional classifiers and ensemble methods, and demonstrates techniques for optimizing model performance.





