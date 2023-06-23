# Glossary

## Accuracy
The fraction of predictions a [classification](#classification) model has correctly identified. Formula:

$$
\text{accuracy} = \frac{\text{True Positives + True Negatives}}{\text{Total amount of data points}}
$$

See here for respective definitions:
[True Positives](#true-positive-tp)
[True Negatives](#true-negative-tn)

## Application Programming Interface (API)
An API allows independent applications to communicate with each other and exchange data.

## Classification
Classification refers to dividing a data set into multiple chunks, which are then considered "classes".

## Confusion Matrix
A confusion matrix is a table that is used to define the performance of a [classification](#classification) algorithm.
It classifies the predictions to be either be [true positive](#true-positive-tp), [true negative](#true-negative-tn),
[false positive](#false-positive-fp) or [false negative](#false-negative-fn).

## Decision Tree
A Decision Tree represents the process of conditional evaluation in a tree diagram.

Implemented in Safe-DS as [Decision Tree][safeds.ml.classical.classification.DecisionTree].

## F1-Score
The harmonic mean of [precision](#precision) and [recall](#recall). Formula:

$$
f_1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}
$$

## False Negative (FN)
An outcome is considered to be a false negative, if the data model has mistakenly predicted a value of negative class.

## False Positive (FP)
An outcome is considered to be a false positive, if the data model has mistakenly predicted a value of positive class.

## Feature
Each feature represents a measurable piece of data that can be used for analysis.
It is analogous to a column within a table.

## Linear Regression
Linear Regression is the supervised Machine Learning model in which the model finds the best fit linear line between the independent and dependent variable
i.e. it finds the linear relationship between the dependent and independent variable.

Implemented in Safe-DS as [LinearRegression][safeds.ml.classical.regression.LinearRegression].

## Machine Learning (ML)
Machine Learning is a generic term for artificially generating knowledge through experience.
To achieve this, one can choose between a variety of model options.

## Metric
A data metric is an aggregated calculation within a raw dataset.

## One Hot Encoder
If a column's entries consist of a non-numerical data type, using a One Hot Encoder will create
a new column for each different entry, filling it with a '1' in the respective places, '0' otherwise.

Implemented in Safe-DS as [OneHotEncoder][safeds.data.tabular.transformation.OneHotEncoder].

## Overfitting
Overfitting is a scenario in which a data model is unable to capture the relationship between the input and output variables accurately,
due to not generalizing enough.

## Positive Class
The "Positive Class" consists of all attributes to be considered positive. Consequently, every attribute to not be in this class is considered to be negative class.

## Precision
The ability of a [classification](#classification) model to identify only the relevant data points. Formula:

$$
\text{precision} = \frac{\text{True Positives}}{\text{True Positives + False Positives}}
$$

See here for respective references:
[True Positives](#true-positive-tp)
[False Positives](#false-positive-fp)

## Random Forest
Random Forest is an ML model that works by generating decision trees at random.

Implemented in Safe-DS as [RandomForest][safeds.ml.classical.regression.RandomForest].

## Recall
The ability of a [classification](#classification) model to identify all the relevant data points. Formula:

$$
\text{recall} = \frac{\text{True Positives}}{\text{True Positives + False Negatives}}
$$

See here for respective references:
[True Positives](#true-positive-tp)
[False Negatives](#false-negative-fn)

## Regression
Regression refers to the estimation of continuous dependent variables.

## Regularization
Regularization refers to techniques that are used to calibrate machine learning models
in order to minimize the adjusted loss function and prevent [overfitting](#overfitting) or [underfitting](#underfitting).

## Sample
A sample is a subset of the whole data set.
It is analyzed to uncover the meaningful information in the larger data set.

## Supervised Learning
Supervised Learning is a subcategory of ML. This approach uses algorithms to learn given data.
Those Algorithms might be able to find hidden meaning in data - without being told where to look.

## Tagged Table
In addition to a regular table, a Tagged Table will mark one column as tagged, meaning that
an applied algorithm will train to predict its entries. The marked column is referred to as ["target"](#target).

## Target
The target variable of a dataset is the feature of a dataset about which you want to gain a deeper understanding.

## Test Set
A set of examples used only to assess the performance of a fully-specified [classifier](#classification).

## Training Set
A set of examples used for learning, that is to fit the parameters of the [classifier](#classification).

## True Negative (TN)
An outcome is considered to be a true negative, if the data model has correctly predicted a value of negative class.

## True Positive (TP)
An outcome is considered to be a true positive, if the data model has correctly predicted a value of positive class.

## Underfitting
Underfitting is a scenario in which a data model is unable to capture the relationship between the input and output variables accurately,
due to generalizing too much.

## Validation Set
A set of examples used to the parameters of a [classifier](#classification).
