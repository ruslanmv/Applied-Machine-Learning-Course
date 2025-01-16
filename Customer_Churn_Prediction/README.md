
# Customer Churn Prediction

Welcome to the Customer Churn Prediction module of the Applied Machine Learning Course!

## Motivation

Customer churn, also known as customer attrition, is a critical metric for any business, especially those based on subscription models like telecommunication services, SaaS, and others. It is more cost-effective to retain existing customers than to acquire new ones. Therefore, predicting churn can enable businesses to take proactive steps to retain customers, such as offering targeted incentives or improving service quality, thus improving profitability.

## Learning Objectives

By the end of this module, you will be able to:

*   Understand the concept of customer churn and its importance.
*   Perform Exploratory Data Analysis (EDA) on a churn dataset.
*   Preprocess data for machine learning, including handling missing values and categorical features.
*   Train and evaluate classification models like Logistic Regression to predict churn.
*   Interpret the results of the model and identify key factors influencing churn.
*   Use Python libraries like Pandas, Scikit-learn, and Matplotlib for churn prediction.

## Real-World Applications

*   **Telecom Companies:** Identify customers at risk of leaving and offer them special deals or improve their service.
*   **Subscription Services (e.g., Netflix, Spotify):** Predict users who might cancel their subscriptions and engage them with personalized content or offers.
*   **Banks:** Detect customers who might close their accounts and proactively address their concerns.
*   **Retail:** Identify customers who are likely to stop shopping and target them with promotions.

## Conceptual Overview

In this module, we will focus on using **classification models** to predict customer churn. Classification is a type of supervised learning where the goal is to predict a categorical outcome, in this case, whether a customer will churn (yes/no).

We will primarily use **Logistic Regression**, a simple yet powerful classification algorithm, as our main tool.  Logistic Regression is well-suited for binary classification problems and provides probabilities that a customer will churn, making it easy to interpret the results.

## Tools

*   **Python:** Our primary programming language.
*   **Pandas:** For data manipulation and analysis.
*   **Scikit-learn:** For building and evaluating machine learning models.
*   **Matplotlib:** For data visualization.
*   **Seaborn**: For nicer plots.

## Dataset

We will use the **Telecom Churn dataset** available on Kaggle.

*   **Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
*   **Description:** The dataset contains information about a fictional telecom company and includes customer demographics, services used, account information, and whether the customer churned.
*   **Features:** The dataset includes features like gender, tenure, internet service, contract type, monthly charges, and more. The target variable is "Churn," indicating whether the customer left the service (Yes/No).
*   **Potential Limitations:** This is a synthetic dataset, so the results may not perfectly reflect real-world scenarios. However, it's a great dataset for learning and practicing churn prediction.

## Project Roadmap

1.  **Data Loading and Exploration:** Load the dataset, understand its structure, and perform exploratory data analysis (EDA).
2.  **Data Preprocessing:** Handle missing values, encode categorical features, and prepare the data for modeling.
3.  **Feature Engineering:** Create potentially useful new features if needed.
4.  **Model Training:** Train a Logistic Regression model to predict churn.
5.  **Model Evaluation:** Evaluate the model's performance using appropriate metrics.
6.  **Interpretation and Insights:** Analyze the model's results to identify factors driving churn.

## Step-by-Step Instructions

### 1. Data Loading and Exploration

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Display the first few rows
print(data.head())

# Get basic information about the dataset
print(data.info())

# Summary statistics
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Visualize the distribution of the target variable (Churn)
sns.countplot(x='Churn', data=data)
plt.show()
````

**Explanation:**

  * We import the necessary libraries.
  * We load the dataset using Pandas.
  * We examine the first few rows, data types, summary statistics, and check for missing values.
  * We visualize the distribution of the "Churn" column to see the class imbalance (if any).

### 2\. Data Preprocessing

```python
from sklearn.preprocessing import LabelEncoder

# Convert TotalCharges to numeric, handling errors
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

# Handle missing values in TotalCharges (e.g., impute with median)
data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)

# Encode categorical features using Label Encoding
categorical_cols = data.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()
for col in categorical_cols:
    if col != 'customerID': # Do not encode customer id
        data[col] = label_encoder.fit_transform(data[col])

print(data.head())
```

**Explanation:**

  * We convert 'TotalCharges' to a numeric type, as it's initially read as an object due to some blank spaces.
  * We impute missing values in 'TotalCharges' with the median value.
  * We use Label Encoding to convert categorical features into numerical representations, which is necessary for many machine learning models.

### 3\. Feature Engineering

```python
# Example: Create a new feature 'Tenure_MonthlyCharges_Ratio'
data['Tenure_MonthlyCharges_Ratio'] = data['MonthlyCharges'] / (data['tenure'] + 1)  # Avoid division by zero

print(data.head())
```

**Explanation:**

  * This is a simple example of feature engineering. We create a new feature that might capture the relationship between monthly charges and tenure. You can explore creating other features based on domain knowledge or intuition.

### 4\. Model Training

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Separate features (X) and target (y)
X = data.drop(['Churn', 'customerID'], axis=1)  # Drop customerID as it's not a predictor
y = data['Churn']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

**Explanation:**

  * We separate the features (X) from the target variable (y).
  * We split the data into training and testing sets (80% train, 20% test) to evaluate the model's performance on unseen data.
  * We initialize a Logistic Regression model and train it using the training data.

### 5\. Model Evaluation

```python
# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ROC Curve and AUC
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
```

**Explanation:**

  * We make predictions on the test set using the trained model.
  * We evaluate the model's performance using accuracy, confusion matrix, and classification report, which provides precision, recall, F1-score, and support for each class.
  * We also generate and plot the ROC curve and calculate the AUC score to evaluate the model's ability to distinguish between the two classes.

### 6\. Interpretation and Insights

```python
# Get feature importances (coefficients) from the model
feature_importances = pd.DataFrame({'feature': X.columns, 'importance': model.coef_[0]})
feature_importances = feature_importances.sort_values('importance', ascending=False)

print("\nFeature Importances:\n", feature_importances)

# Visualize feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importances)
plt.title('Feature Importances in Logistic Regression Model')
plt.show()
```

**Explanation:**

  * We extract the coefficients (feature importances) from the trained Logistic Regression model.
  * We create a bar plot to visualize the feature importances, which can help us understand which features have the most significant impact on churn prediction.

## Exercises

1.  **Experiment with different classification models:** Try other classification algorithms like Decision Trees, Random Forests, or Support Vector Machines (SVM) and compare their performance to Logistic Regression.
2.  **Hyperparameter Tuning:** Use techniques like GridSearchCV or RandomizedSearchCV to find the optimal hyperparameters for your chosen model.
3.  **Advanced Feature Engineering:** Create more complex features or use feature selection techniques to improve model performance. For example, you could try creating interaction terms between features.
4.  **Handle Class Imbalance:** If the dataset has a significant class imbalance (many more non-churners than churners), try techniques like oversampling (e.g., SMOTE) or undersampling to balance the classes and see if it improves performance.
5.  **Try Different Evaluation Metrics:** Investigate other relevant evaluation metrics for churn prediction, such as the area under the precision-recall curve (AUPRC).

## Suggested Solutions (Hints)

  * **Exercise 1:** You can easily import and use other classifiers from Scikit-learn (e.g., `from sklearn.tree import DecisionTreeClassifier`).
  * **Exercise 2:** Scikit-learn provides `GridSearchCV` and `RandomizedSearchCV` for hyperparameter tuning.
  * **Exercise 3:** Think about how different features might interact with each other. For example, you could create a feature that represents the ratio of monthly charges to total charges.
  * **Exercise 4:** The `imblearn` library provides implementations of oversampling and undersampling techniques.
  * **Exercise 5:** Scikit-learn provides functions to calculate AUPRC.

## Further Resources

  * **Scikit-learn Documentation:** [https://scikit-learn.org/stable/](https://www.google.com/url?sa=E&source=gmail&q=https://scikit-learn.org/stable/)
  * **Pandas Documentation:** [https://pandas.pydata.org/pandas-docs/stable/](https://www.google.com/url?sa=E&source=gmail&q=https://pandas.pydata.org/pandas-docs/stable/)
  * **Interpretable Machine Learning (Book):** [https://christophm.github.io/interpretable-ml-book/](https://www.google.com/url?sa=E&source=gmail&q=https://christophm.github.io/interpretable-ml-book/) (Chapter on Logistic Regression)
  * **Kaggle Kernels on Telco Churn:** Explore other people's approaches to this dataset on Kaggle for inspiration.

## Contact

For any queries or feedback related to this module, feel free to open an issue in the main repository or contact us at \[contact@ruslanmv.com].

-----

Happy Learning\!


