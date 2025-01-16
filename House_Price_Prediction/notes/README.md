# House Price Prediction

Welcome to the **House Price Prediction** section of the Applied Machine Learning Course. This module focuses on predicting house prices based on various features such as size, number of bedrooms, location, and more.

---

## Problem Statement
Predict the price of a house using a dataset that includes features like:
- Size of the house (square footage)
- Number of bedrooms
- Location
- Other relevant features

---

## Objectives
By completing this project, you will:
1. Understand and apply **Linear Regression** for predictive modeling.
2. Learn about **Feature Scaling** to preprocess data for improved accuracy.
3. Work with **Scikit-learn** to implement machine learning models.
4. Use the **Boston Housing dataset**, a popular dataset for regression tasks.

---

## Dataset
We will use the **Boston Housing dataset**, which is available on Kaggle. The dataset contains information about houses in Boston and their respective prices.

[Download the Dataset](https://www.kaggle.com)

---

## Tools and Libraries
Ensure you have the following tools and libraries installed:
- Python 3.x
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Jupyter Notebook

Install required libraries using:
```bash
pip install scikit-learn pandas numpy matplotlib
```

---

## Step-by-Step Guide

### 1. Data Loading
Load the Boston Housing dataset using Scikit-learn's built-in dataset loader:
```python
from sklearn.datasets import load_boston
import pandas as pd

# Load dataset
boston = load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['PRICE'] = boston.target

# Display the first few rows
print(data.head())
```

---

### 2. Exploratory Data Analysis (EDA)
Perform EDA to understand the dataset structure:
```python
# Summary statistics
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Plot correlation heatmap
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
```

---

### 3. Feature Scaling
Scale features using **StandardScaler**:
```python
from sklearn.preprocessing import StandardScaler

# Define features and target variable
X = data.drop(columns=['PRICE'])
y = data['PRICE']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

### 4. Model Training
Train a linear regression model:
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Model coefficients
print("Coefficients:", model.coef_)
```

---

### 5. Model Evaluation
Evaluate the model using metrics like **Mean Squared Error (MSE)** and **R² score**:
```python
from sklearn.metrics import mean_squared_error, r2_score

# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)
```

---

### 6. Visualization
Visualize the predicted vs actual values:
```python
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()
```

---

## Results
- **Model Coefficients**: Interpretation of feature importance.
- **Model Performance**: Understand how well the model predicts house prices.

---

## Next Steps
Once you complete this project:
1. Experiment with additional features or datasets to improve accuracy.
2. Explore advanced regression techniques like **Ridge Regression** or **Random Forest Regression**.

---

## Folder Structure
- `data/`: Contains the dataset.
- `notebooks/`: Jupyter notebooks with step-by-step implementation.
- `scripts/`: Python scripts for data preprocessing and modeling.

---

Happy Learning!
