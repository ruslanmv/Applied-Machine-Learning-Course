# House Price Prediction

Welcome to the House Price Prediction module of the Applied Machine Learning Course!

## Motivation

Predicting house prices is a classic problem in real estate and a great introduction to regression tasks in machine learning. Accurate house price predictions are valuable for buyers, sellers, real estate agents, and investors to make informed decisions. This module will guide you through building a machine-learning model to predict house prices based on various features.

## Learning Objectives

By the end of this module, you will be able to:

*   Understand the problem of house price prediction and its real-world applications.
*   Perform Exploratory Data Analysis (EDA) on housing data.
*   Preprocess data for linear regression, including handling missing values and feature scaling.
*   Train and evaluate a linear regression model for house price prediction.
*   Interpret the results of the model and understand the factors that influence house prices.
*   Use Python libraries like Pandas, Scikit-learn, Matplotlib, and Seaborn for regression tasks.

## Real-World Applications

*   **Real Estate Appraisal:** Estimating the market value of properties.
*   **Investment Decisions:** Identifying undervalued or overvalued properties for investment purposes.
*   **Real Estate Platforms:** Providing price estimates to buyers and sellers on online platforms (e.g., Zillow's Zestimate).
*   **Urban Planning:** Understanding the factors that affect housing prices in different areas.
*   **Financial Institutions:** Assessing collateral risk for mortgage lending.

## Conceptual Overview

House price prediction is a **regression** problem because we are trying to predict a continuous target variable (price). We will primarily use **Linear Regression**, a fundamental and widely used statistical method for modeling the relationship between a dependent variable and one or more independent variables.

**Key Concepts:**

*   **Linear Regression:** Assumes a linear relationship between the features and the target variable. The model learns coefficients for each feature that represent the change in the target variable for a unit change in the feature, holding other features constant.
*   **Feature Scaling:** Transforming numerical features to a similar scale (e.g., using standardization or normalization). This helps improve the performance and convergence of gradient-based optimization algorithms used in linear regression.
*   **Model Evaluation:** Assessing the performance of the regression model using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared.

## Tools

*   **Python:** Our primary programming language.
*   **Scikit-learn:** For building and evaluating machine learning models, including linear regression.
*   **Pandas:** For data manipulation and analysis.
*   **Matplotlib:** For data visualization.
*   **Seaborn:** For statistical data visualization.
*   **Jupyter Notebook:** An interactive environment for writing and running code, as well as creating visualizations.

## Dataset

We will use the **Boston Housing dataset**, a classic dataset for regression tasks.

*   **Source:** Available in Scikit-learn's `datasets` module and also on Kaggle: [Boston Housing Dataset](https://www.kaggle.com/vikক্ষেরh/boston-house-prices)
*   **Description:** The dataset contains information about housing in the area of Boston, Massachusetts.
*   **Features:**
    *   `CRIM`: Per capita crime rate by town.
    *   `ZN`: Proportion of residential land zoned for lots over 25,000 sq. ft.
    *   `INDUS`: Proportion of non-retail business acres per town.
    *   `CHAS`: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
    *   `NOX`: Nitric oxides concentration (parts per 10 million).
    *   `RM`: Average number of rooms per dwelling.
    *   `AGE`: Proportion of owner-occupied units built prior to 1940.
    *   `DIS`: Weighted distances to five Boston employment centers.
    *   `RAD`: Index of accessibility to radial highways.
    *   `TAX`: Full-value property-tax rate per $10,000.
    *   `PTRATIO`: Pupil-teacher ratio by town.
    *   `B`: 1000(Bk - 0.63)^2 where Bk is the proportion of people of African American descent by town.
    *   `LSTAT`: % lower status of the population.
    *   `MEDV`: Median value of owner-occupied homes in $1000's (target variable). This is often replaced with `PRICE` depending on where you get the data.
*   **Potential Limitations:** The dataset is relatively small and old (collected in 1978). The feature `B` is based on racial demographics, which raises ethical concerns and would not be used in modern analyses. However, the dataset is still valuable for learning purposes.

## Project Roadmap

1.  **Data Loading and Exploration:** Load the Boston Housing dataset, understand its structure, and perform exploratory data analysis (EDA).
2.  **Data Preprocessing:** Handle missing values (if any) and perform feature scaling.
3.  **Model Training:** Train a Linear Regression model to predict house prices.
4.  **Model Evaluation:** Evaluate the model's performance using appropriate metrics (MAE, MSE, RMSE, R-squared).
5.  **Interpretation and Insights:** Analyze the model's coefficients to understand the factors that influence house prices.

## Step-by-Step Instructions

### 1. Data Loading and Exploration

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston

# Load the Boston Housing dataset
boston_data = load_boston()
data = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
data['PRICE'] = boston_data.target

# Display the first few rows
print(data.head())

# Get basic information about the dataset
print(data.info())

# Check for missing values
print(data.isnull().sum())

# Explore the distribution of the target variable (PRICE)
sns.histplot(data['PRICE'], bins=50)
plt.title('Distribution of House Prices')
plt.xlabel('Price ($1000s)')
plt.show()

# Explore the correlation between features and the target variable
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```

**Explanation:**

  * We import the necessary libraries (Pandas, Matplotlib, Seaborn, `load_boston` from Scikit-learn).
  * We load the Boston Housing dataset using `load_boston()`.
  * We create a Pandas DataFrame from the data and add the target variable (`PRICE`).
  * We examine the first few rows, data types, and check for missing values.
  * We visualize the distribution of the target variable (`PRICE`) using a histogram.
  * We create a correlation matrix and visualize it using a heatmap to understand the relationships between features and the target variable.

### 2. Data Preprocessing

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separate features (X) and target (y)
X = data.drop('PRICE', axis=1)
y = data['PRICE']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

**Explanation:**

  * We separate the features (X) from the target variable (y).
  * We split the data into training and testing sets (80% train, 20% test).
  * We apply feature scaling using `StandardScaler`. This standardizes the features by removing the mean and scaling to unit variance. It's often beneficial for linear regression.

### 3. Model Training

```python
from sklearn.linear_model import LinearRegression

# Initialize and train a Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
```

**Explanation:**

  * We initialize a `LinearRegression` model from Scikit-learn.
  * We train the model using the training data (`X_train`, `y_train`).

### 4. Model Evaluation

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Make predictions on the test set
y_pred = lr_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Visualize actual vs. predicted prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs. Predicted Prices")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red') # Diagonal line
plt.show()
```

**Explanation:**

  * We make predictions on the test set using the trained model.
  * We evaluate the model using the following metrics:
      * **Mean Absolute Error (MAE):** The average absolute difference between the predicted and actual prices.
      * **Mean Squared Error (MSE):** The average squared difference between the predicted and actual prices.
      * **Root Mean Squared Error (RMSE):** The square root of MSE, which is in the same units as the target variable (thousands of dollars in this case).
      * **R-squared (R2):** A measure of how well the model fits the data. It represents the proportion of the variance in the target variable that is explained by the model. A higher R-squared is generally better.
  * We visualize the actual vs. predicted prices using a scatter plot. The diagonal line represents perfect predictions.

### 5. Interpretation and Insights

```python
# Get the coefficients of the linear regression model
coefficients = pd.DataFrame({'feature': X.columns, 'coefficient': lr_model.coef_})
coefficients = coefficients.sort_values('coefficient')

# Plot the coefficients
plt.figure(figsize=(10, 6))
plt.barh(coefficients['feature'], coefficients['coefficient'])
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.title("Feature Coefficients in Linear Regression Model")
plt.show()
```

**Explanation:**

  * We extract the coefficients learned by the `LinearRegression` model.
  * We create a bar plot to visualize the coefficients.

**Interpreting Coefficients:**

*   **Magnitude:** The larger the absolute value of a coefficient, the stronger its influence on the predicted house price.
*   **Sign:**
    *   **Positive coefficient:** Indicates that an increase in the feature's value leads to an increase in the predicted house price, holding other features constant.
    *   **Negative coefficient:** Indicates that an increase in the feature's value leads to a decrease in the predicted house price, holding other features constant.

For example, a large positive coefficient for `RM` (average number of rooms) would suggest that houses with more rooms tend to have higher prices. A large negative coefficient for `LSTAT` (% lower status of the population) would suggest that houses in neighborhoods with a higher percentage of lower-status population tend to have lower prices.

**Important Note:** The Boston Housing dataset has ethical concerns related to the `B` (racial demographics) feature. In a real-world scenario, you would not use such a feature due to its potential for perpetuating bias.

## Exercises

1.  **Polynomial Regression:**  Try using `PolynomialFeatures` from Scikit-learn to create polynomial features (e.g., squared terms, interaction terms) and see if it improves the model's performance.
2.  **Regularization:** Experiment with regularized linear regression models like Ridge Regression (`Ridge`) and Lasso Regression (`Lasso`) to see if they help prevent overfitting and improve generalization.
3.  **Different Dataset:** Apply the same techniques to a different housing dataset or a different regression problem altogether. You can find many datasets on Kaggle or the UCI Machine Learning Repository.
4.  **Feature Engineering:** Create new features that you think might be relevant for predicting house prices. For example, you could create a feature representing the age of the house or the distance to the nearest school.

## Suggested Solutions (Hints)

1.  **Exercise 1:** Use `PolynomialFeatures` to transform the training data before fitting the `LinearRegression` model.
2.  **Exercise 2:** Refer to the Scikit-learn documentation for `Ridge` and `Lasso` models. They have a hyperparameter `alpha` that controls the strength of regularization.
3.  **Exercise 3:** Explore datasets on Kaggle or the UCI Machine Learning Repository.
4.  **Exercise 4:** Think about how different factors might influence house prices and try to create features that capture those relationships.

## Further Resources

*   **Scikit-learn Documentation:** [https://scikit-learn.org/stable/](https://www.google.com/url?sa=E&source=gmail&q=https://scikit-learn.org/stable/)
*   **Kaggle:** [https://www.kaggle.com/](https://www.google.com/url?sa=E&source=gmail&q=https://www.kaggle.com/) (Search for housing datasets and kernels)
*   **UCI Machine Learning Repository:** [https://archive.ics.uci.edu/ml/index.php](https://www.google.com/url?sa=E&source=gmail&q=https://archive.ics.uci.edu/ml/index.php)

## Contact

For any queries or feedback related to this module, feel free to open an issue in the main repository or contact us at \[contact@ruslanmv.com].

-----

Happy Learning\!

