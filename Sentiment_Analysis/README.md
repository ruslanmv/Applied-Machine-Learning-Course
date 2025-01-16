
# Sentiment Analysis of Movie Reviews

Welcome to the Sentiment Analysis of Movie Reviews module of the Applied Machine Learning Course!

## Motivation

Sentiment analysis, also known as opinion mining, is a crucial task in Natural Language Processing (NLP) with broad applications in understanding public opinion, brand perception, and customer feedback. By automatically classifying the sentiment expressed in text, businesses can gain insights into how their products or services are perceived, monitor their brand reputation, and respond to customer concerns in real time.

## Learning Objectives

By the end of this module, you will be able to:

*   Understand the concept of sentiment analysis and its applications.
*   Perform text preprocessing techniques like tokenization, stop word removal, and stemming/lemmatization.
*   Use feature extraction methods like Bag-of-Words (BoW) and TF-IDF to represent text data numerically.
*   Train and evaluate text classification models like Naive Bayes and Logistic Regression for sentiment analysis.
*   Use Python libraries like NLTK and TextBlob for NLP tasks.
*   Interpret the results of sentiment analysis and identify key factors influencing sentiment.

## Real-World Applications

*   **Brand Monitoring:** Track how people are talking about a brand on social media and review sites.
*   **Customer Feedback Analysis:** Analyze customer reviews and feedback to understand customer satisfaction and identify areas for improvement.
*   **Market Research:** Gauge public opinion about new products or political candidates.
*   **Social Media Analysis:** Understand trends and public sentiment during events or crises.
*   **Product Development:** Use customer reviews to identify features that customers like or dislike.

## Conceptual Overview

In this module, we will focus on **binary sentiment classification**, where the goal is to classify movie reviews as either **positive** or **negative**. We will primarily use techniques from **Natural Language Processing (NLP)** to process and analyze the text data.

We will explore two popular feature extraction methods:

1.  **Bag-of-Words (BoW):** Represents text as a collection of individual words, disregarding grammar and word order.
2.  **TF-IDF (Term Frequency-Inverse Document Frequency):**  Weights words based on their importance in a document and across the entire corpus.

For classification, we will use:

1.  **Naive Bayes:** A probabilistic classifier based on Bayes' theorem with a "naive" assumption of independence between features.
2.  **Logistic Regression:** A linear model for binary classification that provides probabilities of sentiment.

## Tools

*   **Python:** Our primary programming language.
*   **NLTK (Natural Language Toolkit):** A comprehensive library for NLP tasks like tokenization, stemming, lemmatization, and more.
*   **TextBlob:** A user-friendly library built on top of NLTK, providing a simple API for common NLP tasks.
*   **Scikit-learn:** For building and evaluating machine learning models.
*   **Pandas:** For data manipulation and analysis.
*   **Matplotlib:** For data visualization.

## Dataset

We will use the **IMDB movie review dataset**, a widely used dataset for sentiment analysis.

*   **Source:**  The dataset is available through different sources, including:
    *   [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
    *   [Stanford AI](https://ai.stanford.edu/~amaas/data/sentiment/)
*   **Description:** The dataset contains 50,000 movie reviews from IMDB, labeled as positive or negative.
*   **Features:** Each review is represented as raw text. The target variable is "sentiment," indicating whether the review is positive or negative (typically encoded as 1 or 0).
*   **Potential Limitations:** The dataset might contain some noise or inconsistencies in labeling. Also, movie reviews can be quite long, which can pose challenges for some NLP techniques.

## Project Roadmap

1.  **Data Loading and Exploration:** Load the dataset, understand its structure, and perform exploratory data analysis (EDA).
2.  **Data Preprocessing:** Clean the text data, perform tokenization, remove stop words, and apply stemming or lemmatization.
3.  **Feature Extraction:** Convert the text data into numerical representations using Bag-of-Words and TF-IDF.
4.  **Model Training:** Train Naive Bayes and Logistic Regression models to classify sentiment.
5.  **Model Evaluation:** Evaluate the models' performance using appropriate metrics.
6.  **Interpretation and Insights:** Analyze the models' results to identify factors driving sentiment.

## Step-by-Step Instructions

### 1. Data Loading and Exploration

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("IMDB Dataset.csv")

# Display the first few rows
print(data.head())

# Get basic information about the dataset
print(data.info())

# Check for missing values
print(data.isnull().sum())

# Visualize the distribution of the target variable (sentiment)
sns.countplot(x='sentiment', data=data)
plt.show()
````

**Explanation:**

  * We import the necessary libraries.
  * We load the dataset using Pandas.
  * We examine the first few rows, data types, and check for missing values.
  * We visualize the distribution of the "sentiment" column to see the class balance.

### 2\. Data Preprocessing

```python
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')

# Initialize stemmer and stop words
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove HTML tags
    text = re.sub('<.*?>', '', text)
    # Remove non-alphanumeric characters
    text = re.sub('[^a-zA-Z0-9\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove stop words and apply stemming
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    # Join the tokens back into a string
    text = ' '.join(tokens)
    return text

# Apply preprocessing to the 'review' column
data['processed_review'] = data['review'].apply(preprocess_text)

print(data[['review', 'processed_review']].head())
```

**Explanation:**

  * We import necessary modules from NLTK for text preprocessing.
  * We define a function `preprocess_text` that performs the following:
      * Removes HTML tags using regular expressions.
      * Removes non-alphanumeric characters.
      * Converts the text to lowercase.
      * Tokenizes the text into individual words using `nltk.word_tokenize`.
      * Removes stop words (common words like "the," "a," "is" that often don't carry much meaning).
      * Applies stemming (reduces words to their root form, e.g., "running" -\> "run").
      * Joins the processed tokens back into a single string.
  * We apply this function to the 'review' column to create a new column 'processed\_review' containing the cleaned text.

### 3\. Feature Extraction

#### Bag-of-Words (BoW)

```python
from sklearn.feature_extraction.text import CountVectorizer

# Initialize CountVectorizer
vectorizer = CountVectorizer(max_features=5000)  # Limit to 5000 most frequent words

# Fit and transform the processed reviews
X_bow = vectorizer.fit_transform(data['processed_review'])

# Convert to DataFrame for better visualization (optional)
X_bow_df = pd.DataFrame(X_bow.toarray(), columns=vectorizer.get_feature_names_out())
print(X_bow_df.head())
```

#### TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Limit to 5000 most frequent words

# Fit and transform the processed reviews
X_tfidf = tfidf_vectorizer.fit_transform(data['processed_review'])

# Convert to DataFrame for better visualization (optional)
X_tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
print(X_tfidf_df.head())
```

**Explanation:**

  * **Bag-of-Words (BoW):**
      * We initialize a `CountVectorizer` to create a vocabulary of the most frequent words (limited to 5000 in this example).
      * `fit_transform` learns the vocabulary from the processed reviews and then transforms the text data into a matrix where each row represents a review and each column represents a word in the vocabulary. The value in each cell is the count of that word in the review.
  * **TF-IDF:**
      * We initialize a `TfidfVectorizer`, which is similar to `CountVectorizer` but also considers the importance of words across the entire corpus.
      * `fit_transform` calculates the TF-IDF scores for each word in each review.

### 4\. Model Training

#### Naive Bayes

```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Split data into training and testing sets (using TF-IDF features)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, data['sentiment'], test_size=0.2, random_state=42)

# Initialize and train a Naive Bayes classifier
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_nb = nb_model.predict(X_test)

# Evaluate the model
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))
```

#### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

# Initialize and train a Logistic Regression classifier (using TF-IDF features)
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_lr = lr_model.predict(X_test)

# Evaluate the model
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
```

**Explanation:**

  * We split the data (using either BoW or TF-IDF features) into training and testing sets (80% train, 20% test).
  * **Naive Bayes:**
      * We initialize a `MultinomialNB` classifier (suitable for text data).
      * We train the model using the training data.
      * We make predictions on the test data and evaluate the model's accuracy and classification report (precision, recall, F1-score).
  * **Logistic Regression:**
      * We initialize a `LogisticRegression` classifier.
      * We train the model, make predictions, and evaluate it similarly to Naive Bayes.

### 5\. Model Evaluation and Comparison

We've already printed the accuracy and classification reports for each model in the previous step. You can further analyze these results to compare the performance of Naive Bayes and Logistic Regression. You might also consider using other evaluation metrics like:

  * **Confusion Matrix:** To visualize the performance of the classifier in more detail.
  * **ROC Curve and AUC:** To evaluate the trade-off between true positive rate and false positive rate.

<!-- end list -->

```python
from sklearn.metrics import confusion_matrix

# Confusion Matrix for Logistic Regression
cm = confusion_matrix(y_test, y_pred_lr)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()
```

### 6\. Interpretation and Insights

For Logistic Regression, you can examine the coefficients to understand which words contribute most strongly to positive or negative sentiment:

```python
# Get feature names from the TF-IDF vectorizer
feature_names = tfidf_vectorizer.get_feature_names_out()

# Get coefficients from the trained Logistic Regression model
coefficients = lr_model.coef_[0]

# Create a DataFrame to display feature names and their coefficients
feature_importance = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# Sort the DataFrame by coefficient values to see the most important features
feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False)

# Display the top 20 positive and negative features
print("Top 20 Positive Features:\n", feature_importance.head(20))
print("\nTop 20 Negative Features:\n", feature_importance.tail(20))
```

**Explanation:**

  * We retrieve the feature names from the TF-IDF vectorizer.
  * We get the coefficients from the trained Logistic Regression model.
  * We create a DataFrame to store feature names and their corresponding coefficients.
  * We sort the DataFrame by coefficient values.
  * We display the top 20 features with the highest positive coefficients (indicating positive sentiment) and the top 20 features with the lowest negative coefficients (indicating negative sentiment).

## Exercises

1.  **Experiment with Hyperparameters:** Try different values for `max_features` in `CountVectorizer` and `TfidfVectorizer` and see how it affects performance.
2.  **Try Different Preprocessing:** Experiment with lemmatization instead of stemming, or try different stop word lists.
3.  **Use N-grams:** Explore using n-grams (e.g., bigrams, trigrams) in `CountVectorizer` or `TfidfVectorizer` to capture phrases instead of just individual words.
4.  **Cross-Validation:** Use k-fold cross-validation to get a more robust estimate of model performance.
5.  **Advanced Models:** Try more advanced classification models like Support Vector Machines (SVM) or Random Forests.

## Suggested Solutions (Hints)

  * **Exercise 1:** Create a loop that trains and evaluates models with different `max_features` values.
  * **Exercise 2:** NLTK provides lemmatization functionality. You can find different stop word lists online or create your own.
  * **Exercise 3:** Use the `ngram_range` parameter in `CountVectorizer` or `TfidfVectorizer`.
  * **Exercise 4:** Use `cross_val_score` from Scikit-learn.
  * **Exercise 5:** Scikit-learn provides `SVC` for SVM and `RandomForestClassifier` for Random Forests.

## Further Resources

  * **NLTK Book:** [https://www.nltk.org/book/](https://www.google.com/url?sa=E&source=gmail&q=https://www.google.com/url?sa=E%26source=gmail%26q=https://www.nltk.org/book/) (A comprehensive guide to NLP with NLTK)
  * **Scikit-learn Documentation:** [https://scikit-learn.org/stable/](https://www.google.com/url?sa=E&source=gmail&q=https://www.google.com/url?sa=E%26source=gmail%26q=https://scikit-learn.org/stable/)
  * **TextBlob Documentation:** [https://textblob.readthedocs.io/en/dev/](https://www.google.com/url?sa=E&source=gmail&q=https://www.google.com/url?sa=E%26source=gmail%26q=https://textblob.readthedocs.io/en/dev/)
  * **Stanford CS224N: Natural Language Processing with Deep Learning:** [https://web.stanford.edu/class/cs224n/](https://www.google.com/url?sa=E&source=gmail&q=https://www.google.com/url?sa=E%26source=gmail%26q=https://web.stanford.edu/class/cs224n/) (More advanced course)

## Contact

For any queries or feedback related to this module, feel free to open an issue in the main repository or contact us at \[contact@ruslanmv.com].

-----

Happy Learning\!

