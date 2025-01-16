# Spam Email Detection

Welcome to the Spam Email Detection module of the Applied Machine Learning Course!

## Motivation

Email spam, also known as junk email, is a significant problem for individuals and organizations. It clutters inboxes, wastes time, and can even pose security risks. Spam filters are essential tools for automatically identifying and filtering out unwanted emails. Machine learning, particularly Natural Language Processing (NLP) techniques, plays a crucial role in building effective spam detection systems.

## Learning Objectives

By the end of this module, you will be able to:

*   Understand the problem of spam email detection and its importance.
*   Perform text preprocessing techniques for NLP, such as tokenization, stop word removal, and stemming/lemmatization.
*   Use feature extraction methods like CountVectorizer (Bag-of-Words) and TF-IDF to represent text data numerically.
*   Train and evaluate a Naive Bayes classifier for spam detection.
*   Use Python libraries like NLTK and Scikit-learn for NLP and machine learning tasks.
*   Interpret the results of the spam detection model and understand the factors that contribute to spam classification.

## Real-World Applications

*   **Email Providers:**  Filtering spam emails for users (e.g., Gmail, Outlook).
*   **Security Systems:** Detecting phishing emails and other malicious content.
*   **Marketing:** Identifying and filtering out spam messages in marketing campaigns.
*   **Content Moderation:** Detecting and removing spam comments on forums and social media platforms.

## Conceptual Overview

Spam email detection is typically framed as a **binary classification** problem. We aim to classify emails as either "spam" (positive class) or "ham" (negative class, meaning not spam).

We will use the following techniques:

1.  **Natural Language Processing (NLP):** To process and analyze the text content of emails.
2.  **Feature Extraction:** To convert text data into numerical features that can be used by machine learning models. We'll use:
    *   **CountVectorizer (Bag-of-Words):** Represents text as a collection of individual words and their frequencies.
    *   **TF-IDF (Term Frequency-Inverse Document Frequency):** Weights words based on their importance in a document and across the entire corpus.
3.  **Naive Bayes Classifier:** A probabilistic classifier based on Bayes' theorem with a "naive" assumption of independence between features (words in this case). It's a simple yet effective method for text classification.

## Tools

*   **Python:** Our primary programming language.
*   **NLTK (Natural Language Toolkit):** A comprehensive library for NLP tasks.
*   **Scikit-learn:** For building and evaluating machine learning models, including Naive Bayes and feature extraction.
*   **Pandas:** For data manipulation and analysis.
*   **Matplotlib:** For data visualization.

## Dataset

We will use a publicly available dataset of spam and ham emails.

*   **Source:** You can find various spam datasets online, such as the [SpamAssassin public corpus](https://spamassassin.apache.org/old/publiccorpus/) or the [Enron-Spam dataset](http://www2.aueb.gr/users/ion/data/enron-spam/). For this example, we'll use a dataset that's often included in machine learning tutorials, commonly referred to as the "SMS Spam Collection Dataset". You can find it on the UCI Machine Learning Repository: [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
*   **Description:** The dataset contains SMS messages labeled as either "spam" or "ham."
*   **Features:** Each message is represented as raw text. The target variable is the label "spam" or "ham."
*   **Potential Limitations:** SMS messages are typically shorter than emails, so the vocabulary and patterns might be slightly different. However, the general principles of spam detection apply to both SMS and email.

## Project Roadmap

1.  **Data Loading and Exploration:** Load the dataset, understand its structure, and perform exploratory data analysis (EDA).
2.  **Data Preprocessing:** Clean the text data, perform tokenization, remove stop words, and apply stemming or lemmatization.
3.  **Feature Extraction:** Convert the text data into numerical representations using CountVectorizer and TF-IDF.
4.  **Model Training:** Train a Multinomial Naive Bayes classifier to classify spam emails.
5.  **Model Evaluation:** Evaluate the model's performance using appropriate metrics (accuracy, precision, recall, F1-score).
6.  **Interpretation and Insights:** Analyze the model's results to identify words or patterns that are strong indicators of spam.

## Step-by-Step Instructions

### 1. Data Loading and Exploration

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("spam.csv", encoding='latin-1')

# Drop unnecessary columns and rename existing ones
data = data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
data = data.rename(columns={'v1': 'label', 'v2': 'text'})

# Display the first few rows
print(data.head())

# Get basic information about the dataset
print(data.info())

# Check for missing values
print(data.isnull().sum())

# Visualize the distribution of the target variable (label)
sns.countplot(x='label', data=data)
plt.title('Distribution of Spam and Ham')
plt.show()

# Explore the length of messages
data['message_length'] = data['text'].apply(len)
plt.figure(figsize=(8, 6))
sns.histplot(data=data, x='message_length', hue='label', bins=50)
plt.title('Message Length Distribution')
plt.show()
```

**Explanation:**

  * We import the necessary libraries (Pandas, Matplotlib, Seaborn).
  * We load the dataset using `pd.read_csv()`. We specify `encoding='latin-1'` because the dataset might contain special characters.
  * We drop unnecessary columns ('Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4') and rename the columns 'v1' and 'v2' to 'label' and 'text'.
  * We examine the first few rows, data types, and check for missing values.
  * We visualize the distribution of the target variable ('label') using a countplot.
  * We create a new feature 'message_length' and plot its distribution for both spam and ham messages.

### 2. Data Preprocessing

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
    # Remove punctuation and convert to lowercase
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove stop words and apply stemming
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    # Join the tokens back into a string
    text = ' '.join(tokens)
    return text

# Apply preprocessing to the 'text' column
data['processed_text'] = data['text'].apply(preprocess_text)

print(data[['text', 'processed_text']].head())
```

**Explanation:**

  * We import necessary modules from NLTK for text preprocessing.
  * We define a function `preprocess_text` that performs the following:
      * Removes punctuation and converts the text to lowercase using regular expressions.
      * Tokenizes the text into individual words using `nltk.word_tokenize`.
      * Removes stop words (common words like "the," "a," "is" that often don't carry much meaning for spam classification).
      * Applies stemming (reduces words to their root form, e.g., "running" -\> "run").
      * Joins the processed tokens back into a single string.
  * We apply this function to the 'text' column to create a new column 'processed\_text' containing the cleaned text.

### 3. Feature Extraction

#### CountVectorizer (Bag-of-Words)

```python
from sklearn.feature_extraction.text import CountVectorizer

# Initialize CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the processed text data
X_bow = vectorizer.fit_transform(data['processed_text'])

# Convert to DataFrame for better visualization (optional)
X_bow_df = pd.DataFrame(X_bow.toarray(), columns=vectorizer.get_feature_names_out())
print(X_bow_df.head())
```

#### TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the processed text data
X_tfidf = tfidf_vectorizer.fit_transform(data['processed_text'])

# Convert to DataFrame for better visualization (optional)
X_tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
print(X_tfidf_df.head())
```

**Explanation:**

  * **CountVectorizer (Bag-of-Words):**
      * We initialize a `CountVectorizer` to create a vocabulary of all unique words in the processed text data.
      * `fit_transform` learns the vocabulary and then transforms the text data into a matrix where each row represents an email and each column represents a word in the vocabulary. The value in each cell is the count of that word in the email.
  * **TF-IDF:**
      * We initialize a `TfidfVectorizer`, which is similar to `CountVectorizer` but also considers the importance of words across the entire corpus.
      * `fit_transform` calculates the TF-IDF scores for each word in each email.

### 4. Model Training

```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Split data into training and testing sets (using TF-IDF features)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, data['label'], test_size=0.2, random_state=42)

# Initialize and train a Multinomial Naive Bayes classifier
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = nb_model.predict(X_test)
```

**Explanation:**

  * We split the data (using either BoW or TF-IDF features) into training and testing sets (80% train, 20% test).
  * We initialize a `MultinomialNB` classifier, which is suitable for text classification with discrete features (like word counts or TF-IDF scores).
  * We train the model using the training data (`X_train`, `y_train`).
  * We make predictions on the test data (`X_test`).

### 5. Model Evaluation

```python
# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

**Explanation:**

  * We evaluate the model's performance using:
      * **Accuracy:** The overall percentage of correct predictions.
      * **Classification Report:** Provides precision, recall, F1-score, and support for each class (spam and ham).
      * **Confusion Matrix:** A table showing the counts of true positives, true negatives, false positives, and false negatives.
  * We plot the confusion matrix as a heatmap using Seaborn for better visualization.

### 6. Interpretation and Insights

**Note:** Interpreting the model directly in terms of specific words is a bit more complex with Naive Bayes compared to, say, Logistic Regression where you have coefficients for each feature. However, we can still get some insights by looking at the words that are most frequently associated with spam and ham emails.

```python
# Get feature names from the TF-IDF vectorizer
feature_names = tfidf_vectorizer.get_feature_names_out()

# Get the top N words most associated with spam and ham (example)
def get_top_words(model, feature_names, n=10):
    # Get the log probabilities for each class
    log_probs_spam = model.feature_log_prob_[1] # Assuming 'spam' is class 1
    log_probs_ham = model.feature_log_prob_[0] # Assuming 'ham' is class 0

    # Create DataFrames for easier sorting
    spam_words = pd.DataFrame({'word': feature_names, 'log_prob': log_probs_spam}).sort_values('log_prob', ascending=False)
    ham_words = pd.DataFrame({'word': feature_names, 'log_prob': log_probs_ham}).sort_values('log_prob', ascending=False)

    print(f"Top {n} words associated with spam:")
    print(spam_words.head(n))

    print(f"\nTop {n} words associated with ham:")
    print(ham_words.head(n))

get_top_words(nb_model, feature_names)
```

**Explanation:**

  * We get the feature names from the `TfidfVectorizer`.
  * We define a function `get_top_words` that does the following:
      * Gets the log probabilities of each word for the spam and ham classes from the trained Naive Bayes model (`model.feature_log_prob_`).
      * Creates Pandas DataFrames to store words and their log probabilities for each class.
      * Sorts the DataFrames by log probability in descending order.
      * Prints the top N words associated with spam and ham.

## Exercises

1.  **Hyperparameter Tuning:** Experiment with different values for the `alpha` parameter in the `MultinomialNB` model (smoothing parameter). You can use `GridSearchCV` or `RandomizedSearchCV` to automate this process.
2.  **Different Feature Extraction:** Try using only `CountVectorizer` or only `TfidfVectorizer`. Compare the performance of the models with these different feature sets.
3.  **Lemmatization:** Instead of stemming, use lemmatization in the `preprocess_text` function. You'll need to use `WordNetLemmatizer` from NLTK.
4.  **N-grams:** Use n-grams (e.g., bigrams, trigrams) in `CountVectorizer` or `TfidfVectorizer` to capture phrases instead of just individual words.
5.  **Other Classifiers:** Explore other classification algorithms like Logistic Regression, Support Vector Machines (SVM), or Random Forests for spam detection.

## Suggested Solutions (Hints)

  * **Exercise 1:** Refer to the Scikit-learn documentation for `MultinomialNB` and `GridSearchCV`.
  * **Exercise 2:** Simply replace the feature extraction method in the code with either `CountVectorizer` or `TfidfVectorizer` and compare the results.
  * **Exercise 3:**  Import `WordNetLemmatizer` from NLTK and use its `lemmatize()` method in your `preprocess_text` function.
  * **Exercise 4:** Use the `ngram_range` parameter in `CountVectorizer` or `TfidfVectorizer`.
  * **Exercise 5:** Scikit-learn provides implementations of various classification algorithms.

## Further Resources

  * **NLTK Book:** [https://www.nltk.org/book/](https://www.google.com/url?sa=E&source=gmail&q=https://www.google.com/url?sa=E%26source=gmail%26q=https://www.nltk.org/book/) (A comprehensive guide to NLP with NLTK)
  * **Scikit-learn Documentation:** [https://scikit-learn.org/stable/](https://www.google.com/url?sa=E&source=gmail&q=https://www.google.com/url?sa=E%26source=gmail%26q=https://scikit-learn.org/stable/)
  * **SpamAssassin Public Corpus:** [https://spamassassin.apache.org/old/publiccorpus/](https://www.google.com/url?sa=E&source=gmail&q=https://www.google.com/url?sa=E%26source=gmail%26q=https://spamassassin.apache.org/old/publiccorpus/)
  * **Enron-Spam Dataset:** [http://www2.aueb.gr/users/ion/data/enron-spam/](https://www.google.com/url?sa=E&source=gmail&q=https://www.google.com/url?sa=E%26source=gmail%26q=http://www2.aueb.gr/users/ion/data/enron-spam/)

## Contact

For any queries or feedback related to this module, feel free to open an issue in the main repository or contact us at \[contact@ruslanmv.com].

-----

Happy Learning\!

````

