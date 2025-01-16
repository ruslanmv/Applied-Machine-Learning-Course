````markdown
# Social Media Analytics: Hashtag Popularity Prediction

Welcome to the Social Media Analytics module of the Applied Machine Learning Course! In this module, we'll focus on predicting the popularity of hashtags on Twitter.

## Motivation

Social media has become a crucial platform for communication, information sharing, marketing, and public opinion formation. Understanding trends and predicting the popularity of hashtags can provide valuable insights for businesses, marketers, researchers, and social media influencers. By analyzing hashtag usage patterns, we can gain insights into public sentiment, identify emerging trends, optimize social media campaigns, and improve content targeting.

## Learning Objectives

By the end of this module, you will be able to:

*   Understand the importance of social media analytics and hashtag popularity prediction.
*   Collect Twitter data using the Tweepy API.
*   Perform text preprocessing and feature engineering on tweet data.
*   Train regression models to predict hashtag popularity (e.g., number of retweets or favorites).
*   Evaluate the performance of regression models using appropriate metrics.
*   Use Python libraries like Pandas, Tweepy, Scikit-learn, and Matplotlib for social media data analysis.

## Real-World Applications

*   **Marketing and Advertising:** Optimize social media marketing campaigns by identifying trending hashtags and predicting their reach.
*   **Brand Monitoring:** Track brand mentions and sentiment on social media.
*   **Trend Analysis:** Detect emerging trends and topics of public interest.
*   **Event Promotion:** Predict the popularity of event-related hashtags to maximize visibility.
*   **Social Media Influencers:** Identify relevant hashtags to increase the reach and engagement of their posts.
*   **Public Opinion Research:** Analyze public opinion on specific topics or events.

## Conceptual Overview

Predicting hashtag popularity is typically framed as a **regression** problem. We aim to predict a continuous target variable, such as the number of retweets, favorites, or overall engagement, based on various features extracted from the tweet and the hashtag itself.

**Key Techniques:**

*   **Text Analysis:** Processing and analyzing the text content of tweets to extract relevant information.
*   **Feature Engineering:** Creating features from the raw tweet data that can be used to predict hashtag popularity.
*   **Regression Models:** Using regression algorithms like Linear Regression, Ridge Regression, Lasso Regression, or others to model the relationship between the features and the target variable.

## Tools

*   **Python:** Our primary programming language.
*   **Pandas:** For data manipulation and analysis.
*   **Tweepy:** A Python library for accessing the Twitter API.
*   **Scikit-learn:** For building and evaluating machine learning models, including regression models.
*   **Matplotlib:** For data visualization.
*   **Seaborn:** For statistical data visualization.
*   **NLTK (optional):**  For advanced text preprocessing like stemming or lemmatization
*   **Regular Expressions (re):** For text cleaning and pattern matching.

## Dataset

We will collect data from Twitter using the **Tweepy** library to access the **Twitter API**.

*   **Data Collection:** We will define search queries based on specific hashtags or keywords to collect relevant tweets.
*   **Data Attributes:** The collected data will include attributes like:
    *   `created_at`: Timestamp of the tweet.
    *   `text`: The text content of the tweet.
    *   `retweet_count`: Number of retweets.
    *   `favorite_count`: Number of favorites (likes).
    *   `user`: Information about the user who posted the tweet (e.g., follower count, verified status).
    *   `entities`: Information about hashtags, user mentions, URLs in the tweet.
*   **Ethical Considerations:** We must adhere to Twitter's API usage guidelines and respect user privacy when collecting and analyzing Twitter data.

## Project Roadmap

1.  **Twitter API Setup:** Obtain API credentials and set up Tweepy for data collection.
2.  **Data Collection:** Use Tweepy to collect tweets based on specific hashtags or keywords.
3.  **Data Exploration and Preprocessing:** Explore the collected data, clean the text, and handle missing values.
4.  **Feature Engineering:** Create features that might be predictive of hashtag popularity.
5.  **Model Training:** Train regression models to predict hashtag popularity (e.g., retweet count).
6.  **Model Evaluation:** Evaluate the models' performance using metrics like MAE, MSE, RMSE, and R-squared.
7.  **Interpretation and Insights:** Analyze the model's results to understand the factors that influence hashtag popularity.

## Step-by-Step Instructions

### 1. Twitter API Setup

**Note:** You need to have a Twitter Developer Account and create a Twitter App to get API credentials.

1.  **Apply for a Twitter Developer Account:** If you don't have one, apply for a developer account at [https://developer.twitter.com/en/apply-for-access](https://www.google.com/url?sa=E&source=gmail&q=https://developer.twitter.com/en/apply-for-access).
2.  **Create a Twitter App:** Once your developer account is approved, create a new app at [https://developer.twitter.com/en/apps](https://www.google.com/url?sa=E&source=gmail&q=https://developer.twitter.com/en/apps).
3.  **Obtain API Credentials:** After creating your app, you'll get the following credentials:
    *   API Key (Consumer Key)
    *   API Secret Key (Consumer Secret)
    *   Access Token
    *   Access Token Secret

**Install Tweepy:**

```bash
pip install tweepy
```

### 2. Data Collection

```python
import tweepy
import pandas as pd

# Replace with your Twitter API credentials
consumer_key = "YOUR_CONSUMER_KEY"
consumer_secret = "YOUR_CONSUMER_SECRET"
access_token = "YOUR_ACCESS_TOKEN"
access_token_secret = "YOUR_ACCESS_TOKEN_SECRET"

# Authenticate with Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

# Define the hashtag to search for
hashtag = "#machinelearning"

# Define the number of tweets to collect
num_tweets = 1000

# Collect tweets
tweets = tweepy.Cursor(api.search_tweets, q=hashtag, lang="en", tweet_mode='extended').items(num_tweets)

# Create a list to store tweet data
tweet_data = []
for tweet in tweets:
    tweet_data.append([tweet.created_at, tweet.full_text, tweet.retweet_count, tweet.favorite_count,
                       tweet.user.followers_count, tweet.user.verified, tweet.entities['hashtags']])

# Create a Pandas DataFrame
columns = ['created_at', 'text', 'retweet_count', 'favorite_count', 'user_followers_count', 'user_verified', 'hashtags']
df = pd.DataFrame(tweet_data, columns=columns)

print(df.head())

# Save the data to a CSV file (optional)
df.to_csv(f"{hashtag}_tweets.csv", index=False)
```

**Explanation:**

*   We import the necessary libraries (Tweepy and Pandas).
*   **Replace Placeholders:** You must replace the placeholder values for `consumer_key`, `consumer_secret`, `access_token`, and `access_token_secret` with your actual Twitter API credentials.
*   We authenticate with the Twitter API using Tweepy's `OAuthHandler`.
*   We define the hashtag we want to search for and the number of tweets to collect.
*   We use `tweepy.Cursor` to search for tweets containing the hashtag.
    *   `q=hashtag`: The search query.
    *   `lang="en"`: Filter tweets in English.
    *   `tweet_mode='extended'`: To get the full text of the tweet (important for tweets longer than 140 characters).
    *   `items(num_tweets)`: Limits the number of tweets to collect.
*   We iterate through the collected tweets and extract relevant information (created\_at, text, retweet\_count, favorite\_count, user information, hashtags).
*   We create a Pandas DataFrame to store the collected data.
*   We print the first few rows of the DataFrame to check the data.
*   (Optional) We save the DataFrame to a CSV file for later use.

### 3. Data Exploration and Preprocessing

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Load the data (if not already loaded)
# df = pd.read_csv("your_hashtag_tweets.csv")

# Explore the data
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Visualize the distribution of retweet counts
plt.figure(figsize=(10, 6))
sns.histplot(df['retweet_count'], bins=50)
plt.xlabel("Retweet Count")
plt.ylabel("Frequency")
plt.title("Distribution of Retweet Counts")
plt.show()

# Text Preprocessing
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
    text = re.sub(r'\d+', '', text) # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

print(df[['text', 'cleaned_text']].head())
```

**Explanation:**

*   We load the data from the CSV file (if it was saved in the previous step).
*   We use `df.info()` and `df.describe()` to get a summary of the data, including data types, non-null counts, and descriptive statistics.
*   We check for missing values using `df.isnull().sum()`.
*   We visualize the distribution of retweet counts (our target variable for this example) using a histogram.
*   **Text Preprocessing:**
    *   We define a function `clean_text` to perform basic text cleaning:
        *   Convert text to lowercase.
        *   Remove URLs.
        *   Remove mentions (starting with @) and hashtags (starting with #).
        *   Remove punctuation.
    *   We apply the `clean_text` function to the 'text' column to create a new 'cleaned\_text' column.

### 4. Feature Engineering

```python
# Feature Engineering

# 1. Number of Hashtags
df['num_hashtags'] = df['text'].apply(lambda x: len([c for c in x if c == '#']))

# 2. Number of Mentions
df['num_mentions'] = df['text'].apply(lambda x: len([c for c in x if c == '@']))

# 3. Number of URLs
df['num_urls'] = df['text'].apply(lambda x: len(re.findall(r"http\S+|www\S+|https\S+", x)))

# 4. Text Length
df['text_length'] = df['text'].apply(len)

# 5. Average Word Length
def avg_word_length(text):
    words = text.split()
    if len(words) == 0:
        return 0
    return sum(len(word) for word in words) / len(words)

df['avg_word_length'] = df['cleaned_text'].apply(avg_word_length)

# 6. Day of the Week
df['created_at'] = pd.to_datetime(df['created_at'])
df['day_of_week'] = df['created_at'].dt.dayofweek

# 7. Hour of the Day
df['hour_of_day'] = df['created_at'].dt.hour

print(df.head())
```

**Explanation:**

We create several new features based on the tweet data:

*   **`num_hashtags`:** The number of hashtags in the tweet.
*   **`num_mentions`:** The number of mentions (@) in the tweet.
*   **`num_urls`:** The number of URLs in the tweet.
*   **`text_length`:** The length (number of characters) of the tweet.
*   **`avg_word_length`:** The average word length in the cleaned text.
*   **`day_of_week`:** The day of the week the tweet was posted (0 = Monday, 6 = Sunday).
*   **`hour_of_day`:** The hour of the day the tweet was posted.

### 5. Model Training

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Define features (X) and target (y)
features = ['num_hashtags', 'num_mentions', 'num_urls', 'text_length', 'avg_word_length',
            'day_of_week', 'hour_of_day', 'user_followers_count', 'user_verified']
target = 'retweet_count'  # You can also try 'favorite_count' or a combination

X = df[features]
y = df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with scaling and linear regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

# Train the model
pipeline.fit(X_train, y_train)
```

**Explanation:**

*   We select the features we created and the target variable (`retweet_count` in this example).
*   We split the data into training and testing sets (80% train, 20% test).
*   We create a pipeline that first scales the features using `StandardScaler` and then trains a `LinearRegression` model.
*   We train the model using `pipeline.fit(X_train, y_train)`.

### 6. Model Evaluation

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Visualize actual vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Retweet Count")
plt.ylabel("Predicted Retweet Count")
plt.title("Actual vs. Predicted Retweet Count")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')  # Diagonal line
plt.show()
```

**Explanation:**

*   We make predictions on the test set using `pipeline.predict(X_test)`.
*   We evaluate the model's performance using the following metrics:
    *   **Mean Absolute Error (MAE):** The average absolute difference between the predicted and actual values.
    *   **Mean Squared Error (MSE):** The average squared difference between the predicted and actual values.
    *   **Root Mean Squared Error (RMSE):** The square root of MSE, in the same units as the target variable.
    *   **R-squared (R2):** A measure of how well the model fits the data (proportion of variance explained).
*   We visualize the actual vs. predicted retweet counts using a scatter plot.

### 7. Interpretation and Insights

```python
# Get feature coefficients from the linear regression model
coefficients = pd.DataFrame({'feature': features, 'coefficient': pipeline.named_steps['model'].coef_})
coefficients = coefficients.sort_values('coefficient', ascending=False)
print(coefficients)
```

**Explanation:**

*   We extract the coefficients from the trained `LinearRegression` model.
*   We create a Pandas DataFrame to display the features and their corresponding coefficients.
*   We sort the coefficients in descending order to see which features have the strongest positive or negative influence on the predicted retweet count.

**Interpreting Coefficients (Linear Regression):**

*   **Magnitude:** The larger the absolute value of a coefficient, the stronger its influence on the prediction.
*   **Sign:**
    *   **Positive coefficient:** Indicates that an increase in the feature's value leads to an increase in the predicted retweet count.
    *   **Negative coefficient:** Indicates that an increase in the feature's value leads to a decrease in the predicted retweet count.

## Exercises

1.  **Different Regression Models:** Try other regression models from Scikit-learn, such as Ridge Regression, Lasso Regression, Decision Tree Regression, or Random Forest Regression.
2.  **Hyperparameter Tuning:** Use `GridSearchCV` or `RandomizedSearchCV` to tune the hyperparameters of your chosen regression model.
3.  **Advanced Text Preprocessing:** Use NLTK for more advanced text preprocessing, such as stemming, lemmatization, or part-