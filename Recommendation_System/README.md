````markdown
# Recommendation Systems

Welcome to the Recommendation Systems module of the Applied Machine Learning Course!

## Motivation

Recommendation systems are ubiquitous in today's digital world. They power the personalized experiences we encounter on e-commerce websites, streaming services, social media platforms, and more. By suggesting relevant items to users, recommendation systems can enhance user engagement, increase sales, and improve customer satisfaction.

## Learning Objectives

By the end of this module, you will be able to:

*   Understand the fundamental concepts of recommendation systems.
*   Differentiate between collaborative filtering and content-based filtering.
*   Implement user-based and item-based collaborative filtering algorithms.
*   Build and evaluate recommendation systems using the Surprise library in Python.
*   Understand the strengths and weaknesses of different recommendation approaches.

## Real-World Applications

*   **E-commerce:** Recommending products to customers based on their browsing and purchase history (e.g., Amazon's "Customers who bought this also bought").
*   **Streaming Services:** Suggesting movies, TV shows, or music based on user preferences (e.g., Netflix's movie recommendations, Spotify's personalized playlists).
*   **Social Networks:** Recommending friends, groups, or content to users (e.g., Facebook's friend suggestions, LinkedIn's job recommendations).
*   **News and Content Aggregators:** Personalizing news feeds and suggesting articles based on user interests (e.g., Google News).
*   **Online Advertising:** Targeting ads to users based on their predicted interests.

## Conceptual Overview

Recommendation systems typically employ one of two main approaches:

1.  **Collaborative Filtering:** This approach leverages the preferences of similar users or the ratings of similar items to make recommendations.
    *   **User-based Collaborative Filtering:** Recommends items based on the preferences of users who are similar to the target user.
    *   **Item-based Collaborative Filtering:** Recommends items that are similar to items the target user has liked or interacted with in the past.
2.  **Content-based Filtering:** This approach recommends items based on their similarity to items the user has liked in the past, using item features or descriptions.

In this module, we will primarily focus on **collaborative filtering** techniques.

**Key Concepts:**

*   **User-Item Matrix:** A matrix representation of user preferences, where rows represent users, columns represent items, and cells contain ratings or interactions (e.g., 1 if the user liked the item, 0 otherwise).
*   **Similarity Measures:** Used to quantify the similarity between users or items (e.g., cosine similarity, Pearson correlation).
*   **Neighborhood:** A set of users or items that are most similar to a target user or item.
*   **Prediction:** Estimating the rating a user would give to an item they haven't interacted with yet.

## Tools

*   **Python:** Our primary programming language.
*   **Surprise:** A Python scikit specifically designed for building and evaluating recommendation systems.
*   **Scikit-learn:** For general machine learning tasks, such as data preprocessing and similarity calculations.
*   **Pandas:** For data manipulation and analysis.
*   **NumPy:** For numerical operations.
*   **Matplotlib:** For data visualization.

## Dataset

We will use the **MovieLens 100K dataset**, a widely used benchmark dataset for recommendation systems.

*   **Source:** [MovieLens Website](https://grouplens.org/datasets/movielens/)
*   **Description:** The dataset contains 100,000 ratings (1-5 stars) from 943 users on 1682 movies.
*   **Files:**
    *   `u.data`: The main dataset with user-item ratings.
    *   `u.item`: Information about the movies (e.g., title, genre).
    *   `u.user`: Demographic information about the users (e.g., age, gender, occupation).
*   **Potential Limitations:** The dataset is relatively small and might not fully represent the complexity of real-world recommendation scenarios. Also, user preferences and movie popularity can change over time.

## Project Roadmap

1.  **Data Loading and Exploration:** Load the MovieLens dataset, understand its structure, and perform exploratory data analysis (EDA).
2.  **Data Preprocessing:** Prepare the data for use with the Surprise library.
3.  **Model Building:** Build recommendation models using different collaborative filtering algorithms (e.g., KNN, SVD).
4.  **Model Evaluation:** Evaluate the models' performance using metrics like RMSE and MAE.
5.  **Hyperparameter Tuning:** Optimize the models' performance by tuning their hyperparameters.
6.  **Making Recommendations:** Use the trained models to generate recommendations for users.

## Step-by-Step Instructions

### 1. Data Loading and Exploration

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the MovieLens dataset
data = pd.read_csv('u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
item_info = pd.read_csv('u.item', sep='|', encoding='latin-1', header=None)
item_info = item_info[[0, 1]]
item_info.columns = ['item_id', 'movie_title']
user_info = pd.read_csv('u.user', sep='|', header=None)
user_info = user_info[[0, 1]]
user_info.columns = ['user_id', 'age']

# Merge dataframes
data = pd.merge(data, item_info, on='item_id')
data = pd.merge(data, user_info, on='user_id')

# Display the first few rows
print(data.head())

# Get basic information about the dataset
print(data.info())

# Explore the distribution of ratings
plt.figure(figsize=(8, 6))
sns.countplot(x='rating', data=data)
plt.title('Distribution of Ratings')
plt.show()

# Explore the number of ratings per movie
plt.figure(figsize=(8, 6))
data.groupby('movie_title')['rating'].count().sort_values(ascending=False).hist(bins=50)
plt.title('Number of Ratings per Movie')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Movies')
plt.show()
```

**Explanation:**

  * We import the necessary libraries (Pandas, Matplotlib, Seaborn).
  * We load the `u.data`, `u.item`, and `u.user` files into Pandas DataFrames.
  * We merge dataframes to get movie title and user age in our main dataframe.
  * We examine the first few rows and data types.
  * We visualize the distribution of ratings and the number of ratings per movie.

### 2. Data Preprocessing

```python
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split

# Define the format of the data for Surprise
reader = Reader(rating_scale=(1, 5))

# Load the data into a Surprise Dataset object
dataset = Dataset.load_from_df(data[['user_id', 'item_id', 'rating']], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(dataset, test_size=0.25, random_state=42)
```

**Explanation:**

  * We import the necessary classes from the Surprise library.
  * We create a `Reader` object to specify the rating scale (1 to 5).
  * We load the data from the Pandas DataFrame into a Surprise `Dataset` object.
  * We split the data into training and testing sets using `train_test_split()`.

### 3. Model Building

#### KNN (k-Nearest Neighbors)

```python
from surprise import KNNBasic

# Initialize the KNN model (user-based collaborative filtering)
sim_options = {
    'name': 'cosine',
    'user_based': True  # Compute  similarities between users
}
knn_model = KNNBasic(sim_options=sim_options)

# Train the model
knn_model.fit(trainset)
```

#### SVD (Singular Value Decomposition)

```python
from surprise import SVD

# Initialize the SVD model
svd_model = SVD()

# Train the model
svd_model.fit(trainset)
```

**Explanation:**

  * **KNN:**
      * We initialize a `KNNBasic` model, which implements a basic k-Nearest Neighbors collaborative filtering algorithm.
      * We set `user_based=True` for user-based collaborative filtering. You can set it to `False` for item-based.
      * We use cosine similarity as the similarity measure.
      * We train the model using `knn_model.fit(trainset)`.
  * **SVD:**
      * We initialize an `SVD` model, which implements a matrix factorization algorithm based on Singular Value Decomposition.
      * We train the model using `svd_model.fit(trainset)`.

### 4. Model Evaluation

```python
from surprise import accuracy

# Make predictions on the test set
knn_predictions = knn_model.test(testset)
svd_predictions = svd_model.test(testset)

# Evaluate the models using RMSE and MAE
print("KNN RMSE:", accuracy.rmse(knn_predictions))
print("KNN MAE:", accuracy.mae(knn_predictions))

print("SVD RMSE:", accuracy.rmse(svd_predictions))
print("SVD MAE:", accuracy.mae(svd_predictions))
```

**Explanation:**

  * We make predictions on the test set using `model.test()`.
  * We evaluate the models using RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error). Lower values are better for both metrics.

### 5. Hyperparameter Tuning

#### KNN

```python
from surprise.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'k': [20, 40, 60],
    'sim_options': {
        'name': ['msd', 'cosine', 'pearson'],
        'user_based': [True, False]
    }
}

# Perform Grid Search with Cross-Validation
gs_knn = GridSearchCV(KNNBasic, param_grid, measures=['rmse', 'mae'], cv=3)
gs_knn.fit(dataset)

# Print the best parameters and best score
print("Best KNN RMSE:", gs_knn.best_score['rmse'])
print("Best KNN parameters:", gs_knn.best_params['rmse'])
```

#### SVD

```python
# Define the parameter grid
param_grid = {
    'n_factors': [50, 100, 150],
    'n_epochs': [10, 20, 30],
    'lr_all': [0.002, 0.005, 0.01],
    'reg_all': [0.02, 0.04, 0.06]
}

# Perform Grid Search with Cross-Validation
gs_svd = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
gs_svd.fit(dataset)

# Print the best parameters and best score
print("Best SVD RMSE:", gs_svd.best_score['rmse'])
print("Best SVD parameters:", gs_svd.best_params['rmse'])
```

**Explanation:**

  * We use `GridSearchCV` to find the best hyperparameter combination for each model.
  * We define a `param_grid` with different values for the hyperparameters we want to tune.
  * `GridSearchCV` trains and evaluates the model with all possible combinations of hyperparameters in the `param_grid` using cross-validation.
  * We print the best RMSE score and the corresponding hyperparameter combination.

### 6. Making Recommendations

```python
# Get the best SVD model from the grid search
best_svd_model = gs_svd.best_estimator['rmse']

# Retrain the best model on the full dataset
trainset = dataset.build_full_trainset()
best_svd_model.fit(trainset)

# Example: Get top 10 movie recommendations for user with user_id = 1
user_id = '1'
user_movies = data[data['user_id'] == int(user_id)]['item_id'].tolist()
all_movies = data['item_id'].unique()
unseen_movies = [movie for movie in all_movies if movie not in user_movies]

predictions = [best_svd_model.predict(user_id, movie) for movie in unseen_movies]
predictions.sort(key=lambda x: x.est, reverse=True) # Sort by estimated rating

top_10_recommendations = [pred.iid for pred in predictions[:10]]
top_10_movie_titles = [item_info[item_info['item_id'] == int(movie_id)]['movie_title'].iloc[0] for movie_id in top_10_recommendations]

print(f"Top 10 movie recommendations for user {user_id}:")
for movie_title in top_10_movie_titles:
    print(movie_title)
```

**Explanation:**

  * We retrieve the best SVD model from the `GridSearchCV` results.
  * We retrain the best model on the entire dataset using `build_full_trainset()`.
  * To generate recommendations for a specific user:
      * We get a list of movies the user has already rated.
      * We get a list of all movies in the dataset.
      * We create a list of movies the user has *not* seen yet.
      * We use the trained model to predict the user's rating for each unseen movie.
      * We sort the predictions by estimated rating in descending order.
      * We get the top 10 movie IDs and convert them to movie titles.
      * We print the top 10 movie recommendations.

## Exercises

1.  **Item-based Collaborative Filtering:** Implement item-based collaborative filtering using `KNNBasic` and compare its performance to user-based collaborative filtering.
2.  **Different Similarity Measures:** Experiment with different similarity measures (e.g., Pearson correlation) in `KNNBasic`.
3.  **Advanced Matrix Factorization:** Explore more advanced matrix factorization techniques like SVD++ or NMF (Non-negative Matrix Factorization) available in the Surprise library.
4.  **Hybrid Recommender:** Combine collaborative filtering and content-based filtering to create a hybrid recommendation system. You could use movie genres from the `u.item` file for content-based filtering.

## Suggested Solutions (Hints)

  * **Exercise 1:** Set `user_based=False` in `KNNBasic`.
  * **Exercise 2:**  Refer to the Surprise documentation for available similarity measures.
  * **Exercise 3:** Check the Surprise documentation for `SVDpp` and `NMF` classes.
  * **Exercise 4:** You could train a content-based model to predict ratings based on movie genres and then combine its predictions with the predictions from a collaborative filtering model (e.g., by averaging the predicted ratings).

## Further Resources

  * **Surprise Documentation:** [https://surprise.readthedocs.io/en/stable/](https://www.google.com/url?sa=E&source=gmail&q=https://surprise.readthedocs.io/en/stable/)
  * **Recommender Systems Handbook:** [https://www.amazon.com/Recommender-Systems-Handbook-Francesco-Ricci/dp/1489976376](https://www.google.com/url?sa=E&source=gmail&q=https://www.amazon.com/Recommender-Systems-Handbook-Francesco-Ricci/dp/1489976376)
  * **MovieLens Dataset Documentation:** [https://grouplens.org/datasets/movielens/](https://www.google.com/url?sa=E&source=gmail&q=https://grouplens.org/datasets/movielens/)

## Contact

For any queries or feedback related to this module, feel free to open an issue in the main repository or contact us at \[contact@ruslanmv.com].

-----

Happy Learning\!
