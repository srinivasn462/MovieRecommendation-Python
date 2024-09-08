# MovieRecommendation-Python


The code uses Python syntax and libraries like pandas for data manipulation and numpy for numerical operations.
It involves basic Python operations like variable assignment, function calls, and control flow statements.
Data Analysis:

The code loads and processes data from files using pandas.
It performs data aggregation and filtering operations, such as calculating mean ratings, counting occurrences, and filtering by specific criteria.

It uses only python, Machine learning not yet implemented. we can implement Collaborative Filtering

Code Example:
import pandas as pd
from scipy.spatial.distance import cosine

# Load data
ratings = pd.read_csv('ratings.csv')

# Create user-item matrix
user_item_matrix = ratings.pivot_table(index='user_id', columns='item_id', values='rating')

# Calculate user similarities
user_similarities = 1 - cosine(user_item_matrix.fillna(0).values)

# Recommend items to a user
def recommend(user_id, top_n=10):
    similar_users = user_similarities[user_id].argsort()[::-1][1:]
    recommendations = user_item_matrix.iloc[similar_users].mean(axis=0)
    return recommendations.sort_values(ascending=False).head(top_n)

Content-Based Filtering:
Code Example:

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data
items = pd.read_csv('items.csv')

# Create item features
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
item_features = tfidf_vectorizer.fit_transform(items['description'])

# Calculate item similarities
item_similarities = 1 - cosine(item_features.toarray())

# Recommend items to a user based on their preferences
def recommend(user_preferences, top_n=10):
    recommendations = item_similarities[user_preferences.nonzero()[0]].mean(axis=0)
    return recommendations.argsort()[::-1][:top_n]
