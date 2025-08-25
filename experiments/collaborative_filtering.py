import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


ratings = pd.read_csv('../data/ratings.csv')
movies = pd.read_csv('../data/movies_enriched.csv')

combined_movies_data = pd.merge(ratings, movies, on='movie_id')

rating_utility_matrix = combined_movies_data.pivot_table(values='rating', index='user_id', columns='movie_title', fill_value=0)

X = rating_utility_matrix.T

SVD = TruncatedSVD(n_components=30)
transformed_matrix = SVD.fit_transform(X)

corr_matrix = np.corrcoef(transformed_matrix)
print(corr_matrix.shape)
movies = rating_utility_matrix.columns
movies_list = list(movies)

movie_toy_story = movies_list.index('Toy Story')
corr_toy_story = corr_matrix[movie_toy_story]
correleted_movies_toy_story = list(movies[(corr_toy_story < 1.0) & (corr_toy_story > 0.7)])

for movie, corr in zip(movies, corr_toy_story):
    if corr < 1.0 and corr > 0.7:
        print(f"{movie}: {corr}")