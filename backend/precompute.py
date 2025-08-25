import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import json
import config

def main():
    ratings = pd.read_csv(config.RATINGS_FILE)
    movies = pd.read_csv(config.MOVIES_FILE)

    # Merge rating + info film
    combined_movies_data = pd.merge(ratings, movies, on="movie_id")

    # Crea matrice utility (users x movies)
    rating_utility_matrix = combined_movies_data.pivot_table(
        values="rating", index="user_id", columns="movie_title", fill_value=0
    )

    # Transposta per lavorare sui film
    X = rating_utility_matrix.T

    # SVD
    SVD = TruncatedSVD(n_components=30, random_state=42)
    transformed_matrix = SVD.fit_transform(X)

    # Matrice di correlazione film-film
    corr_matrix = np.corrcoef(transformed_matrix)

    movies_list = list(rating_utility_matrix.columns)

    # Salva lista film in JSON
    with open(config.MOVIES_LIST_FILE, "w", encoding="utf-8") as f:
        json.dump(movies_list, f, ensure_ascii=False, indent=2)

    # Salva matrice in formato npy
    np.save(config.MOVIES_CORR_FILE, corr_matrix)

    print(f"Salvati {config.MOVIES_LIST_FILE} e {config.MOVIES_CORR_FILE}")

if __name__ == "__main__":
    main()
