# recsys_core/evaluator.py
import numpy as np
import pandas as pd
import math
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional

from config import Config
from profiling import infer_user_prefs
from utils import clip_rating, softmax

class HybridRatingPredictor:
    def __init__(self, config: Config, ratings_path: str, movies_path: str):
        self.config = config
        self._load_data(ratings_path, movies_path)
        
        # Stats per Novelty (Phase 2)
        self.item_counts = self.ratings["movie_id"].value_counts().to_dict()
        self.total_interactions = len(self.ratings)
        
        # Parametri appresi (Phase 1)
        self.global_mean = 0.0
        self.user_biases = {}
        self.item_biases = {}
        
        # Matrici SVD
        self.U = None     # User features
        self.Vt = None    # Item features
        self.user_id_to_idx = {}
        self.movie_id_to_idx = {}

    def _load_data(self, ratings_path, movies_path):
        self.ratings = pd.read_csv(ratings_path)
        self.movies = pd.read_csv(movies_path)
        
        # Preprocessing Film
        self.movies.columns = [c.strip() for c in self.movies.columns]
        for col, val in [("awards", 0), ("director", "Unknown"), ("runtime", 90)]:
            if col not in self.movies.columns:
                self.movies[col] = val
        
        self.movies["movie_id"] = pd.to_numeric(self.movies["movie_id"], errors='coerce').fillna(0).astype(int)
        self.ratings["movie_id"] = pd.to_numeric(self.ratings["movie_id"], errors='coerce').fillna(0).astype(int)

        # Identificazione colonne generi per calcolo Diversity
        metadata = {"movie_id", "movie_title", "runtime", "director", "awards", "release_date", "video_release_date", "IMDb_URL", "unknown", ""}
        self.genre_cols = [c for c in self.movies.columns if c not in metadata]
        for col in self.genre_cols:
             self.movies[col] = pd.to_numeric(self.movies[col], errors='coerce').fillna(0)

    def fit(self, train_df: pd.DataFrame):
        """
        Addestra il modello calcolando Bias e SVD sui residui.
        """
        # print(f"[{self.config.name}] Fitting Model...")
        df = train_df.copy()
        C = self.config.shrink_term

        # 1. Global Mean
        self.global_mean = df["rating"].mean()
        df["rating_norm"] = df["rating"] - self.global_mean

        # 2. Item Biases
        item_stats = df.groupby("movie_id")["rating_norm"].agg(['sum', 'count'])
        self.item_biases = (item_stats['sum'] / (item_stats['count'] + C)).to_dict()
        df["rating_norm"] -= df["movie_id"].map(self.item_biases).fillna(0)

        # 3. User Biases
        user_stats = df.groupby("user_id")["rating_norm"].agg(['sum', 'count'])
        self.user_biases = (user_stats['sum'] / (user_stats['count'] + C)).to_dict()
        df["rating_norm"] -= df["user_id"].map(self.user_biases).fillna(0)

        # 4. SVD sui Residui
        urm = df.pivot_table(index="user_id", columns="movie_id", values="rating_norm", fill_value=0)
        
        self.user_id_to_idx = {uid: i for i, uid in enumerate(urm.index)}
        self.movie_id_to_idx = {mid: i for i, mid in enumerate(urm.columns)}
        
        X = urm.values
        # Gestione caso pochi dati
        if X.shape[0] > 0 and X.shape[1] > 0:
            n_components = min(self.config.svd_components, min(X.shape) - 1)
            if n_components > 0:
                svd = TruncatedSVD(n_components=n_components, random_state=self.config.random_seed)
                self.U = svd.fit_transform(X) # User Matrix
                self.Vt = svd.components_     # Item Matrix
            else:
                self.U = np.zeros((X.shape[0], 1))
                self.Vt = np.zeros((1, X.shape[1]))
        
        # print("Model Fitted.")

    def predict_rating(self, user_id: int, movie_id: int, user_prefs: Dict) -> float:
        """
        Predizione Ibrida Puntuale.
        """
        # A. Baseline
        b_u = self.user_biases.get(user_id, 0.0)
        b_i = self.item_biases.get(movie_id, 0.0)
        baseline = self.global_mean + b_u + b_i
        
        # B. Collaborative Residual (SVD)
        svd_res = 0.0
        u_idx = self.user_id_to_idx.get(user_id)
        i_idx = self.movie_id_to_idx.get(movie_id)
        
        if u_idx is not None and i_idx is not None and self.U is not None:
            svd_res = np.dot(self.U[u_idx], self.Vt[:, i_idx])
            
        # C. Constraint Residual (Content)
        content_res = 0.0
        if user_prefs:
            raw_score = self._calculate_single_constraint(movie_id, user_prefs)
            content_res = raw_score * self.config.constraint_scaler
            
        # D. Ibridazione
        hybrid_residual = (self.config.alpha_cf * svd_res) + ((1 - self.config.alpha_cf) * content_res)
        
        final_rating = baseline + hybrid_residual
        return clip_rating(final_rating)

    def _calculate_single_constraint(self, movie_id: int, prefs: Dict) -> float:
        row = self.movies[self.movies.movie_id == movie_id]
        if row.empty: return 0.0
        row = row.iloc[0]
        
        score = 0.0
        
        # Generi
        for g in prefs.get("generi_desiderati", []):
            if g in row and row[g] > 0: score += 1.0
            
        for g in prefs.get("generi_vietati", []):
            if g in row and row[g] > 0: score -= self.config.forbidden_genre_malus
            
        # Runtime
        if prefs.get("preferred_runtime"):
            target = prefs["preferred_runtime"]
            rt = row["runtime"]
            if pd.isna(rt) or rt == 0:
                score -= self.config.missing_runtime_malus
            else:
                diff = abs(rt - target)
                if diff <= prefs.get("tolleranza_runtime", 15):
                    score += self.config.runtime_weight
                else:
                    malus = min(((diff - 15) / 15), 3) * self.config.runtime_outside_malus
                    score -= malus
                    
        # Awards & Director
        if prefs.get("prefer_award_winning") and row["awards"] > 0:
            score += self.config.award_weight
            
        if row["director"] in prefs.get("favorite_directors", []):
            score += self.config.director_weight
            
        return score

    # ==========================================
    # PHASE 1: PREDICTION EVALUATION (MSE/MAE)
    # ==========================================
    def evaluate_test_set(self, limit_users: int = None) -> Dict:
        train, test = self._split_stratified()
        self.fit(train)
        
        # print("Evaluating Prediction Accuracy (MSE/MAE)...")
        y_true = []
        y_pred = []
        
        users_to_test = test["user_id"].unique()
        if limit_users:
            users_to_test = users_to_test[:limit_users]
            
        for uid in users_to_test:
            u_train = train[train["user_id"] == uid]
            prefs = infer_user_prefs(u_train, self.movies)
            
            u_test = test[test["user_id"] == uid]
            
            for _, row in u_test.iterrows():
                mid = row["movie_id"]
                actual = row["rating"]
                predicted = self.predict_rating(uid, mid, prefs)
                y_true.append(actual)
                y_pred.append(predicted)

        if not y_true:
            return {"mse": 99.9, "mae": 99.9, "n_samples": 0}

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        
        return {"mse": mse, "mae": mae, "n_samples": len(y_true)}

    # ==========================================
    # PHASE 2: EXPLORATION EVALUATION (MAB)
    # ==========================================
    def evaluate_exploration(self, limit_users: int = None) -> Dict:
        train, test = self._split_stratified()
        self.fit(train)
        
        users_to_test = test["user_id"].unique()
        if limit_users: users_to_test = users_to_test[:limit_users]
        
        metrics = {"precision": [], "diversity": [], "novelty": []}
        candidate_movies = self.movies["movie_id"].values
        
        # print(f"Evaluating Exploration (Temp={self.config.temperature})...")
        
        for uid in users_to_test:
            u_train = train[train["user_id"] == uid]
            prefs = infer_user_prefs(u_train, self.movies)
            
            # Pool simulato (Test items + Random items)
            u_test_items = test[test["user_id"] == uid]["movie_id"].values
            random_pool = np.random.choice(candidate_movies, size=100, replace=False) # 100 Random items
            pool_ids = np.unique(np.concatenate([u_test_items, random_pool]))
            
            predictions = []
            for mid in pool_ids:
                pred = self.predict_rating(uid, mid, prefs)
                predictions.append(pred)
            
            scores = np.array(predictions)
            
            # MAB STEP: Softmax Sampling
            probs = softmax(scores, temperature=self.config.temperature)
            
            k = min(self.config.top_k, len(pool_ids))
            
            # Scelta probabilistica
            try:
                rec_indices = np.random.choice(len(pool_ids), size=k, replace=False, p=probs)
                rec_ids = pool_ids[rec_indices]
            except ValueError:
                # Fallback se probs contiene NaN o somma a 0
                rec_ids = pool_ids[:k]
            
            # Calcolo Metriche
            positives = test[(test["user_id"] == uid) & (test["rating"] >= 3.0)]["movie_id"].values
            
            hits = len(set(rec_ids) & set(positives))
            metrics["precision"].append(hits / k)
            metrics["diversity"].append(self._calculate_diversity(rec_ids))
            metrics["novelty"].append(self._calculate_novelty(rec_ids))

        return {k: np.mean(v) if v else 0.0 for k, v in metrics.items()}

    def _calculate_diversity(self, rec_ids):
        unique_genres = set()
        recs = self.movies[self.movies["movie_id"].isin(rec_ids)]
        if recs.empty: return 0.0
        for _, row in recs.iterrows():
            for g in self.genre_cols:
                if row[g] > 0: unique_genres.add(g)
        return len(unique_genres) / 18.0 

    def _calculate_novelty(self, rec_ids):
        if len(rec_ids) == 0: return 0.0
        nov_scores = []
        for mid in rec_ids:
            count = self.item_counts.get(mid, 1)
            prob = count / self.total_interactions
            nov_scores.append(-math.log2(prob))
        return np.mean(nov_scores)

    def _split_stratified(self, test_ratio=0.2):
        # print("Splitting Data (Stratified)...")
        user_counts = self.ratings["user_id"].value_counts()
        valid_users = user_counts[user_counts >= 5].index
        
        valid_ratings = self.ratings[self.ratings["user_id"].isin(valid_users)]
        sparse_ratings = self.ratings[~self.ratings["user_id"].isin(valid_users)]
        
        train_valid, test = train_test_split(
            valid_ratings, 
            test_size=test_ratio, 
            stratify=valid_ratings["user_id"], 
            random_state=self.config.random_seed
        )
        train = pd.concat([train_valid, sparse_ratings])
        return train, test