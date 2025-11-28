# recsys_core/config.py
from dataclasses import dataclass

@dataclass
class Config:
    name: str = ""
    
    # Parametri Base
    alpha_cf: float = 0.5
    temperature: float = 2.0
    top_k: int = 10
    svd_components: int = 30
    shrink_term: int = 5
    
    # Popularity Bias
    use_popularity_bias: bool = False
    popularity_weight: float = 0.0
    
    # --- LOGICA DI SAMPLING ---
    # False: Deterministico (Argmax)
    # True: Probabilistico (np.random.choice)
    use_probabilistic_sampling: bool = False 
    
    # NUOVO: Definisce quanto è grande il pool da cui pescare
    # 50 = Pesca tra i top 50 score (Safe)
    # 200 = Pesca tra i top 200 (More Discovery)
    # None = Pesca su tutto il catalogo (Risky)
    sampling_top_k_pool: int = 50 
    # --------------------------

    # Pesi Content-Based
    award_weight: float = 0.5
    director_weight: float = 0.5
    runtime_weight: float = 0.3
    
    # Malus
    runtime_outside_malus: float = 0.5
    missing_runtime_malus: float = 0.1
    forbidden_genre_malus: float = 5.0
    missing_year_malus: float = 0.1
    year_below_malus_per_year: float = 0.05
    
    random_seed: int = 42
    verbose_users: int = 0