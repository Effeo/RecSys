from dataclasses import dataclass
from typing import Optional

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
    sampling_top_k_pool: Optional[int] = 50 
    # --------------------------

    # --- PESI CONTENT-BASED (BOOSTED x10) ---
    # Aumentati drasticamente per competere con lo score della SVD.
    # Ora il sistema sentirà davvero la differenza se un film ha un premio o il regista giusto.
    award_weight: float = 5.0      # Era 0.5
    director_weight: float = 5.0   # Era 0.5
    runtime_weight: float = 3.0    # Era 0.3
    
    # --- MALUS (BOOSTED) ---
    # Anche i filtri negativi diventano più severi
    runtime_outside_malus: float = 5.0   # Era 0.5
    missing_runtime_malus: float = 1.0   # Era 0.1
    forbidden_genre_malus: float = 50.0  # Era 5.0 -> Ora è un muro quasi invalicabile
    
    # Recency (Lasciamo invariato per ora, è un bias sottile)
    missing_year_malus: float = 0.1
    year_below_malus_per_year: float = 0.05
    
    random_seed: int = 42
    verbose_users: int = 0