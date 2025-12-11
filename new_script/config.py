# recsys_core/config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    name: str = ""
    
    # --- Ibridazione ---
    # 1.0 = Solo SVD (Collaborative)
    # 0.0 = Solo Constraints (Content)
    # 0.5 = Media pesata dei residui
    alpha_cf: float = 0.5 
    
    # --- Parametri Modello ---
    svd_components: int = 20
    shrink_term: int = 10  # Shrinkage per i bias (b_u, b_i)
    
    # --- Constraint Scaling ---
    # I constraint scores (es. +5.0 award) sono troppo grandi per una scala 1-5.
    # Questo fattore li riduce. Es: 5.0 * 0.1 = +0.5 stelle di bonus.
    constraint_scaler: float = 0.1 

    use_popularity: bool = True
    popularity_weight: float = 0.5
    
    # --- Pesi Content-Based (User Constraints) ---
    award_weight: float = 5.0      
    director_weight: float = 5.0   
    runtime_weight: float = 3.0    
    
    # --- Malus ---
    runtime_outside_malus: float = 5.0   
    missing_runtime_malus: float = 1.0   
    forbidden_genre_malus: float = 50.0  
    missing_year_malus: float = 0.1
    year_below_malus_per_year: float = 0.05
    
    # --- Exploration ---
    temperature: float = 1.0 # Usato solo per la softmax finale (opzionale)
    
    random_seed: int = 42
    verbose_users: int = 0