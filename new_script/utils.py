# recsys_core/utils.py
import numpy as np

def softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Converte scores arbitrari (es. rating predetti) in probabilità.
    Usato SOLO per exploration/sampling, non per MSE.
    """
    scores = np.nan_to_num(scores, nan=0.0)
    
    if temperature <= 0:
        return np.zeros_like(scores) # Fallback safe
        
    # Shift per stabilità numerica
    x = scores - np.max(scores)
    exp_x = np.exp(x / temperature)
    sum_exp_x = np.sum(exp_x)
    
    if sum_exp_x == 0:
        return np.ones_like(scores) / len(scores)
        
    return exp_x / sum_exp_x

def clip_rating(rating: float, min_r: float = 1.0, max_r: float = 5.0) -> float:
    """Forza il rating predetto nel range [1, 5]."""
    return max(min_r, min(rating, max_r))