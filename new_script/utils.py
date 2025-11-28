# recsys_core/utils.py
import numpy as np

def softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Calcola la funzione softmax con temperature scaling.
    Gestisce stabilità numerica e casi limite (temperature <= 0).
    """
    scores = np.nan_to_num(scores, nan=0.0)
    
    if temperature <= 0:
        temperature = 1.0
        
    # Shift per stabilità numerica (evita overflow dell'esponenziale)
    x = scores - np.max(scores)
    exp_x = np.exp(x / temperature)
    sum_exp_x = np.sum(exp_x)
    
    if sum_exp_x == 0:
        return np.ones_like(scores) / len(scores)
        
    return exp_x / sum_exp_x