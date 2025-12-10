import argparse
import pandas as pd
import copy
from pathlib import Path
from config import Config
from evaluator import HybridRatingPredictor

def run_full_experiment():
    data_dir = Path("../data")
    ratings_path = data_dir / "ratings.csv"
    movies_path = data_dir / "movies_enriched.csv"
    
    # ==============================================================================
    # FASE 1: RATING PREDICTION TOURNAMENT (MSE/MAE)
    # Obiettivo: Trovare il modello che sbaglia di meno nel prevedere il voto esatto.
    # ==============================================================================
    
    print(f"{'='*80}")
    print(f"PHASE 1: ACCURACY TOURNAMENT (Finding the Best Predictor)")
    print(f"{'='*80}\n")
    
    # Definiamo i candidati
    candidates = [
        # 1. Baseline: Solo Bias (Media globale + bias utente + bias item)
        Config(name="01_Baseline", alpha_cf=0.0, constraint_scaler=0.0, shrink_term=10),
        
        # 2. Pure SVD: Solo Collaborative Filtering
        Config(name="02_Pure_SVD", alpha_cf=1.0, constraint_scaler=0.0, svd_components=20),
        
        # 3. Pure Content: Solo Constraints (Vincoli su Regista, Genere, etc.)
        # Nota: constraint_scaler è alto qui per dare peso ai vincoli
        Config(name="03_Pure_Content", alpha_cf=0.0, constraint_scaler=0.2),
        
        # 4. Hybrid Balanced: 50% SVD / 50% Content
        Config(name="04_Hybrid_Balanced", alpha_cf=0.5, constraint_scaler=0.1, svd_components=20),
        
        # 5. Hybrid CF Dominant: SVD forte + piccola correzione Content
        Config(name="05_Hybrid_CF_Dom", alpha_cf=0.8, constraint_scaler=0.05, svd_components=30),
    ]

    phase1_results = []
    best_mse = float('inf')
    best_config = None

    for conf in candidates:
        print(f">>> Testing Candidate: {conf.name}")
        try:
            predictor = HybridRatingPredictor(conf, ratings_path, movies_path)
            
            # Valutazione (Limita gli utenti se vuoi fare un test veloce, es. limit_users=50)
            metrics = predictor.evaluate_test_set(limit_users=50)
            
            print(f"    [RESULT] MSE: {metrics['mse']:.4f} | MAE: {metrics['mae']:.4f}")
            
            phase1_results.append({
                "Model": conf.name,
                "MSE": metrics["mse"],
                "MAE": metrics["mae"]
            })
            
            # Keep Track of Winner
            if metrics["mse"] < best_mse:
                best_mse = metrics["mse"]
                best_config = conf
                
        except Exception as e:
            print(f"    [ERROR] Failed: {e}")

    # Show Phase 1 Leaderboard
    df_p1 = pd.DataFrame(phase1_results).sort_values("MSE")
    print(f"\n{'-'*80}")
    print("PHASE 1 LEADERBOARD (Lower MSE is Better)")
    print(f"{'-'*80}")
    print(df_p1.to_string(index=False))
    
    if not best_config:
        print("\nCritical Error: No valid model found. Exiting.")
        return

    print(f"\n>>> WINNER: {best_config.name} (MSE: {best_mse:.4f})")
    print(">>> Proceeding to Exploration Phase using the Winner configuration...\n\n")

    # ==============================================================================
    # FASE 2: EXPLORATION LAB (Precision vs Diversity)
    # Obiettivo: Usare il modello migliore e variare la Temperatura per esplorare.
    # ==============================================================================
    
    print(f"{'='*80}")
    print(f"PHASE 2: EXPLORATION LAB (Optimizing User Experience)")
    print(f"Base Model: {best_config.name}")
    print(f"{'='*80}\n")
    
    # Livelli di Esplorazione (MAB)
    exploration_levels = [
        ("Deterministic", 0.01), # Argmax (No Risk)
        ("Conservative",  0.5),  # Low Risk
        ("Balanced",      1.0),  # Standard Softmax
        ("Adventurous",   2.0),  # High Risk
        ("Chaos",         5.0)   # Pure Randomness
    ]
    
    phase2_results = []
    
    for label, temp in exploration_levels:
        print(f">>> Testing Strategy: {label} (Temp={temp})")
        
        # Clona la config vincente e modifica solo la temperatura
        # Usiamo copy per non modificare l'oggetto originale
        current_conf = copy.copy(best_config)
        current_conf.name = f"Winner_{label}"
        current_conf.temperature = temp
        current_conf.top_k = 10 # Fissiamo K per consistenza
        
        try:
            predictor = HybridRatingPredictor(current_conf, ratings_path, movies_path)
            
            # Valutazione Exploration (Sampling)
            exp_metrics = predictor.evaluate_exploration(limit_users=50)
            
            print(f"    [RESULT] Precision: {exp_metrics['precision']:.4f} | Diversity: {exp_metrics['diversity']:.4f}")
            
            phase2_results.append({
                "Strategy": label,
                "Temp": temp,
                "Precision": exp_metrics["precision"],
                "Diversity": exp_metrics["diversity"],
                "Novelty": exp_metrics["novelty"]
            })
            
        except Exception as e:
            print(f"    [ERROR] Failed: {e}")

    # Show Phase 2 Results
    df_p2 = pd.DataFrame(phase2_results)
    print(f"\n{'-'*80}")
    print("PHASE 2 REPORT: TRADE-OFF ANALYSIS")
    print(f"{'-'*80}")
    print(df_p2.to_string(index=False))
    print(f"{'-'*80}")
    print("NOTE: \n - High Precision = Safe recommendations (boring?)")
    print(" - High Diversity = Discovering new genres (risky?)")
    print(" - Look for the 'Sweet Spot' where Diversity jumps up but Precision holds.")
    print(f"{'='*80}")

if __name__ == "__main__":
    run_full_experiment()