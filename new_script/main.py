import argparse
import pandas as pd
import copy
import json
import datetime
from pathlib import Path
from dataclasses import asdict

from config import Config
from evaluator import HybridRatingPredictor

def run_full_experiment():
    data_dir = Path("../data")
    ratings_path = data_dir / "ratings.csv"
    movies_path = data_dir / "movies_enriched.csv"
    
    # Generazione Timestamp
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"results_{ts}.json"
    
    full_experiment_data = {
        "timestamp": ts,
        "phase1_accuracy_tournament": [],
        "winning_config": None,
        "phase2_exploration_lab": []
    }

    # ==============================================================================
    # FASE 1: RATING PREDICTION TOURNAMENT (MSE/MAE)
    # ==============================================================================
    
    print(f"{'='*80}")
    print(f"PHASE 1: ACCURACY TOURNAMENT (Finding the Best Predictor)")
    print(f"{'='*80}\n")
    
    candidates = [
        Config(name="01_Baseline", alpha_cf=0.0, constraint_scaler=0.0, shrink_term=10),
        Config(name="02_Pure_SVD", alpha_cf=1.0, constraint_scaler=0.0, svd_components=20),
        Config(name="03_Pure_Content", alpha_cf=0.0, constraint_scaler=0.2),
        Config(name="04_Hybrid_Balanced", alpha_cf=0.5, constraint_scaler=0.1, svd_components=20),
        Config(name="05_Hybrid_CF_Dom", alpha_cf=0.8, constraint_scaler=0.05, svd_components=30),
    ]

    phase1_results = []
    best_mse = float('inf')
    best_config = None

    for conf in candidates:
        print(f">>> Testing Candidate: {conf.name}")
        try:
            predictor = HybridRatingPredictor(conf, ratings_path, movies_path)
            
            # Nota: Ho rimesso limit_users=50 per velocità, rimuovilo per il test completo
            metrics = predictor.evaluate_test_set(limit_users=50)
            
            print(f"    [RESULT] MSE: {metrics['mse']:.4f} | MAE: {metrics['mae']:.4f}")
            
            # --- CORREZIONE QUI SOTTO (Chiavi Maiuscole) ---
            result_entry = {
                "Model": conf.name,             # Chiave Maiuscola per estetica tabella
                "MSE": float(metrics["mse"]),   # Chiave Maiuscola per matchare sort_values
                "MAE": float(metrics["mae"]),   # Chiave Maiuscola
                "n_samples": metrics["n_samples"]
            }
            phase1_results.append(result_entry)
            
            if metrics["mse"] < best_mse:
                best_mse = metrics["mse"]
                best_config = conf
                
        except Exception as e:
            print(f"    [ERROR] Failed: {e}")

    # Salvataggio dati Fase 1
    full_experiment_data["phase1_accuracy_tournament"] = phase1_results

    # Ora sort_values("MSE") funzionerà perché la chiave nel dizionario è "MSE"
    if phase1_results:
        df_p1 = pd.DataFrame(phase1_results).sort_values("MSE")
        print(f"\n{'-'*80}")
        print("PHASE 1 LEADERBOARD (Lower MSE is Better)")
        print(f"{'-'*80}")
        print(df_p1.to_string(index=False))
    else:
        print("\n[ERROR] No results generated in Phase 1.")
        return
    
    if not best_config:
        print("\nCritical Error: No valid model found. Exiting.")
        return

    print(f"\n>>> WINNER: {best_config.name} (MSE: {best_mse:.4f})")
    
    full_experiment_data["winning_config"] = asdict(best_config)
    
    print(">>> Proceeding to Exploration Phase using the Winner configuration...\n\n")

    # ==============================================================================
    # FASE 2: EXPLORATION LAB
    # ==============================================================================
    
    print(f"{'='*80}")
    print(f"PHASE 2: EXPLORATION LAB (Optimizing User Experience)")
    print(f"Base Model: {best_config.name}")
    print(f"{'='*80}\n")
    
    exploration_levels = [
        ("Deterministic", 0.01),
        ("Conservative",  0.5),
        ("Balanced",      1.0),
        ("Adventurous",   2.0),
        ("Chaos",         5.0)
    ]
    
    phase2_results = []
    
    for label, temp in exploration_levels:
        print(f">>> Testing Strategy: {label} (Temp={temp})")
        
        current_conf = copy.copy(best_config)
        current_conf.name = f"Winner_{label}"
        current_conf.temperature = temp
        current_conf.top_k = 10 
        
        try:
            predictor = HybridRatingPredictor(current_conf, ratings_path, movies_path)
            
            exp_metrics = predictor.evaluate_exploration(limit_users=50)
            
            print(f"    [RESULT] Prec: {exp_metrics['precision']:.4f} | Div: {exp_metrics['diversity']:.4f} | Nov: {exp_metrics['novelty']:.2f}")
            
            phase2_results.append({
                "strategy": label,
                "temperature": temp,
                "precision": float(exp_metrics["precision"]),
                "diversity": float(exp_metrics["diversity"]),
                "novelty": float(exp_metrics["novelty"])
            })
            
        except Exception as e:
            print(f"    [ERROR] Failed: {e}")

    full_experiment_data["phase2_exploration_lab"] = phase2_results

    if phase2_results:
        df_p2 = pd.DataFrame(phase2_results)
        print(f"\n{'-'*80}")
        print("PHASE 2 REPORT: TRADE-OFF ANALYSIS")
        print(f"{'-'*80}")
        print(df_p2.to_string(index=False))
    
    # ==============================================================================
    # SALVATAGGIO SU FILE JSON
    # ==============================================================================
    try:
        with open(output_filename, "w") as f:
            json.dump(full_experiment_data, f, indent=4)
        print(f"\n{'+'*80}")
        print(f"SUCCESS! Full results saved to: {output_filename}")
        print(f"{'+'*80}")
    except Exception as e:
        print(f"\n[ERROR] Could not save JSON file: {e}")

if __name__ == "__main__":
    run_full_experiment()