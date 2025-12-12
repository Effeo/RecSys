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
    
    # candidates = [
    #     Config(name="01_Baseline", alpha_cf=0.0, constraint_scaler=0.0, shrink_term=10),
    #     Config(name="02_Pure_SVD", alpha_cf=1.0, constraint_scaler=0.0, svd_components=20),
    #     Config(name="03_Pure_Content", alpha_cf=0.0, constraint_scaler=0.2),
    #     Config(name="04_Hybrid_Balanced", alpha_cf=0.5, constraint_scaler=0.1, svd_components=20),
    #     Config(name="05_Hybrid_CF_Dom", alpha_cf=0.8, constraint_scaler=0.05, svd_components=30),
    # ]

    candidates = []
    
    # ==============================================================================
    # GRUPPO 1: HYBRID BALANCE (alpha_cf) - Impatto primario su MSE/MAE
    # ------------------------------------------------------------------------------
    # OBIETTIVO: Trovare il bilanciamento tra Collaborative Filtering (CF) e 
    # Content-Based (CB) che massimizza l'accuratezza.
    # ==============================================================================
    candidates.extend([
        Config(name="G1_PureContent", alpha_cf=0.0),            # 100% Content
        Config(name="G1_ContentDominant", alpha_cf=0.2),        # 80% Content, 20% CF
        Config(name="G1_Balanced", alpha_cf=0.5),               # Default
        Config(name="G1_CFDominant", alpha_cf=0.8),             # 20% Content, 80% CF
        Config(name="G1_PureCF", alpha_cf=1.0),                 # 100% CF
        Config(name="G1_ComplexCF", alpha_cf=0.8, svd_components=100), # CF Dominante con maggiore risoluzione SVD
    ])

    # ==============================================================================
    # GRUPPO 2: EXPLORATION & DIVERSITY (temperature) - Impatto primario su P/N/D
    # ------------------------------------------------------------------------------
    # OBIETTIVO: Variare la 'temperature' della Softmax per controllare il trade-off 
    # tra Precisione (bassa T) e Novelty/Diversity (alta T).
    # ==============================================================================
    base_s = 0.5
    candidates.extend([
        Config(name="G2_Deterministic", alpha_cf=base_s, temperature=0.01), # Simula Argmax (Precisione Massima)
        Config(name="G2_Conservative", alpha_cf=base_s, temperature=0.5),
        Config(name="G2_Balanced", alpha_cf=base_s, temperature=1.0),   # Default
        Config(name="G2_Adventurous", alpha_cf=base_s, temperature=2.0),
        Config(name="G2_Chaos", alpha_cf=base_s, temperature=5.0), # Massima Exploration
    ])

    # ==============================================================================
    # GRUPPO 3: POPULARITY BIAS (use_popularity/popularity_weight) - Impatto su MSE/MAE e Novelty
    # ------------------------------------------------------------------------------
    # OBIETTIVO: Vedere come l'aggiunta di un bonus ai film popolari influenzi 
    # l'accuratezza e le metriche di scoperta (Novelty).
    # ==============================================================================
    candidates.extend([
        Config(name="G3_NoBias", alpha_cf=0.3, use_popularity=False),
        Config(name="G3_Subtle", alpha_cf=0.3, use_popularity=True, popularity_weight=0.5), # Default
        Config(name="G3_Moderate", alpha_cf=0.3, use_popularity=True, popularity_weight=5.0),
        Config(name="G3_Strong", alpha_cf=0.3, use_popularity=True, popularity_weight=10.0),
        Config(name="G3_Dominant", alpha_cf=0.3, use_popularity=True, popularity_weight=20.0),
    ])

    # ==============================================================================
    # GRUPPO 4: CONTENT ANATOMY (Pesi Content) - Impatto primario su MSE/MAE
    # ------------------------------------------------------------------------------
    # OBIETTIVO: Isolare l'importanza dei singoli metadati (award_weight, director_weight, runtime_weight). 
    # Alpha_cf basso (0.1) per amplificare l'effetto Content.
    # ==============================================================================
    candidates.extend([
        Config(name="G4_AllEqual", alpha_cf=0.1, award_weight=5.0, director_weight=5.0, runtime_weight=3.0), # Base forte
        Config(name="G4_DirectorOnly", alpha_cf=0.1, award_weight=0.0, director_weight=10.0, runtime_weight=0.0),
        Config(name="G4_AwardOnly", alpha_cf=0.1, award_weight=10.0, director_weight=0.0, runtime_weight=0.0),
        Config(name="G4_RuntimeOnly", alpha_cf=0.1, award_weight=0.0, director_weight=0.0, runtime_weight=10.0),
        Config(name="G4_NoMeta", alpha_cf=0.1, award_weight=0.0, director_weight=0.0, runtime_weight=0.0), # Content solo sui generi
    ])

    # ==============================================================================
    # GRUPPO 5: STRICTNESS (Malus Generi/Runtime) - Impatto primario sul Ranking P/N/D
    # ------------------------------------------------------------------------------
    # OBIETTIVO: Testare se l'essere molto severi sui filtri (forbidden_genre_malus, missing_runtime_malus) 
    # causi liste troppo ristrette o assenti.
    # ==============================================================================
    candidates.extend([
        Config(name="G5_Anarchy", alpha_cf=0.5, forbidden_genre_malus=0.0, missing_runtime_malus=0.0),
        Config(name="G5_Permissive", alpha_cf=0.5, forbidden_genre_malus=5.0, missing_runtime_malus=0.0), 
        Config(name="G5_Standard", alpha_cf=0.5, forbidden_genre_malus=50.0, missing_runtime_malus=1.0), # Default
        Config(name="G5_Strict", alpha_cf=0.5, forbidden_genre_malus=200.0, missing_runtime_malus=10.0),
        Config(name="G5_Draconian", alpha_cf=0.5, forbidden_genre_malus=1000.0, missing_runtime_malus=100.0),
    ])
    
    # ==============================================================================
    # GRUPPO 6: RECENCY BIAS (year_below_malus_per_year) - Impatto primario su MSE/MAE
    # ------------------------------------------------------------------------------
    # OBIETTIVO: Capire se il bias per i film vecchi migliora l'accuratezza.
    # ==============================================================================
    candidates.extend([
        Config(name="G6_Timeless", alpha_cf=0.5, year_below_malus_per_year=0.0),
        Config(name="G6_Nostalgic", alpha_cf=0.5, year_below_malus_per_year=0.01),
        Config(name="G6_Modernist", alpha_cf=0.5, year_below_malus_per_year=0.2),
        Config(name="G6_NewGen", alpha_cf=0.5, year_below_malus_per_year=0.5),
    ])

    # ==============================================================================
    # GRUPPO 7: DATA TRUST (shrink_term) - Impatto primario su MSE/MAE
    # ------------------------------------------------------------------------------
    # OBIETTIVO: Testare l'effetto del termine di riduzione sui bias (b_u, b_i) 
    # per prevenire l'overfitting.
    # ==============================================================================
    base_cf_shrink = 0.8
    candidates.extend([
        Config(name="G7_Naive", alpha_cf=base_cf_shrink, shrink_term=0),   # Nessuna riduzione
        Config(name="G7_Optimistic", alpha_cf=base_cf_shrink, shrink_term=2),
        Config(name="G7_Standard", alpha_cf=base_cf_shrink, shrink_term=10), # Default
        Config(name="G7_Skeptical", alpha_cf=base_cf_shrink, shrink_term=30),
        Config(name="G7_Paranoid", alpha_cf=base_cf_shrink, shrink_term=100), # Riduzione estrema
    ])

    # ==============================================================================
    # GRUPPO 8: MATRIX RESOLUTION (svd_components) - Impatto primario su MSE/MAE
    # ------------------------------------------------------------------------------
    # OBIETTIVO: Trovare la risoluzione ottimale della mappa dei gusti.
    # ==============================================================================
    base_cf_svd = 0.8
    candidates.extend([
        Config(name="G8_LowRes_5", alpha_cf=base_cf_svd, svd_components=5),
        Config(name="G8_MidRes_30", alpha_cf=base_cf_svd, svd_components=30), # Default
        Config(name="G8_HighRes_60", alpha_cf=base_cf_svd, svd_components=60),
        Config(name="G8_UltraRes_150", alpha_cf=base_cf_svd, svd_components=150),
        Config(name="G8_ExtremeRes_300", alpha_cf=base_cf_svd, svd_components=300),
    ])

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
    # NUOVA LOGICA: IDENTIFICAZIONE VINCITORI DI GRUPPO
    # ==============================================================================
    
    # Converti in DataFrame per facilitare il raggruppamento e l'estrazione
    df_p1 = pd.DataFrame(phase1_results)
    
    # Estrai il prefisso del gruppo (es. "G1", "G2", ecc.)
    df_p1["Group"] = df_p1["Model"].str.extract(r"^(G\d+)_")
    
    # Trova l'indice del modello con il minimo MSE per ogni gruppo
    idx_min_mse_per_group = df_p1.groupby("Group")["MSE"].idxmin()
    
    # Filtra il DataFrame per ottenere solo i risultati dei vincitori di gruppo
    group_winners_df = df_p1.loc[idx_min_mse_per_group]
    
    # Mappa i nomi dei modelli vincitori alle loro configurazioni originali
    # (per fare ciò, dobbiamo passare da phase1_results a candidates)
    group_winner_names = group_winners_df["Model"].tolist()
    
    group_winner_configs = []
    for conf in candidates:
        if conf.name in group_winner_names:
            group_winner_configs.append(conf)
            
    print(f"\n{'-'*80}")
    print(f"VINCITORI DI GRUPPO IDENTIFICATI PER LA FASE 2:")
    for conf in group_winner_configs:
        mse_val = group_winners_df[group_winners_df['Model'] == conf.name]['MSE'].iloc[0]
        print(f"* {conf.name} (MSE: {mse_val:.4f})")
    print(f"{'-'*80}")

    print(">>> Procedendo al Laboratorio di Esplorazione con i Vincitori di Gruppo...\n\n")

    # ==============================================================================
    # FASE 2: EXPLORATION LAB
    # ==============================================================================
    
    print(f"{'='*80}")
    print(f"PHASE 2: EXPLORATION LAB (Optimizing User Experience for each Group Winner)")
    print(f"{'='*80}\n")
    
    exploration_levels = [
        ("Deterministic", 0.01),
        ("Conservative",  0.5),
        ("Balanced",      1.0),
        ("Adventurous",   2.0),
        ("Chaos",         5.0)
    ]
    
    phase2_results = []
    
    # CICLO AGGIUNTO: Itera su ogni configurazione vincente per gruppo
    for base_conf in group_winner_configs:
        
        print(f"\n--- TESTING BASE MODEL: {base_conf.name} ---")
        
        for label, temp in exploration_levels:
            print(f">>> Testing Strategy: {label} (Temp={temp})")
            
            # NOTA IMPORTANTE: Si parte da una COPIA del modello base (il vincitore di gruppo)
            current_conf = copy.copy(base_conf)
            current_conf.name = f"{base_conf.name}_{label}"
            current_conf.temperature = temp
            current_conf.top_k = 10 
            
            try:
                predictor = HybridRatingPredictor(current_conf, ratings_path, movies_path)
                
                # Ho rimesso limit_users=50 per velocità, rimuovilo per il test completo
                exp_metrics = predictor.evaluate_exploration(limit_users=50) 
                
                print(f"    [RESULT] Prec: {exp_metrics['precision']:.4f} | Div: {exp_metrics['diversity']:.4f} | Nov: {exp_metrics['novelty']:.2f}")
                
                phase2_results.append({
                    "base_model": base_conf.name, # Aggiunge il nome del modello base per l'analisi
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
        # Si usa 'base_model' per raggruppare i risultati nella stampa
        df_p2 = pd.DataFrame(phase2_results)
        df_p2_styled = df_p2.sort_values(by=["base_model", "temperature"]).to_string(index=False)
        print(f"\n{'-'*80}")
        print("PHASE 2 REPORT: TRADE-OFF ANALYSIS (Per Vincitore di Gruppo)")
        print(f"{'-'*80}")
        print(df_p2_styled)
    
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