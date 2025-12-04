import argparse
import json
import datetime
from pathlib import Path
from config import Config
from evaluator import HybridEvaluator
import pandas as pd

def run_experiment(mode: str):
    data_dir = Path("../data")
    configs_to_run = []
    experiment_results = []
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"results_batteries_{ts}.json"
    csv_filename = f"results_batteries_{ts}.csv"


    if mode == "test":
        print(">>> TEST MODE (Smoke Test) <<<")
        # Test veloce su pochi utenti per verificare che non ci siano bug
        configs_to_run.append(Config(name="QuickTest_Smoke", verbose_users=1, top_k=5))
        user_limit = 5
    else:
        print(">>> FULL GRID SEARCH: EXTENDED BATTERY <<<")
        user_limit = None 
        
        # ==============================================================================
        # GRUPPO 1: HYBRID BALANCE (Chi comanda? Storia o Contenuto?)
        # ------------------------------------------------------------------------------
        # OBIETTIVO: Trovare il punto in cui massimizziamo la precisione storica (CF) 
        # senza perdere la capacità di raccomandare item meno votati (Content).
        # PARAMETRI CHIAVE: 
        # - alpha_cf: 1.0 = Solo voti utenti (CF), 0.0 = Solo metadati film (CB).
        # - svd_components: Risoluzione della mappa utenti (Bassa=Macro gusti, Alta=Dettagli).
        # ==============================================================================
        configs_to_run.extend([
            Config(name="G1_PureContent", alpha_cf=0.0),            # Solo Content (Safe, ma bolla)
            Config(name="G1_ContentDominant", alpha_cf=0.2),        # Mix sbilanciato su Content
            Config(name="G1_Balanced", alpha_cf=0.5),               # Mix 50/50
            Config(name="G1_CFDominant", alpha_cf=0.8),             # Mix sbilanciato su CF (Voti utenti pesano di più)
            Config(name="G1_PureCF", alpha_cf=1.0),                 # Solo CF (Max Serendipity, rischio Cold Start)
            Config(name="G1_ComplexCF", alpha_cf=0.8, svd_components=100), # CF ad alta risoluzione (rischio Overfitting)
        ])

        # ==============================================================================
        # GRUPPO 2: SAMPLING & DISCOVERY (Noia vs Rischio)
        # ------------------------------------------------------------------------------
        # OBIETTIVO: Capire se l'utente preferisce una lista fissa e perfetta (Deterministica)
        # o una lista che cambia e sorprende (Probabilistica).
        # PARAMETRI CHIAVE:
        # - pool: Quanto è grande il secchio da cui pescare? (Piccolo=Sicuro, Grande=Rischioso)
        # - temperature: Quanto appiattire le differenze di score? (Alta=Caos/Varietà).
        # *Nota: Teniamo alpha_cf fisso a 0.4 per avere una base stabile.*
        # ==============================================================================
        base_s = 0.4
        configs_to_run.extend([
            Config(name="G2_Deterministic", alpha_cf=base_s, use_probabilistic_sampling=False), # Baseline (Argmax)
            Config(name="G2_Conservative", alpha_cf=base_s, use_probabilistic_sampling=True, sampling_top_k_pool=20, temperature=0.5), # Poco rischio, poca varietà
            Config(name="G2_Balanced", alpha_cf=base_s, use_probabilistic_sampling=True, sampling_top_k_pool=50, temperature=1.0),     # Standard
            Config(name="G2_Adventurous", alpha_cf=base_s, use_probabilistic_sampling=True, sampling_top_k_pool=100, temperature=1.5), # Alta variabilità
            Config(name="G2_Chaos", alpha_cf=base_s, use_probabilistic_sampling=True, sampling_top_k_pool=200, temperature=3.0),       # Rischio massimo (High Diversity)
            Config(name="G2_DeepCatalog", alpha_cf=base_s, use_probabilistic_sampling=True, sampling_top_k_pool=None, temperature=0.2),# Pesca su tutto, ma con temp bassa
        ])

        # ==============================================================================
        # GRUPPO 3: POPULARITY BIAS (Mainstream vs Nicchia)
        # ------------------------------------------------------------------------------
        # OBIETTIVO: Vedere se "aiutare" i film famosi aumenta il CTR o uccide la personalizzazione.
        # PARAMETRI CHIAVE:
        # - popularity_weight: Bonus aggiunto agli item globalmente famosi.
        # ==============================================================================
        configs_to_run.extend([
            Config(name="G3_NoBias", alpha_cf=0.3, use_popularity_bias=False), # Purista
            Config(name="G3_Subtle", alpha_cf=0.3, use_popularity_bias=True, popularity_weight=0.1), # Leggero aiuto
            Config(name="G3_Moderate", alpha_cf=0.3, use_popularity_bias=True, popularity_weight=0.5),
            Config(name="G3_Strong", alpha_cf=0.3, use_popularity_bias=True, popularity_weight=1.0), # Mainstream
            Config(name="G3_Dominant", alpha_cf=0.3, use_popularity_bias=True, popularity_weight=2.0), # La fama vince sui gusti
        ])

        # ==============================================================================
        # GRUPPO 4: CONTENT ANATOMY (Feature Ablation Study)
        # ------------------------------------------------------------------------------
        # OBIETTIVO: Capire quali metadati contano davvero per gli utenti (Regista? Premi? Durata?).
        # PARAMETRI CHIAVE: Pesi delle feature (award, director, runtime).
        # *Nota: Teniamo alpha_cf MOLTO BASSO (0.1) per isolare l'effetto del Content.*
        # ==============================================================================
        configs_to_run.extend([
            Config(name="G4_AllEqual", alpha_cf=0.1, award_weight=0.5, director_weight=0.5, runtime_weight=0.5), # Base
            Config(name="G4_DirectorOnly", alpha_cf=0.1, award_weight=0.0, director_weight=1.0, runtime_weight=0.0), # Autore conta
            Config(name="G4_AwardOnly", alpha_cf=0.1, award_weight=1.0, director_weight=0.0, runtime_weight=0.0),   # Prestigio conta
            Config(name="G4_RuntimeOnly", alpha_cf=0.1, award_weight=0.0, director_weight=0.0, runtime_weight=1.0), # Tempo conta
            Config(name="G4_NoMeta", alpha_cf=0.1, award_weight=0.0, director_weight=0.0, runtime_weight=0.0),      # Nessun peso extra
        ])

        # ==============================================================================
        # GRUPPO 5: STRICTNESS (I Guardiani della Soglia)
        # ------------------------------------------------------------------------------
        # OBIETTIVO: Capire se essere severi sui filtri (Malus) migliora la qualità percepita
        # o causa il problema "Zero Results".
        # PARAMETRI CHIAVE: Malus per genere proibito e dati mancanti.
        # ==============================================================================
        configs_to_run.extend([
            Config(name="G5_Anarchy", alpha_cf=0.5, forbidden_genre_malus=0.0, missing_runtime_malus=0.0), # Nessun filtro
            Config(name="G5_Permissive", alpha_cf=0.5, forbidden_genre_malus=0.5, missing_runtime_malus=0.0), # Soft Nudge
            Config(name="G5_Standard", alpha_cf=0.5, forbidden_genre_malus=5.0, missing_runtime_malus=0.1),   # Default
            Config(name="G5_Strict", alpha_cf=0.5, forbidden_genre_malus=20.0, missing_runtime_malus=1.0),    # Severo
            Config(name="G5_Draconian", alpha_cf=0.5, forbidden_genre_malus=100.0, missing_runtime_malus=10.0), # Hard Filter
        ])

        # ==============================================================================
        # GRUPPO 6: RECENCY BIAS (Il Fattore Tempo)
        # ------------------------------------------------------------------------------
        # OBIETTIVO: Capire il tasso di decadimento dell'interesse. I vecchi film valgono?
        # PARAMETRI CHIAVE: year_below_malus_per_year (Penalità cumulativa per ogni anno).
        # ==============================================================================
        configs_to_run.extend([
            Config(name="G6_Timeless", alpha_cf=0.5, year_below_malus_per_year=0.0), # 1950 = 2024
            Config(name="G6_Nostalgic", alpha_cf=0.5, year_below_malus_per_year=0.001), # Penalità impercettibile
            Config(name="G6_Modernist", alpha_cf=0.5, year_below_malus_per_year=0.02),  # -2% score per anno
            Config(name="G6_NewGen", alpha_cf=0.5, year_below_malus_per_year=0.05),     # -5% score (dopo 20 anni score è 0)
            Config(name="G6_FreshOnly", alpha_cf=0.5, year_below_malus_per_year=0.15),   # Uccide i classici subito
        ])

        # ==============================================================================
        # GRUPPO 7: DATA TRUST (Sparsity & Shrinkage)
        # ------------------------------------------------------------------------------
        # OBIETTIVO: Gestire i falsi positivi (film con 1 voto da 5 stelle).
        # PARAMETRI CHIAVE: shrink_term (freno per item con pochi voti).
        # *Nota: Teniamo alpha_cf ALTO (0.8) perché lo shrink agisce sulla parte CF.*
        # ==============================================================================
        configs_to_run.extend([
            Config(name="G7_Naive", alpha_cf=0.8, shrink_term=0),   # Fiducia cieca anche con 1 voto
            Config(name="G7_Optimistic", alpha_cf=0.8, shrink_term=2),
            Config(name="G7_Standard", alpha_cf=0.8, shrink_term=10), # Bilanciato
            Config(name="G7_Skeptical", alpha_cf=0.8, shrink_term=30), # Richiede molti voti per fidarsi
            Config(name="G7_Paranoid", alpha_cf=0.8, shrink_term=100), # Solo blockbuster consolidati
        ])

        # ==============================================================================
        # GRUPPO 8: MATRIX RESOLUTION (SVD Sensitivity)
        # ------------------------------------------------------------------------------
        # OBIETTIVO: Capire la "risoluzione" ottimale della mappa dei gusti.
        # - Pochi componenti (Underfitting): Il sistema generalizza troppo (es. "Ti piacciono i film").
        # - Troppi componenti (Overfitting): Il sistema impara il rumore (es. "Ti piace questo film solo perché hai cliccato per sbaglio").
        # *Nota: Fissiamo alpha_cf alto (0.8) per assicurarci che l'SVD sia il motore principale.*
        # ==============================================================================
        base_cf_svd = 0.8
        configs_to_run.extend([
            # Bassissima risoluzione: coglie solo i macro-generi (es. Action vs Romance)
            Config(name="G8_LowRes_5", alpha_cf=base_cf_svd, svd_components=5),
            # Bassa risoluzione: veloce e generalista
            Config(name="G8_LowRes_15", alpha_cf=base_cf_svd, svd_components=15),
            # Media risoluzione (Spesso il punto ottimale per dataset medi)
            Config(name="G8_MidRes_30", alpha_cf=base_cf_svd, svd_components=30),
            # Alta risoluzione: coglie sfumature sottili (es. "Cyberpunk anni '80")
            Config(name="G8_HighRes_60", alpha_cf=base_cf_svd, svd_components=60),
            # Altissima risoluzione: Rischio overfitting alto e tempi di calcolo maggiori
            Config(name="G8_UltraRes_150", alpha_cf=base_cf_svd, svd_components=150),
            # Estrema (solo se hai TANTI utenti/item, altrimenti è rumore puro)
            Config(name="G8_ExtremeRes_300", alpha_cf=base_cf_svd, svd_components=300),
        ])

        # ==============================================================================
        # GRUPPO 9: OUTPUT SCALE SENSITIVITY (Widget vs Catalog)
        # ------------------------------------------------------------------------------
        # OBIETTIVO: Capire come degrada la qualità all'aumentare dei risultati richiesti.
        # - top_k basso (5-10): Simuliamo un carosello "Top Picks" in Home Page.
        # - top_k alto (50-100): Simuliamo una pagina "Vedi tutti i consigliati".
        # *Domanda:* La precisione crolla verticalmente dopo il 10° item? Se sì, non fare pagine lunghe.
        # ==============================================================================
        base_k_cf = 0.5
        configs_to_run.extend([
            Config(name="G9_Widget_Small", alpha_cf=base_k_cf, top_k=3),   # Solo la crème de la crème
            Config(name="G9_Widget_Std", alpha_cf=base_k_cf, top_k=10),    # Standard
            Config(name="G9_Page_Medium", alpha_cf=base_k_cf, top_k=25),   # Una schermata piena
            Config(name="G9_Page_Large", alpha_cf=base_k_cf, top_k=50),    # Scroll impegnativo
            Config(name="G9_Catalog_Deep", alpha_cf=base_k_cf, top_k=100), # Deep dive (qui la precisione crollerà)
        ])

        # ==============================================================================
        # GRUPPO 10: DURATION TOLERANCE (Ho tempo o vado di fretta?)
        # ------------------------------------------------------------------------------
        # OBIETTIVO: Testare specificamente il `runtime_outside_malus`.
        # Differenza col Gruppo 4: Lì testavamo se la durata è una feature utile. 
        # Qui testiamo quanto aggressivamente PUNIRE chi esce dal range preferito.
        # ==============================================================================
        configs_to_run.extend([
            # "Non mi importa quanto dura, basta che sia bello"
            Config(name="G10_Time_Agnostic", alpha_cf=0.3, runtime_outside_malus=0.0),
            # "Preferirei durate simili, ma accetto eccezioni"
            Config(name="G10_Time_Flexible", alpha_cf=0.3, runtime_outside_malus=0.2),
            # "Se dura troppo/troppo poco, mi infastidisco" (Valore Default)
            Config(name="G10_Time_Standard", alpha_cf=0.3, runtime_outside_malus=0.5),
            # "Ho i minuti contati: penalizza forte chi sfora"
            Config(name="G10_Time_Strict", alpha_cf=0.3, runtime_outside_malus=2.0),
            # "Filtro Hard: Se la durata non è in target, il film sparisce"
            Config(name="G10_Time_Nazi", alpha_cf=0.3, runtime_outside_malus=10.0),
        ])

    print(f"Total Configs to Run: {len(configs_to_run)}")

    # --- EXECUTION LOOP ---
    for i, config in enumerate(configs_to_run):

        # Skip all configs except the first two (DEBUG)
        if i!=1 and i!=2:
            continue

        print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] ({i+1}/{len(configs_to_run)}) Running {config.name}...")
        
        try:
            evaluator = HybridEvaluator(config, ratings_path=data_dir / "ratings.csv", movies_path=data_dir / "movies_enriched.csv")
            results_df = evaluator.evaluate(limit_users=user_limit)
            
            if not results_df.empty:
                metrics = {
                    "mse": float(results_df['mse'].mean()), 
                    "mae": float(results_df['mae'].mean()),
                    "precision": float(results_df['precision'].mean()), 
                    "recall": float(results_df['recall'].mean()),
                    "novelty": float(results_df['novelty'].mean()), 
                    "diversity": float(results_df['diversity'].mean()),
                    "hits": float(results_df['hits'].mean())
                }
                
                print(f"   -> MSE: {metrics['mse']:.4f} | MAE: {metrics['mae']:.4f} | Prec@K: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | Nov: {metrics['novelty']:.2f} | Div: {metrics['diversity']:.2f} | Hits: {metrics['hits']:.2f}")
                
                # Aggiungiamo alla lista in memoria
                experiment_results.append({
                    "config_name": config.name,
                    "metrics": metrics,
                    "params": {k:v for k,v in config.__dict__.items() if not k.startswith('_')}
                })

                # --- SALVATAGGIO INCREMENTALE (JSON + CSV) ---
                if mode != "test":
                    try:
                        # 1. Salva JSON (Backup raw)
                        with open(json_filename, "w") as f: 
                            json.dump(experiment_results, f, indent=4)
                        
                        # 2. Crea e Salva CSV (Formattato e Pulito)
                        # json_normalize "appiattisce" metrics e params in colonne
                        df_results = pd.json_normalize(experiment_results)
                        
                        # Pulizia Nomi Colonne (Opzionale ma consigliato per leggibilità)
                        df_results.columns = df_results.columns.str.replace("metrics.", "", regex=False)
                        df_results.columns = df_results.columns.str.replace("params.", "p_", regex=False)
                        
                        # Riordino Colonne: Mettiamo Name e Metriche all'inizio
                        cols = list(df_results.columns)
                        priority_cols = ['config_name', 'mse', 'precision', 'recall', 'novelty', 'diversity']
                        # Mettiamo prima le priority (se esistono), poi il resto
                        final_order = [c for c in priority_cols if c in cols] + [c for c in cols if c not in priority_cols]
                        df_results = df_results[final_order]

                        # Scrittura su disco
                        df_results.to_csv(csv_filename, index=False)
                        
                    except Exception as save_err:
                        print(f"   -> WARNING: Could not save progress: {save_err}")

            else:
                print("   -> WARNING: No results generated (Empty DataFrame).")
                
        except Exception as e:
            print(f"   -> ERROR in {config.name}: {e}")

    print(f"\n>>> Experiment Complete. Results saved in {csv_filename} <<<")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "full"], default="full")
    args = parser.parse_args()
    run_experiment(args.mode)