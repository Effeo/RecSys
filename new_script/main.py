import argparse
import json
import datetime
from pathlib import Path
from config import Config
from evaluator import HybridEvaluator

def run_experiment(mode: str):
    data_dir = Path("../data")
    configs_to_run = []
    experiment_results = []

    if mode == "test":
        print(">>> TEST MODE <<<")
        configs_to_run.append(Config(name="QuickTest", verbose_users=5))
        user_limit = 20
    else:
        print(">>> FULL GRID SEARCH <<<")
        user_limit = None
        
        # 1. BASELINE DETERMINISTICHE (Impact Popularity Bias)
        configs_to_run.extend([
            Config(name="Det_Pop0.2", alpha_cf=0.2, use_popularity_bias=True, popularity_weight=0.2, use_probabilistic_sampling=False),
            Config(name="Det_Pop0.5", alpha_cf=0.2, use_popularity_bias=True, popularity_weight=0.5, use_probabilistic_sampling=False), # Champion
            Config(name="Det_Pop0.8", alpha_cf=0.2, use_popularity_bias=True, popularity_weight=0.8, use_probabilistic_sampling=False),
        ])

        # 2. PROBABILISTICHE: SAFE (Top-50 Pool)
        # Verifichiamo se il sampling sui Top-50 mantiene la precisione ma alza diversity
        configs_to_run.extend([
            Config(name="Prob_Pool50_Temp0.5", alpha_cf=0.2, use_popularity_bias=True, popularity_weight=0.5, 
                   use_probabilistic_sampling=True, sampling_top_k_pool=50, temperature=0.5),
            
            Config(name="Prob_Pool50_Temp1.0", alpha_cf=0.2, use_popularity_bias=True, popularity_weight=0.5, 
                   use_probabilistic_sampling=True, sampling_top_k_pool=50, temperature=1.0),
        ])

        # 3. PROBABILISTICHE: WIDE (Top-100/200 Pool)
        # Allarghiamo il pool per vedere quando crolla la precisione
        configs_to_run.extend([
            Config(name="Prob_Pool100", alpha_cf=0.2, use_popularity_bias=True, popularity_weight=0.5, 
                   use_probabilistic_sampling=True, sampling_top_k_pool=100, temperature=0.5),
            
            Config(name="Prob_Pool200", alpha_cf=0.2, use_popularity_bias=True, popularity_weight=0.5, 
                   use_probabilistic_sampling=True, sampling_top_k_pool=200, temperature=0.5),
        ])

        # 4. STRESS TEST (Zero Malus vs High Malus)
        configs_to_run.extend([
            Config(name="Det_NoMalus", alpha_cf=0.2, use_popularity_bias=True, popularity_weight=0.5, forbidden_genre_malus=0.0),
            Config(name="Det_HighMalus", alpha_cf=0.2, use_popularity_bias=True, popularity_weight=0.5, forbidden_genre_malus=10.0),
        ])

    print(f"Total Configs: {len(configs_to_run)}")

    for i, config in enumerate(configs_to_run):
        print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Running {config.name}...")
        evaluator = HybridEvaluator(config, ratings_path=data_dir / "ratings.csv", movies_path=data_dir / "movies_enriched.csv")
        results_df = evaluator.evaluate(limit_users=user_limit)
        
        if not results_df.empty:
            metrics = {
                "mse": float(results_df['mse'].mean()), "mae": float(results_df['mae'].mean()),
                "precision": float(results_df['precision'].mean()), "recall": float(results_df['recall'].mean()),
                "novelty": float(results_df['novelty'].mean()), "diversity": float(results_df['diversity'].mean()),
                "hits": float(results_df['hits'].mean())
            }
            print(f"-> Prec: {metrics['precision']:.4f} | Nov: {metrics['novelty']:.2f} | Div: {metrics['diversity']:.2f}")
            
            experiment_results.append({
                "config": config.name,
                "metrics": metrics,
                "params": {k:v for k,v in config.__dict__.items() if not k.startswith('_')}
            })

    if mode != "test" and experiment_results:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"results_gridsearch_{ts}.json", "w") as f: json.dump(experiment_results, f, indent=4)
        print(f"\nSaved to results_gridsearch_{ts}.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "full"], default="test")
    args = parser.parse_args()
    run_experiment(args.mode)