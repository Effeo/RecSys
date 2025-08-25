#!/usr/bin/env python3
"""
Binarizza la colonna 'awards' (o 'premio') di un CSV:
    1  → quando il film ha almeno un premio elencato
    0  → quando non ha premi (celle vuote o NaN)

Uso da terminale:
    python binarize_awards.py --input movies_enriched.csv --output movies_awards_binary.csv
"""

import pandas as pd


def binarize_awards(input_csv: str, output_csv: str, column: str = "awards") -> None:
    df = pd.read_csv(input_csv)

    if column not in df.columns:
        raise ValueError(
            f"Colonna '{column}' non trovata nel file. "
            "Specificane un’altra con --column se necessario."
        )

    # 1 se testo non vuoto / non NaN, altrimenti 0
    df[column] = df[column].apply(lambda x: 0 if pd.isna(x) or str(x).strip() == "" else 1)

    df.to_csv(output_csv, index=False)
    print(f"✔ File salvato in: {output_csv}")


def main() -> None:
    
    binarize_awards("movies_enriched.csv", "movies_enriched2.csv")


if __name__ == "__main__":
    main()
