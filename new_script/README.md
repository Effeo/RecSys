# Documentazione Tecnica: Hybrid Recommender System

## 1. Panoramica Architetturale
Il sistema è un **Hybrid Recommender System** che combina due approcci distinti:
1.  **Constraint-Based (Content-Based)**: Applica regole rigide e punteggi basati sui metadati dei film (genere, durata, anno, regista) confrontati con il profilo dell'utente.
2.  **Collaborative Filtering (Item-Item SVD)**: Utilizza la decomposizione a valori singolari (SVD) sui residui dei rating per trovare similarità latenti tra i film.

I punteggi vengono combinati linearmente e trasformati in probabilità di visione tramite una funzione Softmax con *temperature scaling*.

---

## 2. Profiling Utente (`profiling.py`)
Il modulo `profiling.py` è responsabile della creazione di un "modello mentale" delle preferenze dell'utente basandosi sul suo storico (`user_train`).

### Logica di Selezione
Il sistema divide lo storico in due gruppi:
* **Liked Items**: Rating $\ge 4.0$. (Se non esistono rating $\ge 4.0$, viene usato l'intero storico come fallback).
* **Disliked Items**: Rating $\le 2.0$.

### Estrazione delle Preferenze 
Dallo storico filtrato vengono derivati i seguenti attributi:

1.  **Generi Desiderati (`generi_desiderati`)**:
    * Somma delle occorrenze dei generi nei *Liked Items*.
    * Vengono selezionati i **top 3** generi più frequenti.
2.  **Generi Vietati (`generi_vietati`)**:
    * Somma delle occorrenze dei generi nei *Disliked Items*.
    * Vengono selezionati i **top 3**, a condizione che **non** siano presenti tra i *Generi Desiderati* (per evitare contraddizioni).
3.  **Durata Preferita (`preferred_runtime`)**:
    * Media aritmetica del campo `runtime` dei *Liked Items*.
4.  **Registi Preferiti (`favorite_directors`)**:
    * Vengono estratti i registi che appaiono **più di una volta** nei film piaciuti (top 3 per frequenza).
    * Se nessun regista appare più di una volta, viene preso il regista del film più piaciuto.
5.  **Anno Minimo (`min_release_year`)**:
    * Calcolato come il **10° percentile** degli anni di uscita dei *Liked Items*. Questo serve a evitare di raccomandare film troppo vecchi se l'utente preferisce opere moderne (default a 1980 se dati mancanti).

---

## 3. Logica di Training e SVD (`evaluator.py`)
Il cuore collaborativo del sistema si basa su una matrice Utente-Oggetto (URM) normalizzata.

### Normalizzazione dei Rating
Prima di applicare la SVD, i rating vengono "puliti" dai bias:
$$R_{norm} = R_{ui} - \mu - b_i - b_u$$
Dove:
* $\mu$: Media globale dei rating.
* $b_i$: Bias dell'oggetto (Item), calcolato con shrinking factor ($C=5$).
* $b_u$: Bias dell'utente, calcolato sui residui dopo la rimozione del bias oggetto.

### Truncated SVD
Sulla matrice dei residui $R_{norm}$ (dove i valori mancanti sono riempiti a 0), viene applicata la **Truncated SVD**:
$$X \approx U \Sigma V^T \rightarrow Z$$
* **Componenti**: Configurabile via `config.svd_components` (default 30).
* **Output**: Una matrice ridotta $Z$ che rappresenta i film nello spazio latente.

### Matrice di Similarità
Viene calcolata la matrice di correlazione di Pearson tra i vettori latenti dei film:
`self.item_corr_matrix = np.corrcoef(Z)`

---

## 4. Logica di Scoring (`evaluator.py`)

La predizione per un utente avviene in `predict_user` combinando due score.


### A. Constraint Score (Content-Based)
Il sistema parte da un punteggio base di 0.0 per ogni film e applica bonus/malus:

| Criterio | Azione | Peso (Config) |
| :--- | :--- | :--- |
| **Genere Desiderato** | Aggiunge peso per ogni genere matchato | +1.0 (implicito) |
| **Genere Vietato** | Sottrae un forte malus se presente | `-forbidden_genre_malus` (default 5.0) |
| **Premi (Awards)** | Bonus se il film ha vinto premi | `+award_weight` (0.5) |
| **Regista** | Bonus se il regista è nei preferiti | `+director_weight` (0.5) |

#### Logica Runtime (Durata)
Viene definita una "zona di tolleranza" (default $\pm 15$ minuti rispetto alla media preferita):
* **Dentro la tolleranza**: Bonus `+runtime_weight` (0.3).
* **Fuori tolleranza**: Malus proporzionale alla distanza, clippato a un massimo di 3 volte il malus base.
    $$Malus = \min\left(3, \frac{|runtime - target| - tol}{tol}\right) \times weight$$

#### Logica Anno (Decadimento)
Se l'anno del film è inferiore al `min_release_year` dell'utente:
* Malus lineare per ogni anno di distanza: `diff * year_below_malus_per_year`.

### B. Collaborative Score (Seed-Based)
Per generare lo score collaborativo, il sistema **non** usa l'intero vettore utente, ma un approccio basato su un **Seed Movie**:
1.  Identifica il film con il rating più alto nello storico di training dell'utente (`top_rated`).
2.  Recupera la riga corrispondente a quel film nella `item_corr_matrix` (similitudine item-item).
3.  Il vettore di similarità diventa il `cf_score`.

### C. Ibridazione
I due punteggi vengono fusi usando un parametro $\alpha$ (`alpha_cf`):
$$Score_{Final} = Score_{Constraint} + (\alpha \times Score_{CF})$$

### D. Probabilità (Softmax)
I film già visti vengono mascherati. Sugli score rimanenti viene applicata la Softmax con temperatura:
$$P(i) = \frac{e^{s_i / T}}{\sum e^{s_j / T}}$$
* Una temperatura $T$ alta (es. 2.0) appiattisce la distribuzione (più esplorazione).
* Una temperatura bassa rende la scelta quasi deterministica (solo il top score).

---

## 5. Valutazione e Metriche (`evaluator.py`)

Il metodo `evaluate` esegue una simulazione offline.

### Strategia di Split
Viene utilizzato uno **Stratified Split**:
1.  Identifica gli utenti con almeno 5 rating ("valid users").
2.  Per questi utenti, divide i rating in 80% Train e 20% Test, mantenendo la proporzione per utente.
3.  Gli utenti con meno di 5 rating vengono messi interamente nel Train (cold start protection).

### Metriche Calcolate
Per ogni utente nel test set, vengono confrontati i Top-K film raccomandati (ordinati per probabilità) con i film effettivamente visti e piaciuti (Rating $\ge 3.0$ nel Test Set).

1.  **Hits**: Numero di film raccomandati presenti nel test set positivo.
2.  **Precision@K**:
    $$\frac{Hits}{K}$$
3.  **Recall@K**:
    $$\frac{Hits}{|Test_{pos}|}$$
4.  **MSE (Mean Squared Error) sulle Probabilità**:
    Confronta la distribuzione di probabilità predetta con una distribuzione "ideale" (dove i film visti hanno probabilità $1/N_{visti}$ e gli altri 0).
    *Nota: Questa è una metrica non standard per ranking, specifica di questa implementazione per valutare la calibrazione della Softmax.*

---

## 6. Configurazione (`config.py`)

Tutti i parametri sono centralizzati nella dataclass `Config`. I più impattanti sono:
* `alpha_cf`: Bilancia Constraint vs Collaborative (0.5 = bilanciato).
* `shrink_term`: Riduce l'impatto dei bias su pochi dati (Smoothing bayesiano).
* `temperature`: Controlla la "fiducia" della Softmax.
* `forbidden_genre_malus`: Quanto penalizzare i generi odiati (fondamentale per il filtraggio negativo).


Ecco il **README.md** completo e definitivo.

Ho integrato la sezione tecnica sull'**Implicit Knowledge Base** all'interno della descrizione architetturale, fondendo la spiegazione del funzionamento dei vincoli con la logica deduttiva che abbiamo discusso.

-----

# Hybrid Recommender System

## Panoramica del Progetto

Questo progetto implementa un sistema di raccomandazione ibrido progettato per combinare la **precisione dei vincoli logici** (Constraint-Based) con la **serendipity del filtraggio collaborativo** (Latent Factor Model).

Il sistema è progettato per superare i limiti dei classici approcci vettoriali, introducendo un motore di inferenza che costruisce a runtime una "conoscenza" del profilo utente per applicare filtri intelligenti su generi, durata e recenza dei contenuti.
