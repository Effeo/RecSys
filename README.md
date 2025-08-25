# Contenuti dei file

## Backend

- config.py: variabili di configurazione.

- main.py: route e logica dell'API.

- precompute.py: calcolo e memorizzazione matrice delle corrispondenze per il collaborative filtering e salva la lista dei film disponibili in json.

## Data

- movies_corr.npy: matrice delle corrispondenze precalcolata.

- movies_enriched.csv: film con informazioni aggiunte tramite dbpedia e wikidata, al momento i dati in più sono: regista, durata e premi (0 non ha ricevuto premi, 1 ha ricevuto premi).

- movies_list.json: lista dei film disponibili.

- movies.csv: dataset movielens.

- ratings.csv: dataset movielens con i voti dei film presenti in movies.csv.

- users.json: utenti dell'app.

## Experiments

- collaborative_filtering.py: codice per il collaborative filtering.

- constraint_based.py: codice per il constraint based.

- enrich_nmovies.py: codice per arricchire i film con altre informazioni prese da wikidata e dbpedita.

## Frontend
Frontend in flutter-web.

