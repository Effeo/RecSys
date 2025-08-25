from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import time
import random
import re
from typing import  Tuple, Optional, Dict

# ---------------------------------
# Helper utilities
# ---------------------------------

def _clean_title_and_year(raw_title: str) -> Tuple[str, Optional[int]]:
    """Return title without trailing "(YYYY)" and the year as int if present."""
    if not isinstance(raw_title, str):
        return raw_title, None

    match = re.search(r"\((\d{4})\)\s*$", raw_title)
    year = int(match.group(1)) if match else None
    cleaned = re.sub(r"\s*\(\d{4}\)\s*$", "", raw_title).strip()
    return cleaned, year


def _sleep_with_jitter(base: float = 1.0, jitter: float = 1.0):
    """Sad but necessary – help not getting rate‑limited."""
    time.sleep(base + random.uniform(0, jitter))


# ---------------------------------
# DBpedia query
# ---------------------------------

def _query_dbpedia(title: str) -> Optional[Dict[str, any]]:
    """Return {'director': str|None, 'runtime': int|None, 'awards': List[str]} or None"""
    print(f"DBpedia -> {title}")
    query = f"""
        SELECT DISTINCT ?directorLabel ?runtimeVal ?awardLabel WHERE {{
          ?film a dbo:Film ;
                rdfs:label "{title}"@en .

          OPTIONAL {{
            ?film dbo:director ?director .
            ?director rdfs:label ?directorLabel .
            FILTER (LANG(?directorLabel) = "en")
          }}

          OPTIONAL {{ ?film dbo:runtime ?runtimeVal . }}

          OPTIONAL {{
            ?film dbo:award ?awardURI .
            ?awardURI rdfs:label ?awardLabel .
            FILTER (LANG(?awardLabel) = "en")
          }}
        }} LIMIT 25
    """

    sparql = SPARQLWrapper("https://dbpedia.org/sparql")
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()
    except Exception as exc:
        print(f"DBpedia error: {exc}")
        return None

    bindings = results.get("results", {}).get("bindings", [])
    if not bindings:
        return None

    # reduce over rows to unique values
    director = None
    runtime = None
    awards = set()

    for b in bindings:
        if not director and "directorLabel" in b:
            director = b["directorLabel"]["value"]
        if not runtime and "runtimeVal" in b:
            # runtime can be literal minutes or xsd:duration – handle numbers
            try:
                runtime = int(float(b["runtimeVal"]["value"]))
            except Exception:
                pass
        if "awardLabel" in b:
            awards.add(b["awardLabel"]["value"])

    return {
        "director": director,
        "runtime": runtime,
        "awards": sorted(awards),
    }


# ---------------------------------
# Wikidata query
# ---------------------------------

def _query_wikidata(title: str, year: Optional[int] = None) -> Optional[Dict[str, any]]:
    """Return {'director': str|None, 'runtime': int|None, 'awards': List[str]} or None"""
    print(f"Wikidata -> {title}{f' ({year})' if year else ''}")

    # language filter fallback
    year_filter = f"FILTER(YEAR(?pubdate) = {year})" if year else ""

    query = f"""
        SELECT DISTINCT ?directorLabel ?durationVal ?awardLabel WHERE {{
          ?film wdt:P31 wd:Q11424 ;          # instance of film
                rdfs:label "{title}"@en .
          OPTIONAL {{ ?film wdt:P577 ?pubdate . }}
          {year_filter}
          OPTIONAL {{
            ?film wdt:P57 ?director .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
          }}
          OPTIONAL {{
            ?film wdt:P2047 ?durationVal .  # duration, minutes
          }}
          OPTIONAL {{
            ?film wdt:P166 ?award .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
          }}
        }} LIMIT 25
    """

    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.addCustomHttpHeader("User-Agent", "MoviesEnricher/0.1 (https://example.com)")

    try:
        results = sparql.query().convert()
    except Exception as exc:
        print(f"Wikidata error: {exc}")
        return None

    bindings = results.get("results", {}).get("bindings", [])
    if not bindings:
        return None

    director = None
    runtime = None
    awards = set()

    for b in bindings:
        if not director and "directorLabel" in b:
            director = b["directorLabel"]["value"]
        if not runtime and "durationVal" in b:
            try:
                runtime = int(float(b["durationVal"]["value"]))
            except Exception:
                pass
        if "awardLabel" in b:
            awards.add(b["awardLabel"]["value"])

    return {
        "director": director,
        "runtime": runtime,
        "awards": sorted(awards),
    }


# ---------------------------------
# Main pipeline
# ---------------------------------

def enrich_movies(movies_csv: str, output_csv: str, backup_every: int = 50):
    """Read movies_csv, enrich, write output_csv."""

    df = pd.read_csv(movies_csv)

    # Ensure required columns
    if "movie_title" not in df.columns:
        raise ValueError("Expected a 'movie_title' column in the CSV")

    # Prepare new columns
    new_cols = ["director", "runtime", "awards"]
    for col in new_cols:
        if col not in df.columns:
            df[col] = pd.NA

    for idx, row in df.iterrows():
        raw_title = row["movie_title"]
        clean_title, year = _clean_title_and_year(raw_title)

        # Skip if already enriched
        if pd.notna(row["director"]) and pd.notna(row["runtime"]) and pd.notna(row["awards"]):
            continue

        # 1 DBpedia first
        data = _query_dbpedia(clean_title)

        # 2️ Wikidata fallback if needed
        need_fallback = (
            data is None or
            (data.get("director") is None and data.get("runtime") is None and not data.get("awards"))
        )
        if need_fallback:
            data = _query_wikidata(clean_title, year) or data

        if data:
            df.at[idx, "director"] = data.get("director")
            df.at[idx, "runtime"] = data.get("runtime")
            awards = data.get("awards")
            df.at[idx, "awards"] = "; ".join(awards) if awards else pd.NA

        # Replace title with cleaned version (no year)
        df.at[idx, "movie_title"] = clean_title

        # Periodic backup
        if idx % backup_every == 0 and idx != 0:
            df.to_csv(output_csv, index=False)
            print(f"Backup saved at row {idx} → {output_csv}")

    # Final save
    df.to_csv(output_csv, index=False)
    print(f"Enriched CSV saved to {output_csv}")


if __name__ == "__main__":

    file_input = "data/movies.csv"
    file_output = "movies_enriched.csv"

    enrich_movies(file_input, file_output)
