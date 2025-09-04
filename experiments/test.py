import random
import time

def epsilon_greedy_recommendation ( results_with_scores ,
top_k , epsilon =0.2) :
    """
    Seleziona top_k film usando epsilon - greedy . Con
    probabilit `a epsilon , sceglie a caso ;
    altrimenti , preferisce quelli con punteggi pi`u alti
    ma introduce un elemento casuale .
    """
    print(f" Applicando ␣ epsilon - greedy ␣con␣ epsilon ={epsilon }... ")

    if not results_with_scores :
        print(" Nessun ␣ risultato ␣ disponibile ␣per␣ epsilon -greedy .")
        return {}

    # Estrai la lista di film corretta
    films_list = [ list ( film . values () ) [0] for film in
    results_with_scores ] # Estrai il dizionario corretto

    # Crea una lista di tuple con il punteggio randomizzato
    randomized_scores = [(film .get(" title ", " Unknown "),
                          film , film .get(" score ", 0) + random . uniform ( -0.5 ,0.5) ) for film in films_list ]


    randomized_scores.sort(key= lambda x: x[2], reverse = True )

    print (" Risultati disponibili per epsilon - greedy (con randomizzazione ):")
    for title , film , score in randomized_scores :
        print (f"- Titolo : { title }, Punteggio randomizzato:{ score }, Motivo :{ film .get ( ' explanation ' , ' N/A ') }")
    
    selected = {}
    selected_titles = set () # Per evitare duplicati
    available = randomized_scores [:] # Copia della lista

    random.seed(time.time())

    while len(selected) < top_k and available:
        random_value = random.random ()
        print (f" Valore ␣ casuale ␣ generato :␣{ random_value }")

        if random_value < epsilon and len( available ) > 1:
            chosen = random.choice( available )
        else :
            chosen = available [0]
        
        title , film , score = chosen

        if title not in selected_titles :
            film_copy = film.copy()
            film_copy[" score "] = score 
            selected [ title ] = film_copy 

            selected_titles .add( title )

            print(f" Film scelto :{ title } con punteggio{score }, Motivo :{ film .get (' explanation ' ,  'N/A ') }")

        # Rimuove il film scelto dalla lista disponibile
        available = [ movie for movie in available if  movie [0] != title ]

    return selected