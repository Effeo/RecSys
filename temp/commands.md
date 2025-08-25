# Comandi per docker-compose

## Spostarsi nella directory 
```sh
cd res-actai/demo
```
## Per eseguire la demo in locale 

#### Terminale 1
Modificare il path per mostrare i file pdf in controller.py
```sh
cd be
EXPORT GEMINI_API_KEY="..."
python -m uvicorn src.main:app --reload --host 127.0.0.1 --port 7777
```
#### Terminale 2
Modificare il baseURL in services/api con localhost
```sh
cd fe
npm run build
npm start
```

## Per buildare e runnare docker-compose
```sh
./script.sh
```

## Per stoppare docker-compose
```sh
docker-compose down
```

## Per controllare log del compose
```sh
docker-compose logs -f
```