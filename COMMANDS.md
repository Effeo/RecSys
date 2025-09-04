### Locale
flutter run -d web-server --web-port 8057
uvicorn main:app --reload --host 0.0.0.0 --port 8058


### Docker
chmod +x build.sh
./build.sh

### Fermare compose
Ho avuto problemi a fermare i compose con `docker compose down`. Al momento sto facendo così, per poi riavviarlo con ./build.sh
```sudo kill -9 $(docker ps -q | xargs -r docker inspect --format '{{.State.Pid}}')```