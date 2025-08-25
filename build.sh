#!/bin/bash

cd frontend
flutter build web --release
cd ..

docker build -t recsys_fe frontend
docker build -t recsys_be backend

docker compose up -d


