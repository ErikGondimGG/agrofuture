#!/bin/bash

# Construir a imagem Docker
docker build -t agrofuture .

# Executar o container
docker run -it --rm -p 8888:8888 -v $(pwd):/app agrofuture
