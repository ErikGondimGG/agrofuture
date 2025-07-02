#!/bin/bash

clear

set -e

# Criar diretórios necessários (se não existirem)
echo "📂 Criando diretórios necessários..."
mkdir -p data/raw
mkdir -p outputs/models
mkdir -p outputs/predictions
mkdir -p outputs/reports

echo "🐳 Construindo containers Docker..."
docker compose up --build --detach

echo "🐳 Iniciando containers Docker..."
docker compose start

clear

echo "🐳 Container iniciado com sucesso!"

echo "Selecione o que fazer a seguir:"
echo "1) Treinar modelo de previsão"
echo "2) Gerar previsões"
echo "3) Sair"
read -p "Opção [1/2/3]: " choice

while true; do
  if [ "$choice" = "1" ]; then
    clear
    echo "Treinando modelo de previsão..."
    docker compose run pipeline 
  elif [ "$choice" = "2" ]; then
    clear
    echo "Gerando previsões..."
    read -p "Data de previsão (YYYY-MM-DD): " prediction_date
    docker compose run predictions $prediction_date 
  elif [ "$choice" = "3" ]; then
    clear
    echo "Saindo..."
    break
  else
    clear
    echo "Opção inválida."
  fi

  # Remove containers órfãos a cada loop
  docker compose down --remove-orphans

  echo
  echo "Selecione o que fazer a seguir:"
  echo "1) Treinar modelo de previsão"
  echo "2) Gerar previsões"
  echo "3) Sair"
  read -p "Opção [1/2/3]: " choice
done

echo "🐳 Encerrando containers Docker..."
docker compose down --remove-orphans