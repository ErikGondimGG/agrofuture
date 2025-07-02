# Projeto Agrofuture - Previsão de Comportamento de Empresas

## Visão Geral

Este projeto utiliza machine learning para prever quais empresas estarão vendendo em um determinado dia, baseado em dados históricos de transações e de mercado. O sistema implementa um modelo XGBoost com validação cruzada temporal e thresholds dinâmicos para previsões multi-label.

## Pré-requisitos

- Docker (versão 20.10+)
- Docker Compose (versão 2.0+)

## Estrutura de Diretórios

**text**

```
.
├── data/
│   └── raw/                  # Dados brutos (arquivos Excel)
├── outputs/
│   ├── models/               # Modelos treinados
│   ├── predictions/          # Previsões geradas
│   └── reports/              # Relatórios de avaliação
├── src/                      # Código fonte
│   ├── data_loader.py        # Carregamento de dados
│   ├── feature_engineer.py   # Engenharia de features
│   ├── model_trainer.py      # Treinamento do modelo
│   ├── run_pipeline.py       # Pipeline de treinamento
│   └── generate_predictions.py # Geração de previsões
├── start.sh                  # Script principal
└── docker-compose.yml        # Configuração Docker
```

## Como Executar

### Passo 1: Dar permissão de execução

**bash**

```
chmod +x start.sh
```

### Passo 2: Iniciar o sistema

**bash**

```
./start.sh
```

### Menu Interativo

Ao executar o script, você verá as opções:

**text**

```
Selecione o que fazer a seguir:
1) Treinar modelo de previsão
2) Gerar previsões
3) Sair
```

## Fluxo Recomendado

1. **Treinar modelo inicial:**
   - Selecione opção `1`
   - O sistema criará automaticamente:
     - Estrutura de diretórios necessária
     - Modelo treinado em `outputs/models/`
     - Relatórios de avaliação em `outputs/reports/`
2. **Gerar previsões:**
   - Selecione opção `2`
   - Informe a data no formato `YYYY-MM-DD`
   - As previsões serão salvas em `outputs/predictions/`

## Scripts Principais

### `start.sh`

- Ponto de entrada do sistema
- Gerencia containers Docker
- Oferece menu interativo
- Cria estrutura de diretórios automaticamente

### `run_pipeline.py`

Fluxo completo de treinamento:

1. Carrega dados de transações e mercado
2. Realiza engenharia de features
3. Treina modelo XGBoost com validação temporal
4. Gera relatórios de performance
5. Salva modelo treinado e thresholds

### `generate_predictions.py`

Gera previsões para datas específicas:

**bash**

```
# Uso (dentro do container)
python generate_predictions.py YYYY-MM-DD
```

Funcionalidades:

- Aceita datas históricas e futuras
- Para datas futuras, usa extrapolação de features
- Salva resultados em CSV com probabilidades por empresa

## Funcionalidades Avançadas

### Thresholds Dinâmicos

- Calcula limite de decisão ótimo para cada empresa
- Baseado na maximização do F1-score
- Armazenado em `outputs/reports/thresholds_*.json`

### Validação Cruzada Temporal

- Divisão temporal mantendo integridade das datas
- 5 folds para avaliação robusta
- Métricas reportadas:
  - F1-score
  - Precision
  - Recall

### Engenharia de Features

- Features temporais (média móvel, tendências)
- Features agregadas (por dia e por empresa)
- Percentual de participação por produto
- Spread entre preço de transação e commodities

## Exemplo de Saída

### Relatório de Treinamento

**text**

```
📌 Modelo: MultiOutputClassifier
🎯 Targets:
  - CompanyA
  - CompanyB

📈 Cross-Validation Results:
🔁 Fold 1
   F1-score : 0.8723
   Precision: 0.8541
   Recall   : 0.8912
   Thresholds por classe:
     - CompanyA: threshold = 0.4213 | f1 = 0.8821
...

🧪 Teste Final (Hold-out):
   F1-score : 0.8654
   Precision: 0.8476
   Recall   : 0.8839

🎯 Thresholds Finais:
   - CompanyA: 0.4321
   - CompanyB: 0.3876
```

### Arquivo de Previsões

`predictions_2024-11-05.csv`:

**csv**

```
Empresa,Probabilidade (%),Data,Tipo
CompanyA,95.25,2024-11-05,Futura
CompanyB,82.17,2024-11-05,Futura
```

## Solução de Problemas

**Erro: "Nenhum modelo encontrado"**

- Execute primeiro o treinamento (opção 1)
- Verifique se existem arquivos em `outputs/models/`

**Erro de permissão:**

**bash**

```
sudo chmod +x start.sh
sudo ./start.sh
```

**Docker não inicia:**

**bash**

```
docker info              # Verifique se o Docker está rodando
docker compose logs      # Verifique os logs dos containers
```

**Data não disponível:**

- Verifique o formato: deve ser `YYYY-MM-DD`
- Confira o intervalo de datas nos arquivos de dados brutos
