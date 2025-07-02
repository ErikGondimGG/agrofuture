# AgroFuture - IA na Previsão de Negociações


## Visão Geral

Este projeto tem como objetivo prever quais empresas estarão vendendo em um determinado dia, baseado em dados históricos de transações e de mercado. O sistema utiliza aprendizado de máquina com XGBoost para fazer previsões multi-label, onde cada empresa é tratada como uma classe binária independente.

## Pré-requisitos

Antes de começar, certifique-se de ter instalado:

* Docker (versão 20.10+)
* Docker Compose (versão 2.0+)

## Passo 1: Iniciar o Sistema

1. **Clone o repositório** (se aplicável):
   **bash**

   ```
   git clone <https://github.com/ErikGondimGG/agrofuture.git>
   cd <agrofuture>
   ```
2. **Dê permissão de execução ao script** :
   **bash**

```
   chmod +x start.sh
```

1. **Execute o script** :
   **bash**

```
   ./start.sh
```

Se os containers não iniciarem:

* Verifique se o Docker está rodando: `docker info`
* Verifique os logs: `docker compose logs`


### Passo 2: Selecionar ação

Após iniciar, o sistema apresentará um menu com as opções:

**text**

```
Selecione o que fazer a seguir:
1) Treinar modelo de previsão
2) Gerar previsões
3) Sair
```

### Fluxo Recomendados

1. **Primeira execução:**
   * Selecione a opção `1` para treinar o modelo inicial
   * O sistema criará automaticamente a estrutura de diretórios necessária
   * O modelo treinado será salvo em `outputs/models/`
2. **Gerar previsões:**
   * Selecione a opção `2` e informe a data desejada no formato `YYYY-MM-DD`
   * As previsões serão salvas em `outputs/predictions/`

## Estrutura do Projeto

### Diretórios Principais

**text**

```
data/
└── raw/              # Dados brutos (arquivos Excel)
outputs/
├── models/           # Modelos treinados (arquivos .joblib)
├── predictions/      # Previsões geradas (arquivos CSV)
└── reports/          # Relatórios de treinamento e avaliação
scripts/ 	      # Scripts de interacao
src/                  # Código fonte
```

### Scripts Principais

1. **`start.sh`**
   * Ponto de entrada do sistema
   * Gerencia containers Docker e menu interativo
2. **`run_pipeline.py`**
   * Orquestra o fluxo completo de treinamento:
     1. Carregamento de dados
     2. Engenharia de features
     3. Treinamento do modelo
     4. Geração de relatórios
3. **`generate_predictions.py`**
   * Gera previsões para uma data específica
   * Aceita tanto datas históricas quanto futuras
4. **`data_loader.py`**
   * Carrega e combina dados de transações e mercado
   * Obtém valores de dólar em tempo real via API
5. **`feature_engineer.py`**
   * Cria features temporais e agregadas
   * Prepara o target multi-label
6. **`model_trainer.py`**
   * Treina modelo XGBoost com validação cruzada temporal
   * Calcula thresholds dinâmicos por empresa
   * Gera relatórios de performance

## Fluxo de Dados

```
graph LR
A[Dados Brutos] --> B[Carregamento]
B --> C[Engenharia de Features]
C --> D[Treinamento do Modelo]
D --> E[Modelo Treinado]
E --> F[Geração de Previsões]
F --> G[Relatórios e Resultados]
```

## Funcionalidades Chave

### Para Datas Futuras

Ao gerar previsões para datas futuras, o sistema:

1. Usa o último dia conhecido para criar dados sintéticos
2. Mantém constantes os valores de commodities (CBOT e dólar)
3. Calcula features com base no histórico recente
4. Mantém tendências do último período conhecido

### Thresholds Dinâmicos

* Calcula limite de decisão ótimo para cada empresa
* Baseado na maximização do F1-score
* Armazenado em arquivos JSON em `outputs/reports/`

### Validação Cruzada Temporal

* Divisão temporal dos dados mantendo integridade das datas
* 5 folds para avaliação robusta do modelo
* Métricas: F1-score, Precision e Recall

## Saídas do Sistema

1. **Modelos Treinados**
   * Formatos: `.joblib`
   * Localização: `outputs/models/`
   * Nomenclatura: `xgboost_model_<TIMESTAMP>.joblib`
2. **Previsões**
   * Formatos: `.csv`
   * Localização: `outputs/predictions/`
   * Estrutura:| Empresa | Probabilidade (%) | Data       | Tipo   |
     | ------- | ----------------- | ---------- | ------ |
     | EmpA    | 95.25             | 2024-11-05 | Futura |
3. **Relatórios**
   * Formatos: `.txt` e `.json`
   * Localização: `outputs/reports/`
   * Conteúdo:
     * Performance por fold de validação
     * Métricas no conjunto de teste
     * Thresholds por empresa
     * Importância de features

## Solução de Problemas Comuns

**Erro: "Nenhum modelo encontrado"**

* Certifique-se que executou o treinamento (opção 1) antes de gerar previsões
* Verifique o diretório `outputs/models/`

**Erro de permissão:**

**bash**

```
chmod +x start.sh
./start.sh
```

**Docker não inicia:**

* Verifique se o Docker está rodando: `docker info`
* Verifique os logs: `docker compose logs`

**Data não disponível:**

* Verifique o formato: deve ser `YYYY-MM-DD`
* Confira o intervalo de datas disponíveis nos dados brutos
