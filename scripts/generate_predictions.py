#!/usr/bin/env python3
"""
Gera previsões para datas futuras usando modelo treinado
"""

import sys
sys.path.insert(0, '/app/src')

import pandas as pd
from pathlib import Path
import joblib

from agrofuture.data_loader import load_data, merge_data
from agrofuture.feature_engineer import create_features, prepare_target

# Configuração de paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "outputs" / "models"
PREDICTIONS_DIR = BASE_DIR / "outputs" / "predictions"

def prepare_future_date(df, target_date):
    """
    Prepara dados para uma data futura não presente no dataset
    
    Args:
        df: DataFrame completo com dados históricos
        target_date: Data futura para previsão (datetime)
    
    Returns:
        DataFrame estendido com a data futura
    """
    # Pega o último registro conhecido
    last_row = df.sort_values("date").iloc[-1].copy()
    # Atualiza a data para a data futura desejada
    last_row["date"] = target_date
    # Adiciona o novo registro ao início do DataFrame
    extended_df = pd.concat([pd.DataFrame([last_row]), df], ignore_index=True)
    return extended_df

def main(date_str):
    # Converter para datetime
    try:
        target_date = pd.to_datetime(date_str)
    except ValueError:
        print("Formato de data inválido. Use YYYY-MM-DD")
        sys.exit(1)
        
    print(f"\nGerando previsões para {date_str}...")
    

    # Carregar modelo mais recente
    model_files = sorted(MODELS_DIR.glob("xgboost_model_*.joblib"), reverse=True)
    if not model_files:
        print("Nenhum modelo encontrado em:", MODELS_DIR)
        sys.exit(1)
        
    model_file = model_files[0]
    print(f"Usando modelo: {model_file.name}")
    model = joblib.load(model_file)

    # Carregar dados
    transactions_path = BASE_DIR / "data" / "raw" / "transações-desafio.xlsx"
    commodities_path = BASE_DIR / "data" / "raw" / "mercado-desafio.xlsx"
    transacoes, mercado = load_data(transactions_path, commodities_path)
    merged_df = merge_data(transacoes, mercado)

    # Verificar se a data é futura
    last_known_date = merged_df["date"].max()
    is_future_date = target_date > last_known_date
    
    if is_future_date:
        print(f"AVISO: {target_date.date()} é uma data futura (após {last_known_date.date()})")
        print("Gerando dados sintéticos baseados no último dia conhecido...")
        
        # Estender dataset com dados sintéticos
        extended_df = prepare_future_date(merged_df, target_date)
        
        # Criar features com dados estendidos
        df_features = create_features(extended_df)
        
        # Mensagem sobre premissas
        print("\nPremissas para previsão futura:")
        print("- Valores de commodities mantidos constantes")
        print("- Features calculadas com base no histórico recente")
        print("- Tendências mantidas do último período conhecido")
    else:
        # Criar features normalmente
        df_features = create_features(merged_df)

    # Verificar se a data existe no índice
    if target_date not in df_features["date"].unique():
        print(f"\nERRO: Data {target_date.date()} não disponível após processamento")
        print(f"Datas disponíveis: {df_features['date'].min().date()} a {df_features['date'].max().date()}")
        sys.exit(1)
        
    _, company_classes = prepare_target(df_features)    
    # Isolar linha para a data desejada
    X_target = df_features[df_features["date"] == target_date].drop(columns=['empresas_vendedoras'], errors='ignore')

    # Remover coluna 'date' antes de prever
    X_target = X_target.drop(columns=['date'], errors='ignore')

    # Reordenar colunas para coincidir com o modelo
    if hasattr(model, 'feature_names_in_'):
        X_target = X_target[model.feature_names_in_]
    elif hasattr(model, 'estimators_') and hasattr(model.estimators_[0], 'feature_names_in_'):
        X_target = X_target[model.estimators_[0].feature_names_in_]

    # Fazer previsões
    probabilities = model.predict_proba(X_target)
    
    # Formatando resultados
    print(f"\nProbabilidades para {target_date.date()}:")
    # threshold = 0.95
    results = []
    for i, company in enumerate(company_classes):
        # if probabilities[i][0][1] < threshold:
        #     continue

        prob = probabilities[i][0][1] * 100  # Probabilidade em porcentagem
        
        results.append({
            'Empresa': company,
            'Probabilidade (%)': prob,
            'Data': target_date.date(),
            'Tipo': 'Futura' if is_future_date else 'Histórica'
        })
        print(f"- {company}: {prob:.2f}%")

    # Salvar resultados em CSV
    predictions_path = PREDICTIONS_DIR / f"predictions_{target_date.date()}.csv"
    pd.DataFrame(results).to_csv(predictions_path, index=False)
    print(f"\nResultados salvos em: {predictions_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python generate_predictions.py YYYY-MM-DD")
        sys.exit(1)
    
    main(sys.argv[1])