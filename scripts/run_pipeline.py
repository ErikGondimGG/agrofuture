"""
Pipeline completo:

1. Carregamento dos dados
2. Engenharia de Features
3. Treinamento do Modelo
4. Validação e Avaliação (04/11/2024)
5. Geração de Previsões (05/11/2024)
"""

import sys 
sys.path.insert(0, '/app/src')

import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from agrofuture.data_loader import load_data, merge_data
from agrofuture.feature_engineer import create_features
from agrofuture.model_trainer import train_and_validate

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

def main():
  print(f"\n{'='*50}")
  print(f" Agrofuture - Execução em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
  print(f"{'='*50}\n")

  # Carregar dados
  print("Carregando dados...")
  transactions_path = RAW_DATA_DIR / "transações-desafio.xlsx"
  commodities_path = RAW_DATA_DIR / "mercado-desafio.xlsx"
  transacoes, mercado = load_data(transactions_path, commodities_path) 
  merged_df = merge_data( transacoes, mercado)
  merged_df["date"] = pd.to_datetime(merged_df["date"])
  merged_df = merged_df.sort_values("date")

  # Treinamento do modelo
  print("Treinando modelo...")
  model, results, thresholds = train_and_validate(merged_df)

  # salvar modelo treinado
  
  model_file = MODELS_DIR / f"xgboost_model_{datetime.now().strftime('%Y%m%d%H%M%S')}.joblib"
  joblib.dump(model, model_file)

  # salvar relatorio
  report_file = OUTPUTS_DIR / "reports" / f"relatorio_{datetime.now().strftime('%Y%m%d')}_xgboost_model_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
  with open(report_file, "w") as f:
    f.write(str(results))

  # salvar thresholds
  thresholds_file = OUTPUTS_DIR / "reports" / f"thresholds_{datetime.now().strftime('%Y%m%d')}_xgboost_model_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
  with open(thresholds_file, "w") as f:
    f.write(str(thresholds))

  # Previsões
  # print("Gerando previsões...")
  # prediction_date = pd.Timestamp("2024-11-05")
  # prediction_data = final_df[final_df["Date"] == prediction_date].copy()

  # if prediction_data.empty:
  #   raise ValueError(f"Nenhum dado encontrado para a data {prediction_date}. Verifique os dados processados.")
  
  # Fazer previsões
#   X_pred = prediction_data.drop(columns=["Date", "Company"])
#   prediction_data["probabilidade_venda"] = model.predict_proba(X_pred)[:, 1]

#   # Selecionar Top Vendedores
  # top_vendedores = prediction_data.sort_values("probabilidade_venda", ascending=False).head(100)

#   # salvar previsões
#   predictions_file = PREDICTIONS_DIR / f"previsoes_{prediction_date.strftime('%Y%m%d')}.csv"
#   top_vendedores[["Company", "probabilidade_venda"]].to_csv(predictions_file, index=False)
#   print(f"Previsões salvas em: {predictions_file}")

  # generate_report(results, top_vendedores)

# def generate_report(results, top_vendedores):
#   """
#   Gerar relatório com os resultados do pipeline.
#   """

#   report_file = OUTPUTS_DIR / "reports" / "resumo_execucao.txt"

#   with open(report_file, "w") as f:
#     f.write("="*50 + "\n")
#     f.write(" RELATÓRIO AGROFUTURE\n")
#     f.write("="*50 + "\n\n")
    
#     f.write(f"Data de execução: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n")
    
#     f.write("== Resultados de Validação (04/11/2024) ==\n")
#     f.write(f"Modelo: {results['model_name']}\n")
#     f.write(f"F1-Score: {results['f1_score']:.4f}\n")
#     f.write(f"Precision: {results['precision']:.4f}\n")
#     f.write(f"Recall: {results['recall']:.4f}\n\n")
    
#     f.write("== Top 5 Features ==\n")
#     for feat, importance in results['top_features'][:5]:
#         f.write(f"{feat}: {importance:.4f}\n")
#     f.write("\n")
    
#     f.write("== Top 10 Vendedores Previstos (05/11/2024) ==\n")
#     for i, row in enumerate(top_vendedores.head(10).itertuples()):
#         f.write(f"{i+1}. Vendedor {row.id_vendedor}: {row.probabilidade_venda:.2%}\n")


if __name__ == "__main__":
    main()