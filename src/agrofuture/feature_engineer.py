import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def create_features(df: pd.DataFrame) -> pd.DataFrame:
  """
  Cria features temporais a partir dos dados
  """

  vendas_por_dia = (
        df[df['amount'] > 0]
        .groupby('date')['company_transacoes']
        .apply(list)
        .reset_index(name='empresas_vendedoras')
    )
  
  df_merged = df.groupby('date').agg(
      total_vendas=('amount', 'sum'),
      avg_preco=('price_transacoes', 'mean'),
      produtos_negociados=('product', 'nunique'),
      qtd_transacoes=('amount', 'count')
  ).reset_index()

  df_merged['dia_semana'] = df_merged['date'].dt.dayofweek
  df_merged['mes'] = df_merged['date'].dt.month
  df_merged['trimestre'] = df_merged['date'].dt.quarter

  df_merged = pd.merge(df_merged, vendas_por_dia, on='date', how='left') 

  company_features = []
  for date in df_merged['date']:
    historico = df[df['date'] < date]

    features = {}
    for company in df['company_transacoes'].unique():
      company_df = historico[historico['company_transacoes'] == company]
      prefixo = f"comp_{company}_"

      features.update({
          f"{prefixo}media_30d": company_df["amount"].tail(30).mean(),
          f"{prefixo}frequencia": company_df["date"].nunique(),
          f"{prefixo}ultima_venda": (date - company_df["date"].max()).days if not company_df.empty else 365,
      })

    company_features.append(features)

  company_df = pd.DataFrame(company_features)
  df_merged = pd.concat([df_merged, company_df], axis=1)

  return df_merged

# Transformar target para formato multi-label
def prepare_target(df_merged: pd.DataFrame):
    mlb = MultiLabelBinarizer(sparse_output=False)
    y = mlb.fit_transform(df_merged['empresas_vendedoras'])

    return y, mlb.classes_