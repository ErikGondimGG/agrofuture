from pathlib import Path
import pandas as pd
import requests

def load_data(transacoes_path: Path, mercado_path: Path) -> tuple:
  transacoes = pd.read_excel(transacoes_path)
  mercado = pd.read_excel(mercado_path)
  return transacoes, mercado

def merge_data(transacoes: pd.DataFrame, mercado: pd.DataFrame) -> pd.DataFrame:
    """
    Indexes(['date', 'time', 'company_transacoes', 'seller id', 'buyer id', 'price_transacoes',
         'amount', 'product', 'origin_city', 'origin_state', 'company_mercado',
         'destination_city', 'destination_state', 'price_mercado', 'cbot', 'dolar'],
        dtype='object')
    """

    # Padroniza os nomes das colunas para garantir correspondência, ignorando maiúsculas/minúsculas
    transacoes.columns = [col.lower() for col in transacoes.columns]
    mercado.columns = [col.lower() for col in mercado.columns]

    #substituir por dados reais
    # Obtém o valor real do dólar usando uma API ou fonte confiável
    def get_real_dolar():
        try:
            response = requests.get("https://economia.awesomeapi.com.br/json/last/USD-BRL")
            response.raise_for_status()
            data = response.json()
            return float(data["USDBRL"]["bid"])
        except Exception:
            # fallback para um valor padrão caso a API falhe
            return 5.0

    dolar_real = get_real_dolar()
    mercado['dolar'] = mercado['cbot'] * dolar_real


    merged = pd.merge(
        transacoes,
        mercado,
        on=["date", "product", "origin_city", "origin_state"],
        how="left",
        suffixes=("_transacoes", "_mercado")
    )

    # print(merged.columns)

    return merged