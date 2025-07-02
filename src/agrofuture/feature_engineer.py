from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['company_transacoes', 'date'])

    # ===============================
    # 1. Features globais por dia
    # ===============================
    vendas_por_dia = (
        df[df['amount'] > 0]
        .groupby('date')['company_transacoes']
        .apply(list)
        .reset_index(name='empresas_vendedoras')
    )

    df_diario = (
        df.groupby('date').agg(
            total_vendas=('amount', 'sum'),
            avg_preco=('price_transacoes', 'mean'),
            produtos_negociados=('product', 'nunique'),
            qtd_transacoes=('amount', 'count')
        )
        .reset_index()
    )

    df_diario['dia_semana'] = df_diario['date'].dt.dayofweek
    df_diario['mes'] = df_diario['date'].dt.month
    df_diario['trimestre'] = df_diario['date'].dt.quarter

    produto_pct_df = add_produto_pct_features(df, top_n=5)
    num_estados_origem = df.groupby('date')['origin_state'].nunique().reset_index(name='num_estados_origem')
    preco_cb_spread = df.assign(spread=df['price_transacoes'] - df['cbot']).groupby('date')['spread'].mean().reset_index(name='preco_cb_spread')
    num_rotas_unicas = (
        df.groupby('date')[['origin_city', 'destination_city']]
        .apply(lambda x: x.drop_duplicates().shape[0])
        .reset_index(name='num_rotas_unicas')
    )
    peso_medio_transacao = df.groupby('date')['amount'].mean().reset_index(name='peso_medio_transacao')

    df_diario = (
        df_diario
        .merge(vendas_por_dia, on='date', how='left')
        .merge(produto_pct_df, on='date', how='left')
        .merge(num_estados_origem, on='date', how='left')
        .merge(preco_cb_spread, on='date', how='left')
        .merge(num_rotas_unicas, on='date', how='left')
        .merge(peso_medio_transacao, on='date', how='left')
    )

    # ===============================
    # 2. Features por empresa (rolling e agregadas)
    # ===============================
    df['vendeu'] = df['amount'] > 0

    rolling_window = 30
    df['vendas_30d'] = df.groupby('company_transacoes')['amount'].transform(lambda x: x.rolling(rolling_window, min_periods=1).sum())
    df['media_30d'] = df.groupby('company_transacoes')['amount'].transform(lambda x: x.rolling(rolling_window, min_periods=1).mean())
    df['desvio_30d'] = df.groupby('company_transacoes')['amount'].transform(lambda x: x.rolling(rolling_window, min_periods=1).std().fillna(0))
    df['media_7d'] = df.groupby('company_transacoes')['amount'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df['tendencia'] = df['media_7d'] - df['media_30d']
    df['dias_desde_ultima_venda'] = df.groupby('company_transacoes')['date'].transform(lambda x: (x.max() - x).dt.days)

    df_empresa_agg = df.groupby('date').agg(
        media_vendas_30d=('media_30d', 'mean'),
        desvio_vendas_30d=('desvio_30d', 'mean'),
        tendencia_media=('tendencia', 'mean'),
        dias_ult_venda_media=('dias_desde_ultima_venda', 'mean'),
    ).reset_index()

    # ===============================
    # 3. Features específicas por empresa
    # ===============================
    empresas = df['company_transacoes'].unique()
    for empresa in empresas:
        nome = empresa.lower()
        vendas_valor = df[df['company_transacoes'] == empresa].groupby('date')['amount'].sum().reset_index(name=f'total_vendas_{nome}')
        df_diario = df_diario.merge(vendas_valor, on='date', how='left')
        df_diario[f'{nome}_vendeu_ontem'] = add_empresa_vendeu_ontem(df_diario, empresa)
        df_diario[f'dias_desde_ultima_venda_{nome}'] = add_dias_desde_ultima_venda_empresa(df_diario, empresa)
        df_diario[f'{nome}_freq_ultimos_7d'] = add_freq_ultimos_7d(df_diario, empresa)
        df_diario[f'{nome}_media_vendas_7d'] = df_diario[f'total_vendas_{nome}'].rolling(window=7, min_periods=1).mean().fillna(0)
        df_diario[f'{nome}_domina_vendas'] = (df_diario[f'total_vendas_{nome}'] / df_diario['total_vendas']).fillna(0).apply(lambda x: int(x > 0.5))

    df_final = (
        df_diario
        .merge(df_empresa_agg, on='date', how='left')
        .sort_values('date')
        .reset_index(drop=True)
    )

    # Garante que todas as datas entre min e max estejam presentes, inclusive datas futuras
    full_range = pd.date_range(df_final['date'].min(), df_final['date'].max(), freq='D')
    df_final = df_final.set_index('date').reindex(full_range).reset_index().rename(columns={'index': 'date'})

    return df_final

def prepare_target(df_merged: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    df_merged['empresas_vendedoras'] = df_merged['empresas_vendedoras'].apply(lambda x: x if isinstance(x, list) else [])
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df_merged['empresas_vendedoras'])
    return y, mlb.classes_ # type: ignore

def add_produto_pct_features(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    top_produtos = df['product'].value_counts().nlargest(top_n).index.tolist()
    produto_pct = (
        df.groupby(['date', 'product'])['amount'].count()
        .unstack(fill_value=0)
        .apply(lambda x: x / x.sum(), axis=1)
        .reset_index()
    )
    produto_pct = produto_pct.rename(columns={p: f'pct_produto_{p}' for p in top_produtos if p in produto_pct.columns})
    cols = ['date'] + [f'pct_produto_{p}' for p in top_produtos if f'pct_produto_{p}' in produto_pct.columns]
    return produto_pct[cols]

def add_empresa_vendeu_ontem(df: pd.DataFrame, empresa: str) -> pd.Series:
    df_sorted = df.sort_values('date').copy()
    flag = df_sorted['empresas_vendedoras'].apply(lambda x: empresa in x)
    return flag.shift(1).fillna(False).astype(bool).astype(int)

def add_dias_desde_ultima_venda_empresa(df: pd.DataFrame, empresa: str) -> pd.Series:
    df_sorted = df.sort_values('date').copy()
    last_seen = -1
    counter = []
    for vendeu in df_sorted['empresas_vendedoras'].apply(lambda x: empresa in x):
        if vendeu:
            last_seen = 0
        elif last_seen >= 0:
            last_seen += 1
        else:
            last_seen = -1
        counter.append(last_seen if last_seen >= 0 else np.nan)
    return pd.Series(counter, index=df_sorted.index).ffill().fillna(-1)

def add_freq_ultimos_7d(df: pd.DataFrame, empresa: str) -> pd.Series:
    df_sorted = df.sort_values('date').copy()
    flags = df_sorted['empresas_vendedoras'].apply(lambda x: empresa in x).astype(int)
    return flags.rolling(window=7, min_periods=1).sum()
