from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb
import pandas as pd
import numpy as np
import json
from typing import List, Tuple, Dict, Any
from agrofuture.feature_engineer import create_features, prepare_target
from tqdm import tqdm

def calculate_dynamic_thresholds(model, X: pd.DataFrame, y: np.ndarray, company_classes: List[str]) -> Dict[str, float]:
    """
    Calcula thresholds ótimos para cada empresa usando a curva Precision-Recall
    
    Args:
        model: Modelo treinado
        X: Features do conjunto de treino
        y: Targets do conjunto de treino
        company_classes: Lista de empresas
    
    Returns:
        Dicionário com thresholds para cada empresa
    """
    thresholds = {}
    y_proba = model.predict_proba(X)
    
    for i, company in enumerate(company_classes):
        # Obter probabilidades para a classe positiva
        pos_probs = y_proba[i][:, 1]
        
        # Calcular curva Precision-Recall
        precision, recall, thresh_vals = precision_recall_curve(y[:, i], pos_probs)
        
        # Calcular F1-score para cada threshold
        f1_scores = (2 * precision * recall) / (precision + recall + 1e-8)
        
        # Encontrar threshold que maximiza o F1-score
        best_idx = np.argmax(f1_scores)
        thresholds[company] = thresh_vals[best_idx] if best_idx < len(thresh_vals) else 0.5
    
    return thresholds

def temporal_train_test_split(X: pd.DataFrame, y: np.ndarray, test_size: float = 0.2):
    """Divisão temporal mantendo integridade das datas"""
    dates = X["date"].unique()
    dates_sorted = np.sort(dates)
    split_idx = int(len(dates_sorted) * (1 - test_size))

    train_idx = dates_sorted[:split_idx]
    test_idx = dates_sorted[split_idx:]

    X_train = X[X["date"].isin(train_idx)]
    y_train = y[X["date"].isin(train_idx)]

    X_test = X[X["date"].isin(test_idx)]
    y_test = y[X["date"].isin(test_idx)]

    return X_train, y_train, X_test, y_test

def train_and_validate(df: pd.DataFrame, test_size: float = 0.2, n_splits: int = 5) -> Tuple[Any, Dict , Dict[str, float]]:
    """
    Treina e valida modelo, retornando thresholds dinâmicos
    
    Retorna:
        tuple: (modelo, relatórios de validação, dicionário de thresholds)
    """
    # 1. Criação de features
    print("Criando features...")
    df_features = create_features(df)
    
    # 2. Preparar target multi-label
    print("Preparando target...")
    y, company_classes = prepare_target(df_features)
    
    # 3. Preparar features mantendo a data para split temporal
    X = df_features.drop(columns=['empresas_vendedoras'])
    
    # 4. Divisão temporal
    print("Dividindo dados temporalmente...")
    X_train, y_train, X_test, y_test = temporal_train_test_split(X, y, test_size)  # type: ignore
  
    # 5. Preparar dados
    X_train_no_date = X_train.drop(columns=['date'])
    X_test_no_date = X_test.drop(columns=['date'])
    
    # 6. Modelo Multi-label
    model = MultiOutputClassifier(
        xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=200,         # Increased for more boosting rounds
            max_depth=6,              # Slightly deeper trees
            learning_rate=0.01,       # Lower learning rate for finer updates
            subsample=0.9,            # More data per tree
            colsample_bytree=0.9,     # More features per tree
            reg_alpha=0.1,            # L1 regularization
            reg_lambda=1.0,           # L2 regularization
            random_state=242,
            eval_metric='logloss',
            tree_method="hist"
        ),
        n_jobs=-1
    )

    # 7. Validacao cruzada temporal
    print("Iniciando validação cruzada temporal...")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_reports = []

    for fold, (train_idx, val_idx) in enumerate(
        tqdm(tscv.split(X_train_no_date), total=n_splits, desc="\n--- Cross-validation ---"), 1):
        

        X_train_fold, X_val_fold = X_train_no_date.iloc[train_idx], X_train_no_date.iloc[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        model.fit(X_train_fold, y_train_fold)

        # Avaliar modelo
        y_pred = model.predict(X_val_fold)
        y_proba = model.predict_proba(X_val_fold)

        # Calcular métricas
        results = {
            "fold": fold,
            "f1_score": f1_score(y_val_fold, y_pred, average='micro'),
            "precision": precision_score(y_val_fold, y_pred, average='micro'),
            "recall": recall_score(y_val_fold, y_pred, average='micro'),
            "thresholds": {}
        }
        
        # Calcular thresholds para este fold
        for i, company in enumerate(company_classes):
            # Calcular curva Precision-Recall
            precision, recall, thresh_vals = precision_recall_curve(
                y_val_fold[:, i], 
                y_proba[i][:, 1]
            )
            
            # Calcular F1-score para cada threshold
            f1_scores = (2 * precision * recall) / (precision + recall + 1e-8)
            best_idx = np.argmax(f1_scores)
            
            # Armazenar melhor threshold para esta empresa
            results["thresholds"][company] = {
                "value": thresh_vals[best_idx] if best_idx < len(thresh_vals) else 0.5,
                "f1": f1_scores[best_idx]
            }
        
        fold_reports.append(results)

    # 8. Treinar modelo final com todos os dados de treino
    print("\nTreinando modelo final com todo o conjunto de treino...")
    model.fit(X_train_no_date, y_train)
    
    # 9. Calcular thresholds finais usando todo o conjunto de treino
    print("Calculando thresholds dinâmicos finais...")
    final_thresholds = calculate_dynamic_thresholds(
        model, 
        X_train_no_date, 
        y_train, 
        list(company_classes)
    )
    
    # 10. Avaliar no conjunto de teste
    print("\nAvaliando no conjunto de teste...")
    y_test_pred = model.predict(X_test_no_date)
    test_report = {
        "f1_score": f1_score(y_test, y_test_pred, average='micro'),
        "precision": precision_score(y_test, y_test_pred, average='micro'),
        "recall": recall_score(y_test, y_test_pred, average='micro'),
    }
    
    # Adicionar resultados de teste ao relatório final
    final_report = {
        "model": model.__class__.__name__,
        "target_names": company_classes,
        "feature_names": list(X_train_no_date.columns),
        "cross_validation": fold_reports,
        "test_performance": test_report,
        "thresholds": final_thresholds
    }
    
    # Calcular importância de features
    print("Calculando importância de features...")
    feature_importances = get_feature_importances(
        model, 
        company_classes, 
        list(X_train_no_date.columns)
    )
    final_report["feature_importances"] = feature_importances.to_dict()

    return model, final_report, final_thresholds


def get_feature_importances(model, company_classes, feature_names):
    """Calcula importância média das features entre todos os classificadores"""
    importance_df = pd.DataFrame(index=feature_names)
    
    for i, company in enumerate(company_classes):
        booster = model.estimators_[i].get_booster()
        imp = booster.get_score(importance_type='gain')
        
        # Preencher importâncias (features não usadas recebem 0)
        for f in feature_names:
            importance_df.loc[f, company] = imp.get(f, 0)
    
    # Calcular estatísticas
    importance_df['mean_importance'] = importance_df.mean(axis=1)
    importance_df['std_importance'] = importance_df.std(axis=1)
    importance_df['max_importance'] = importance_df.max(axis=1)
    
    return importance_df.sort_values('mean_importance', ascending=False)