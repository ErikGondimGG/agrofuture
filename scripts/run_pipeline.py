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
from typing import Dict, Any

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
  save_model_report(results, OUTPUTS_DIR)

  # salvar thresholds
  thresholds_file = OUTPUTS_DIR / "reports" / f"thresholds_{datetime.now().strftime('%Y%m%d')}_xgboost_model_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
  with open(thresholds_file, "w") as f:
    f.write(str(thresholds))

def save_model_report(report: Dict[str, Any], OUTPUTS_DIR: Path) -> Path:
    report_lines = []

    # Cabeçalho
    report_lines.append(f"📌 Modelo: {report.get('model', 'N/A')}\n")

    target_names = report.get("target_names", [])
    feature_names = report.get("feature_names", [])

    report_lines.append("🎯 Targets:")
    for t in target_names:
        report_lines.append(f" - {t}")
    report_lines.append("")

    # Cross-validation
    report_lines.append("📈 Cross-Validation Results:")
    for fold_data in report.get("cross_validation", []):
        fold = fold_data["fold"]
        f1 = fold_data["f1_score"]
        prec = fold_data["precision"]
        rec = fold_data["recall"]
        report_lines.append(f"\n🔁 Fold {fold}")
        report_lines.append(f"   F1-score : {f1:.4f}")
        report_lines.append(f"   Precision: {prec:.4f}")
        report_lines.append(f"   Recall   : {rec:.4f}")
        report_lines.append("   Thresholds por classe:")
        for cls, thr in fold_data["thresholds"].items():
            report_lines.append(f"     - {cls:<10}: threshold = {thr['value']:.4f} | f1 = {thr['f1']:.4f}")
    report_lines.append("")

    # Test performance
    test = report.get("test_performance", {})
    report_lines.append("🧪 Teste Final (Hold-out):")
    report_lines.append(f"   F1-score : {test.get('f1_score', 0):.4f}")
    report_lines.append(f"   Precision: {test.get('precision', 0):.4f}")
    report_lines.append(f"   Recall   : {test.get('recall', 0):.4f}")
    report_lines.append("")

    # Thresholds finais
    thresholds = report.get("thresholds", {})
    report_lines.append("🎯 Thresholds Finais por Classe:")
    for cls, val in thresholds.items():
        report_lines.append(f"   - {cls:<10}: {float(val):.4f}")
    report_lines.append("")

    # Feature importance
    feature_importance = report.get("feature_importances", {})
    mean_importance = feature_importance.get("mean_importance", {})
    std_importance = feature_importance.get("std_importance", {})

    report_lines.append("📊 Importância Média das Features (por classe):")
    if mean_importance:
        report_lines.append(f"\n{'Feature':<25}{'Média':>10} {'Std':>10}")
        report_lines.append("-" * 45)
        for feat in feature_names:
            mean = mean_importance.get(feat, 0)
            std = std_importance.get(feat, 0)
            report_lines.append(f"{feat:<25}{mean:>10.4f} {std:>10.4f}")
    else:
        report_lines.append("Nenhuma importância de feature disponível.")

    report_lines.append("\n✅ Fim do relatório.\n")

    # Caminho do arquivo
    now = datetime.now()
    report_file = OUTPUTS_DIR / "reports" / f"relatorio_{now.strftime('%Y%m%d')}_xgboost_model_{now.strftime('%Y%m%d%H%M%S')}.txt"
    report_file.parent.mkdir(parents=True, exist_ok=True)

    # Salvar em arquivo
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"📄 Relatório salvo em: {report_file}")
    return report_file


if __name__ == "__main__":
    main()