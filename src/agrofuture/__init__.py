"""
AgroFuture - IA na Previsão de Negociações

Este pacote fornece funções para treinar e predizer negociações
de commodities agrícolas com base em dados históricos.
"""

from .data_loader import load_data, merge_data
from .feature_engineer import create_features, prepare_target
from .model_trainer import train_and_validate

__version__ = "0.1.0"

__all__ = [
    "load_data",
    "merge_data",
    "create_features",
    "prepare_target",
    "train_and_validate",
]