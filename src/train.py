"""Мультиклассовая классификация исхода (Status) на датасете Mayo Clinic PBC."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier, plot_importance

from data import load

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
CLASSES = ["C", "CL", "D"]
TARGET_MAP = {"C": 0, "CL": 1, "D": 2}

# Все категории здесь либо бинарные, либо упорядоченные (Edema: нет < слабый < есть),
# поэтому целочисленные коды не навязывают ложного порядка. Словари фиксированы,
# так что кодировка одинакова на любой выборке.
ENCODINGS = {
    "Drug": {"D-penicillamine": 0, "Placebo": 1},
    "Sex": {"F": 0, "M": 1},
    "Ascites": {"N": 0, "Y": 1},
    "Hepatomegaly": {"N": 0, "Y": 1},
    "Spiders": {"N": 0, "Y": 1},
    "Edema": {"N": 0, "S": 1, "Y": 2},
}

PARAMS = dict(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=3,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42,
)

N_SPLITS = 5


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Кодирует категории и отделяет таргет. Пропуски не заполняются: XGBoost
    обрабатывает NaN сам, а в этом датасете пропуск сам по себе информативен
    (106 пациентов не входили в рандомизированное испытание)."""
    y = df["Status"].map(TARGET_MAP).to_numpy()
    X = df.drop(columns=["id", "Status"]).copy()
    for col, mapping in ENCODINGS.items():
        unknown = set(X[col].dropna().unique()) - set(mapping)
        assert not unknown, f"неизвестные категории в {col}: {unknown}"
        X[col] = X[col].map(mapping)
    return X.astype("float64"), y


def cross_validate(X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Возвращает out-of-fold вероятности по стратифицированным фолдам."""
    oof = np.zeros((len(y), len(CLASSES)))
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(folds.split(X, y), start=1):
        model = XGBClassifier(**PARAMS)
        model.fit(X.iloc[train_idx], y[train_idx])
        proba = model.predict_proba(X.iloc[val_idx]).astype("float64")
        # XGBoost отдаёт float32, после приведения суммы отходят от единицы
        # на ~1e-7, и log_loss на это ругается.
        oof[val_idx] = proba / proba.sum(axis=1, keepdims=True)
        fold_loss = log_loss(y[val_idx], oof[val_idx], labels=[0, 1, 2])
        print(f"  фолд {fold}: log loss = {fold_loss:.4f}")
    return oof


def baseline_probabilities(y: np.ndarray) -> np.ndarray:
    """Вероятности тривиальной модели, предсказывающей априорное распределение."""
    dummy = DummyClassifier(strategy="prior").fit(np.zeros((len(y), 1)), y)
    return dummy.predict_proba(np.zeros((len(y), 1)))


def report_metrics(name: str, y: np.ndarray, proba: np.ndarray) -> dict:
    pred = proba.argmax(axis=1)
    metrics = {
        "log loss": log_loss(y, proba, labels=[0, 1, 2]),
        "accuracy": accuracy_score(y, pred),
        "balanced accuracy": balanced_accuracy_score(y, pred),
        "macro F1": f1_score(y, pred, average="macro"),
    }
    print(f"\n{name}")
    for key, value in metrics.items():
        print(f"  {key:18} {value:.4f}")
    return metrics


def save_plots(X: pd.DataFrame, y: np.ndarray, oof: np.ndarray) -> None:
    REPORTS.mkdir(exist_ok=True)

    model = XGBClassifier(**PARAMS).fit(X, y)
    # plot_importance создаёт свои оси, поэтому фигуру заводим через subplots
    # и передаём ax — иначе рядом останется пустая фигура.
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_importance(model, ax=ax, max_num_features=12, importance_type="gain",
                    show_values=False, title="Важность признаков (gain)")
    fig.tight_layout()
    fig.savefig(REPORTS / "feature_importance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y, oof.argmax(axis=1), display_labels=CLASSES, ax=ax, colorbar=False,
    )
    ax.set_title("Out-of-fold confusion matrix")
    fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def main() -> None:
    # Бэкенд переключаем только при запуске скриптом, чтобы импорт из ноутбука
    # не ломал inline-графики.
    matplotlib.use("Agg")

    df = load()
    X, y = build_features(df)
    print(f"Датасет: {X.shape[0]} пациентов, {X.shape[1]} признаков")
    print("Распределение классов:",
          dict(zip(CLASSES, np.bincount(y, minlength=3).tolist())))
    print(f"Доля пропусков: {X.isna().to_numpy().mean():.1%}")

    print(f"\nStratified {N_SPLITS}-fold CV:")
    oof = cross_validate(X, y)

    report_metrics("XGBoost (out-of-fold)", y, oof)
    report_metrics("Baseline (априорные вероятности)", y, baseline_probabilities(y))

    print("\nClassification report (out-of-fold):")
    print(classification_report(y, oof.argmax(axis=1), target_names=CLASSES,
                                zero_division=0))

    save_plots(X, y, oof)
    print(f"Графики сохранены в {REPORTS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
