"""Мультиклассовая классификация исхода (Status) на датасете Mayo Clinic PBC."""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
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
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from data import load

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
CLASSES = ["C", "CL", "D"]
TARGET_MAP = {"C": 0, "CL": 1, "D": 2}

# N_Days — это время до события, то есть часть исхода, а не исходный признак:
# цензурирование наступает в момент закрытия исследования, поэтому у пациентов
# со статусом C наблюдение длинное по построению. Подробности в README.
LEAKY_COLUMNS = ["N_Days"]

# Все категории либо бинарные, либо упорядоченные (Edema: нет < слабый < есть),
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

N_SPLITS = 5
SEEDS = [42, 43, 44, 45, 46]

CAT_PARAMS = dict(iterations=900, learning_rate=0.02, depth=4, l2_leaf_reg=10.0,
                  loss_function="MultiClass", verbose=0, allow_writing_files=False)
XGB_PARAMS = dict(n_estimators=400, learning_rate=0.03, max_depth=3,
                  min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=2.0, objective="multi:softprob", num_class=3,
                  eval_metric="mlogloss")


class BalancedMixin:
    """Взвешивает классы обратно их частоте."""

    def fit(self, X, y, **kw):
        return super().fit(X, y, sample_weight=compute_sample_weight("balanced", y), **kw)


class BalancedCat(BalancedMixin, CatBoostClassifier):
    pass


class BalancedXGB(BalancedMixin, XGBClassifier):
    pass


def make_model(name: str, balanced: bool, seed: int):
    if name == "cat":
        cls = BalancedCat if balanced else CatBoostClassifier
        return cls(random_seed=seed, **CAT_PARAMS)
    cls = BalancedXGB if balanced else XGBClassifier
    return cls(random_state=seed, **XGB_PARAMS)


def build_features(df: pd.DataFrame, with_ndays: bool) -> tuple[pd.DataFrame, np.ndarray]:
    """Кодирует категории и отделяет таргет. Пропуски не заполняются: и XGBoost,
    и CatBoost обрабатывают NaN сами, а в этом датасете пропуск информативен —
    106 пациентов не входили в рандомизированное испытание."""
    y = df["Status"].map(TARGET_MAP).to_numpy()
    X = df.drop(columns=["id", "Status"]).copy()
    if not with_ndays:
        X = X.drop(columns=LEAKY_COLUMNS)
    for col, mapping in ENCODINGS.items():
        unknown = set(X[col].dropna().unique()) - set(mapping)
        assert not unknown, f"неизвестные категории в {col}: {unknown}"
        X[col] = X[col].map(mapping)
    return X.astype("float64"), y


def cross_validate(model_name: str, balanced: bool, X: pd.DataFrame,
                   y: np.ndarray, n_repeats: int) -> list[np.ndarray]:
    """Out-of-fold вероятности отдельно по каждому повтору стратифицированной CV."""
    oofs = []
    for seed in SEEDS[:n_repeats]:
        oof = np.zeros((len(y), len(CLASSES)))
        folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for train_idx, val_idx in folds.split(X, y):
            model = make_model(model_name, balanced, seed)
            model.fit(X.iloc[train_idx], y[train_idx])
            proba = np.asarray(model.predict_proba(X.iloc[val_idx]), dtype="float64")
            # Обе библиотеки отдают float32, суммы отходят от единицы на ~1e-7.
            oof[val_idx] = proba / proba.sum(axis=1, keepdims=True)
        oofs.append(oof)
        print(f"  повтор seed={seed}: log loss = "
              f"{log_loss(y, oof, labels=[0, 1, 2]):.4f}")
    return oofs


def metrics_frame(y: np.ndarray, oofs: list[np.ndarray]) -> pd.DataFrame:
    rows = []
    for oof in oofs:
        pred = oof.argmax(axis=1)
        rows.append({
            "log loss": log_loss(y, oof, labels=[0, 1, 2]),
            "accuracy": accuracy_score(y, pred),
            "balanced accuracy": balanced_accuracy_score(y, pred),
            "macro F1": f1_score(y, pred, average="macro"),
        })
    return pd.DataFrame(rows)


def report_metrics(name: str, y: np.ndarray, oofs: list[np.ndarray]) -> None:
    frame = metrics_frame(y, oofs)
    print(f"\n{name}")
    for col in frame.columns:
        spread = f" +- {frame[col].std():.4f}" if len(frame) > 1 else ""
        print(f"  {col:18} {frame[col].mean():.4f}{spread}")


def baseline_oof(y: np.ndarray) -> list[np.ndarray]:
    zeros = np.zeros((len(y), 1))
    dummy = DummyClassifier(strategy="prior").fit(zeros, y)
    return [dummy.predict_proba(zeros)]


def save_plots(model_name: str, balanced: bool, X: pd.DataFrame, y: np.ndarray,
               oof: np.ndarray, suffix: str) -> None:
    REPORTS.mkdir(exist_ok=True)
    model = make_model(model_name, balanced, SEEDS[0]).fit(X, y)

    importance = pd.Series(model.feature_importances_, index=X.columns)
    importance = importance.sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance.index, importance.to_numpy())
    ax.set_xlabel("Важность признака")
    ax.set_title(f"Важность признаков ({model_name})")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS / f"feature_importance{suffix}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y, oof.argmax(axis=1), display_labels=CLASSES, ax=ax, colorbar=False,
    )
    ax.set_title("Out-of-fold confusion matrix")
    fig.tight_layout()
    fig.savefig(REPORTS / f"confusion_matrix{suffix}.png", dpi=150)
    plt.close(fig)


def save_leakage_plot(df: pd.DataFrame) -> None:
    """Показывает, почему N_Days нельзя использовать как признак."""
    REPORTS.mkdir(exist_ok=True)
    quartile = pd.qcut(df["N_Days"], 4,
                       labels=["Q1\nкороткое", "Q2", "Q3", "Q4\nдлинное"])
    share = pd.crosstab(quartile, df["Status"], normalize="index")[CLASSES] * 100

    fig, ax = plt.subplots(figsize=(7, 4.5))
    share.plot(kind="bar", stacked=True, ax=ax, rot=0,
               color=["#4c72b0", "#dd8452", "#55a868"])
    ax.set_xlabel("Квартиль N_Days (длительность наблюдения)")
    ax.set_ylabel("Доля исходов, %")
    ax.set_title("Исход почти определяется длительностью наблюдения")
    ax.legend(title="Status", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(REPORTS / "ndays_leakage.png", dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["cat", "xgb"], default="cat")
    parser.add_argument("--with-ndays", action="store_true",
                        help="включить N_Days — постановка как на Kaggle, метрика "
                             "лучше, но признак недоступен на момент прогноза")
    parser.add_argument("--balanced", action="store_true",
                        help="взвесить классы: выше balanced accuracy, хуже log loss")
    parser.add_argument("--repeats", type=int, default=len(SEEDS),
                        choices=range(1, len(SEEDS) + 1))
    return parser.parse_args()


def main() -> None:
    # Бэкенд переключаем только при запуске скриптом, чтобы импорт из ноутбука
    # не ломал inline-графики.
    matplotlib.use("Agg")
    args = parse_args()

    df = load()
    X, y = build_features(df, with_ndays=args.with_ndays)
    setting = "с N_Days (как на Kaggle)" if args.with_ndays else "без N_Days (прогностическая)"
    print(f"Датасет: {X.shape[0]} пациентов, {X.shape[1]} признаков")
    print(f"Постановка: {setting}")
    print(f"Модель: {args.model}{' + балансировка классов' if args.balanced else ''}")
    print("Распределение классов:",
          dict(zip(CLASSES, np.bincount(y, minlength=3).tolist())))
    print(f"Доля пропусков: {X.isna().to_numpy().mean():.1%}")

    print(f"\nStratified {N_SPLITS}-fold CV, {args.repeats} повтор(ов):")
    oofs = cross_validate(args.model, args.balanced, X, y, args.repeats)

    report_metrics(f"{args.model} (out-of-fold)", y, oofs)
    report_metrics("Baseline (априорные вероятности)", y, baseline_oof(y))

    print("\nClassification report (первый повтор):")
    print(classification_report(y, oofs[0].argmax(axis=1), target_names=CLASSES,
                                zero_division=0))

    suffix = "_with_ndays" if args.with_ndays else ""
    save_plots(args.model, args.balanced, X, y, oofs[0], suffix)
    save_leakage_plot(df)
    print(f"Графики сохранены в {REPORTS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
