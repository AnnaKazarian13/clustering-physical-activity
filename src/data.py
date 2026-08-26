"""Загрузка Mayo Clinic PBC и приведение к схеме cirrhosis."""

from pathlib import Path
from urllib.request import urlopen

import pandas as pd

PBC_URL = (
    "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets"
    "/master/csv/survival/pbc.csv"
)
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_NAME = "pbc_raw.csv"

# Схема, в которой датасет известен по Kaggle-версии playground-series-s3e26.
COLUMNS = [
    "id", "N_Days", "Drug", "Age", "Sex", "Ascites", "Hepatomegaly", "Spiders",
    "Edema", "Bilirubin", "Cholesterol", "Albumin", "Copper", "Alk_Phos",
    "SGOT", "Tryglicerides", "Platelets", "Prothrombin", "Stage", "Status",
]

RENAME = {
    "time": "N_Days", "trt": "Drug", "age": "Age", "sex": "Sex",
    "ascites": "Ascites", "hepato": "Hepatomegaly", "spiders": "Spiders",
    "edema": "Edema", "bili": "Bilirubin", "chol": "Cholesterol",
    "albumin": "Albumin", "copper": "Copper", "alk.phos": "Alk_Phos",
    "ast": "SGOT", "trig": "Tryglicerides", "platelet": "Platelets",
    "protime": "Prothrombin", "stage": "Stage", "status": "Status",
}

# В исходнике status: 0 — наблюдение прекращено, 1 — пересадка печени, 2 — смерть.
STATUS_MAP = {0: "C", 1: "CL", 2: "D"}
DRUG_MAP = {1: "D-penicillamine", 2: "Placebo"}
SEX_MAP = {"f": "F", "m": "M"}
YN_MAP = {0: "N", 1: "Y"}
EDEMA_MAP = {0.0: "N", 0.5: "S", 1.0: "Y"}

DAYS_PER_YEAR = 365.25


def download_raw(data_dir: Path) -> Path:
    """Скачивает pbc.csv, если его ещё нет."""
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_path = data_dir / RAW_NAME
    if not raw_path.exists():
        with urlopen(PBC_URL, timeout=60) as response:
            raw_path.write_bytes(response.read())
    return raw_path


def to_cirrhosis_schema(raw: pd.DataFrame) -> pd.DataFrame:
    """Переименовывает колонки и раскодирует числовые категории в метки."""
    df = raw.rename(columns=RENAME)

    df["Status"] = df["Status"].map(STATUS_MAP)
    df["Drug"] = df["Drug"].map(DRUG_MAP)
    df["Sex"] = df["Sex"].map(SEX_MAP)
    for col in ("Ascites", "Hepatomegaly", "Spiders"):
        df[col] = df[col].map(YN_MAP)
    df["Edema"] = df["Edema"].map(EDEMA_MAP)

    # В схеме cirrhosis возраст в днях, в исходнике — в годах.
    df["Age"] = (df["Age"] * DAYS_PER_YEAR).round().astype("Int64")

    assert df["Status"].notna().all(), "остались нераскодированные Status"
    return df[COLUMNS]


def load(data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Возвращает готовый датафрейм, при необходимости скачав данные."""
    raw = pd.read_csv(download_raw(Path(data_dir)))
    return to_cirrhosis_schema(raw)


if __name__ == "__main__":
    df = load()
    print(df.shape)
    print(df["Status"].value_counts())
    print(df.isna().sum().loc[lambda s: s > 0])
