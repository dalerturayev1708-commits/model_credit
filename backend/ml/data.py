import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

CACHE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(CACHE_DIR, "credit_card_data.csv")

# Metadata mapping for all features in UCI Dataset 350
FEATURE_METADATA: List[Dict[str, Any]] = [
    {
        "id": "X1",
        "name": "LIMIT_BAL",
        "label": "Credit Limit (NT$)",
        "label_uz": "Kredit limiti (NT$)",
        "type": "numeric",
        "min": 10000,
        "max": 1000000,
        "default": 140000,
        "step": 10000,
        "description": "Amount of given credit in NT dollars (includes individual and family/supplementary credit)."
    },
    {
        "id": "X2",
        "name": "SEX",
        "label": "Gender",
        "label_uz": "Jinsi",
        "type": "categorical",
        "default": 2,
        "options": [
            {"value": 1, "label": "Male (Erkak)"},
            {"value": 2, "label": "Female (Ayol)"}
        ],
        "description": "1 = Male, 2 = Female."
    },
    {
        "id": "X3",
        "name": "EDUCATION",
        "label": "Education Level",
        "label_uz": "Ma'lumoti darajasi",
        "type": "categorical",
        "default": 2,
        "options": [
            {"value": 1, "label": "Graduate School (Magistratura/Doktorantura)"},
            {"value": 2, "label": "University (Oliy ma'lumotli / Bakalavr)"},
            {"value": 3, "label": "High School (O'rta ma'lumotli)"},
            {"value": 4, "label": "Others (Boshqa)"}
        ],
        "description": "1 = Graduate school, 2 = University, 3 = High school, 4 = Others."
    },
    {
        "id": "X4",
        "name": "MARRIAGE",
        "label": "Marital Status",
        "label_uz": "Oilaviy holati",
        "type": "categorical",
        "default": 2,
        "options": [
            {"value": 1, "label": "Married (Oilali)"},
            {"value": 2, "label": "Single (Bo'ydoq / Turmush qurmagan)"},
            {"value": 3, "label": "Others (Boshqa / Ajrashgan)"}
        ],
        "description": "1 = Married, 2 = Single, 3 = Others."
    },
    {
        "id": "X5",
        "name": "AGE",
        "label": "Age (Years)",
        "label_uz": "Yoshi (Yil)",
        "type": "numeric",
        "min": 21,
        "max": 79,
        "default": 35,
        "step": 1,
        "description": "Age in years (21 to 79)."
    },
    {
        "id": "X6",
        "name": "PAY_0",
        "label": "Payment Status - Sep",
        "label_uz": "To'lov holati - Sentabr",
        "type": "categorical",
        "default": 0,
        "options": [
            {"value": -1, "label": "Pay duly (-1: O'z vaqtida to'langan)"},
            {"value": 0, "label": "Revolving credit (0: Kredit hisobida)"},
            {"value": 1, "label": "1 month delay (1: 1 oy kechiktirilgan)"},
            {"value": 2, "label": "2 months delay (2: 2 oy kechiktirilgan)"},
            {"value": 3, "label": "3 months delay (3: 3 oy kechiktirilgan)"}
        ],
        "description": "Repayment status in September 2005."
    },
    {
        "id": "X7",
        "name": "PAY_2",
        "label": "Payment Status - Aug",
        "label_uz": "To'lov holati - Avgust",
        "type": "categorical",
        "default": 0,
        "options": [
            {"value": -1, "label": "Pay duly (-1: O'z vaqtida to'langan)"},
            {"value": 0, "label": "Revolving credit (0: Kredit hisobida)"},
            {"value": 1, "label": "1 month delay (1: 1 oy kechiktirilgan)"},
            {"value": 2, "label": "2 months delay (2: 2 oy kechiktirilgan)"}
        ],
        "description": "Repayment status in August 2005."
    },
    {
        "id": "X8",
        "name": "PAY_3",
        "label": "Payment Status - Jul",
        "label_uz": "To'lov holati - Iyul",
        "type": "categorical",
        "default": 0,
        "options": [
            {"value": -1, "label": "Pay duly (-1)"},
            {"value": 0, "label": "Revolving credit (0)"},
            {"value": 1, "label": "1 month delay (1)"},
            {"value": 2, "label": "2 months delay (2)"}
        ],
        "description": "Repayment status in July 2005."
    },
    {
        "id": "X9",
        "name": "PAY_4",
        "label": "Payment Status - Jun",
        "label_uz": "To'lov holati - Iyun",
        "type": "categorical",
        "default": 0,
        "options": [
            {"value": -1, "label": "Pay duly (-1)"},
            {"value": 0, "label": "Revolving credit (0)"},
            {"value": 1, "label": "1 month delay (1)"},
            {"value": 2, "label": "2 months delay (2)"}
        ],
        "description": "Repayment status in June 2005."
    },
    {
        "id": "X10",
        "name": "PAY_5",
        "label": "Payment Status - May",
        "label_uz": "To'lov holati - May",
        "type": "categorical",
        "default": 0,
        "options": [
            {"value": -1, "label": "Pay duly (-1)"},
            {"value": 0, "label": "Revolving credit (0)"},
            {"value": 1, "label": "1 month delay (1)"},
            {"value": 2, "label": "2 months delay (2)"}
        ],
        "description": "Repayment status in May 2005."
    },
    {
        "id": "X11",
        "name": "PAY_6",
        "label": "Payment Status - Apr",
        "label_uz": "To'lov holati - Aprel",
        "type": "categorical",
        "default": 0,
        "options": [
            {"value": -1, "label": "Pay duly (-1)"},
            {"value": 0, "label": "Revolving credit (0)"},
            {"value": 1, "label": "1 month delay (1)"},
            {"value": 2, "label": "2 months delay (2)"}
        ],
        "description": "Repayment status in April 2005."
    },
    {
        "id": "X12",
        "name": "BILL_AMT1",
        "label": "Bill Amount - Sep (NT$)",
        "label_uz": "Hisob varaqasi - Sentabr (NT$)",
        "type": "numeric",
        "min": -50000,
        "max": 1000000,
        "default": 47000,
        "step": 1000,
        "description": "Amount of bill statement in September 2005."
    },
    {
        "id": "X13",
        "name": "BILL_AMT2",
        "label": "Bill Amount - Aug (NT$)",
        "label_uz": "Hisob varaqasi - Avgust (NT$)",
        "type": "numeric",
        "min": -50000,
        "max": 1000000,
        "default": 45000,
        "step": 1000,
        "description": "Amount of bill statement in August 2005."
    },
    {
        "id": "X14",
        "name": "BILL_AMT3",
        "label": "Bill Amount - Jul (NT$)",
        "label_uz": "Hisob varaqasi - Iyul (NT$)",
        "type": "numeric",
        "min": -50000,
        "max": 1000000,
        "default": 42000,
        "step": 1000,
        "description": "Amount of bill statement in July 2005."
    },
    {
        "id": "X15",
        "name": "BILL_AMT4",
        "label": "Bill Amount - Jun (NT$)",
        "label_uz": "Hisob varaqasi - Iyun (NT$)",
        "type": "numeric",
        "min": -50000,
        "max": 1000000,
        "default": 40000,
        "step": 1000,
        "description": "Amount of bill statement in June 2005."
    },
    {
        "id": "X16",
        "name": "BILL_AMT5",
        "label": "Bill Amount - May (NT$)",
        "label_uz": "Hisob varaqasi - May (NT$)",
        "type": "numeric",
        "min": -50000,
        "max": 1000000,
        "default": 38000,
        "step": 1000,
        "description": "Amount of bill statement in May 2005."
    },
    {
        "id": "X17",
        "name": "BILL_AMT6",
        "label": "Bill Amount - Apr (NT$)",
        "label_uz": "Hisob varaqasi - Aprel (NT$)",
        "type": "numeric",
        "min": -50000,
        "max": 1000000,
        "default": 36000,
        "step": 1000,
        "description": "Amount of bill statement in April 2005."
    },
    {
        "id": "X18",
        "name": "PAY_AMT1",
        "label": "Previous Payment - Sep (NT$)",
        "label_uz": "Oldingi to'lov - Sentabr (NT$)",
        "type": "numeric",
        "min": 0,
        "max": 500000,
        "default": 5000,
        "step": 500,
        "description": "Amount of previous payment in September 2005."
    },
    {
        "id": "X19",
        "name": "PAY_AMT2",
        "label": "Previous Payment - Aug (NT$)",
        "label_uz": "Oldingi to'lov - Avgust (NT$)",
        "type": "numeric",
        "min": 0,
        "max": 500000,
        "default": 4500,
        "step": 500,
        "description": "Amount of previous payment in August 2005."
    },
    {
        "id": "X20",
        "name": "PAY_AMT3",
        "label": "Previous Payment - Jul (NT$)",
        "label_uz": "Oldingi to'lov - Iyul (NT$)",
        "type": "numeric",
        "min": 0,
        "max": 500000,
        "default": 4000,
        "step": 500,
        "description": "Amount of previous payment in July 2005."
    },
    {
        "id": "X21",
        "name": "PAY_AMT4",
        "label": "Previous Payment - Jun (NT$)",
        "label_uz": "Oldingi to'lov - Iyun (NT$)",
        "type": "numeric",
        "min": 0,
        "max": 500000,
        "default": 3500,
        "step": 500,
        "description": "Amount of previous payment in June 2005."
    },
    {
        "id": "X22",
        "name": "PAY_AMT5",
        "label": "Previous Payment - May (NT$)",
        "label_uz": "Oldingi to'lov - May (NT$)",
        "type": "numeric",
        "min": 0,
        "max": 500000,
        "default": 3000,
        "step": 500,
        "description": "Amount of previous payment in May 2005."
    },
    {
        "id": "X23",
        "name": "PAY_AMT6",
        "label": "Previous Payment - Apr (NT$)",
        "label_uz": "Oldingi to'lov - Aprel (NT$)",
        "type": "numeric",
        "min": 0,
        "max": 500000,
        "default": 3000,
        "step": 500,
        "description": "Amount of previous payment in April 2005."
    }
]

FEATURE_IDS = [f["id"] for f in FEATURE_METADATA]
FEATURE_NAME_MAP = {f["id"]: f["name"] for f in FEATURE_METADATA}
NAME_TO_ID_MAP = {f["name"]: f["id"] for f in FEATURE_METADATA}

# In-memory cached dataset
_cached_X: pd.DataFrame = None
_cached_y: pd.Series = None


def load_uci_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads the real UCI Default of Credit Card Clients dataset (id=350).
    Uses ucimlrepo and caches locally for high performance and offline resilience.
    Drops any 'ID' column, checks and handles missing values.
    """
    global _cached_X, _cached_y

    if _cached_X is not None and _cached_y is not None:
        return _cached_X.copy(), _cached_y.copy()

    # Check if local CSV backup exists
    if os.path.exists(CACHE_FILE):
        try:
            df = pd.read_csv(CACHE_FILE)
            target_col = "Y" if "Y" in df.columns else df.columns[-1]
            y = df[target_col].astype(int)
            X = df.drop(columns=[target_col])
            if "ID" in X.columns:
                X = X.drop(columns=["ID"])
            if "id" in X.columns:
                X = X.drop(columns=["id"])
            # Ensure correct feature order
            X = X[FEATURE_IDS]
            _cached_X = X
            _cached_y = y
            return _cached_X.copy(), _cached_y.copy()
        except Exception:
            pass

    # Fetch from UCI Machine Learning Repository
    try:
        dataset = fetch_ucirepo(id=350)
        X = dataset.data.features.copy()
        y = dataset.data.targets.iloc[:, 0].copy()

        # Remove ID column if present
        for col in ["ID", "id", "Id"]:
            if col in X.columns:
                X = X.drop(columns=[col])

        # Check and handle missing values
        if X.isnull().sum().sum() > 0:
            X = X.fillna(X.median())
        if y.isnull().sum() > 0:
            y = y.fillna(y.mode()[0])

        y = y.astype(int)

        # Standardize column names if needed
        # Ensure columns are X1..X23
        if list(X.columns) != FEATURE_IDS:
            if set(FEATURE_IDS).issubset(set(X.columns)):
                X = X[FEATURE_IDS]
            elif set(NAME_TO_ID_MAP.keys()).issubset(set(X.columns)):
                X = X.rename(columns=NAME_TO_ID_MAP)[FEATURE_IDS]

        # Save to local cache file for offline resilience
        full_df = X.copy()
        full_df["Y"] = y
        full_df.to_csv(CACHE_FILE, index=False)

        _cached_X = X
        _cached_y = y
        return _cached_X.copy(), _cached_y.copy()

    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch UCI dataset 350. Please ensure internet connectivity or retry. Error: {str(e)}"
        )


def get_dataset_split() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Returns train and test split with test_size=0.2, random_state=42, and stratify=y.
    Same split is used across all models to ensure fair comparison.
    """
    X, y = load_uci_data()
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def get_dataset_summary() -> Dict[str, Any]:
    """
    Returns comprehensive dataset summary statistics, feature information,
    and target distribution.
    """
    X, y = load_uci_data()

    total_rows = int(len(X))
    total_features = int(X.shape[1])

    non_default_count = int((y == 0).sum())
    default_count = int((y == 1).sum())

    missing_counts = {col: int(X[col].isnull().sum()) for col in X.columns}

    stats = {}
    for col in X.columns:
        s = X[col]
        stats[col] = {
            "mean": round(float(s.mean()), 2),
            "std": round(float(s.std()), 2),
            "min": float(s.min()),
            "max": float(s.max()),
            "median": float(s.median())
        }

    return {
        "dataset_name": "UCI Default of Credit Card Clients (ID: 350)",
        "source_url": "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
        "total_rows": total_rows,
        "total_features": total_features,
        "target_column": "Y (default payment next month)",
        "target_explanation": {
            "0": "Non-Default (Mijoz kredit to'lovlarini o'z vaqtida bajargan)",
            "1": "Default (Mijoz keyingi oy kredit to'lovini kechiktirgan / to'lay olmagan)"
        },
        "target_distribution": {
            "non_default_0": non_default_count,
            "default_1": default_count,
            "non_default_pct": round((non_default_count / total_rows) * 100, 2),
            "default_pct": round((default_count / total_rows) * 100, 2),
        },
        "missing_values_total": sum(missing_counts.values()),
        "missing_values_by_feature": missing_counts,
        "features": FEATURE_METADATA,
        "statistics": stats
    }
