import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path(__file__).resolve().parent / "G1.csv"
MODEL_PATH = Path(__file__).resolve().parent / "random_forest_model.pkl"


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates().copy()
    df = df.replace(["", " ", "nan", "NaN", "None", "NULL"], np.nan)
    df = df.drop(columns=["Response_ID", "Timestamp"], errors="ignore")

    df["Best_Shopping_Mode"] = df["Best_Shopping_Mode"].astype(str).str.strip()
    df["Best_Shopping_Mode"] = df["Best_Shopping_Mode"].replace({
        "Equal / both": "Both",
        "Equal mix": "Both",
    })
    df = df[df["Best_Shopping_Mode"].isin(["Online", "Offline", "Both"])].copy()

    target_map = {"Online": 0, "Offline": 1, "Both": 2}
    df["Best_Shopping_Mode"] = df["Best_Shopping_Mode"].map(target_map)

    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Age"] = df["Age"].fillna(df["Age"].median())

    X = df.drop(columns=["Best_Shopping_Mode"])
    y = df["Best_Shopping_Mode"]

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(X[col].median())

    for col in categorical_cols:
        X[col] = X[col].fillna("Missing")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42)),
    ])

    model.fit(X_train, y_train)

    artifact = {
        "model": model,
        "feature_order": list(X.columns),
        "label_map": {0: "Online", 1: "Offline", 2: "Both"},
    }

    with MODEL_PATH.open("wb") as f:
        pickle.dump(artifact, f)

    print(f"Model saved to: {MODEL_PATH}")
    print("Training accuracy:", model.score(X_train, y_train))
    print("Test accuracy:", model.score(X_test, y_test))


if __name__ == "__main__":
    main()
