"""
preprocess.py — Feature engineering and train/test split for the fraud dataset.

Input:  creditcard.csv  (Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
Output: data/X_train.csv, data/X_test.csv, data/y_train.csv, data/y_test.csv

Usage:
    python src/preprocess.py --input creditcard.csv --output-dir data/
"""

import argparse
import os

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows from {path}")
    fraud_rate = df["Class"].mean()
    print(f"Fraud rate: {fraud_rate:.4%} ({df['Class'].sum()} cases)")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convert seconds-since-start into hour-of-day (0–23) for periodicity
    df["Hour"] = (df["Time"] / 3600) % 24

    # RobustScaler on Amount is preferable to StandardScaler here because
    # transaction amounts have heavy tails; RobustScaler uses median/IQR
    # and is not distorted by extreme values.
    scaler = RobustScaler()
    df["scaled_amount"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))

    df.drop(["Time", "Amount"], axis=1, inplace=True)
    print("Feature engineering done: Hour created, Amount RobustScaled, raw columns dropped.")
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # StratifiedShuffleSplit preserves the fraud ratio in both splits
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_idx, test_idx in sss.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"Train: {len(X_train):,} rows | fraud rate: {y_train.mean():.4%}")
    print(f"Test:  {len(X_test):,} rows  | fraud rate: {y_test.mean():.4%}")
    return X_train, X_test, y_train, y_test


def save_splits(X_train, X_test, y_train, y_test, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    print(f"Saved 4 split files to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Preprocess creditcard.csv for fraud detection training.")
    parser.add_argument("--input", default="creditcard.csv", help="Path to raw creditcard.csv")
    parser.add_argument("--output-dir", default="data/", help="Directory to write train/test CSVs")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction reserved for test set")
    args = parser.parse_args()

    df = load_data(args.input)
    df = engineer_features(df)
    X_train, X_test, y_train, y_test = split_data(df, test_size=args.test_size)
    save_splits(X_train, X_test, y_train, y_test, args.output_dir)


if __name__ == "__main__":
    main()
