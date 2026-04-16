"""
train.py — Two-stage hybrid fraud detection pipeline.

Stage 1: Isolation Forest generates an anomaly score as an engineered feature.
Stage 2: Benchmark three models × three sample-weight strategies.
         The best configuration (XGBoost + frequency weights) is then retrained
         with SHAP analysis and threshold-tuned to a 1% operational alert cap.

Input:  data/X_train.csv, data/X_test.csv, data/y_train.csv, data/y_test.csv
        (produced by src/preprocess.py)

Output: models/champion_xgb.json          — serialised XGBoost model
        models/business_config.json        — operational threshold
        reports/shap_bar_importance.png    — SHAP global feature importance
        reports/threshold_tuning.png       — Precision/Recall vs threshold curve

Usage:
    python src/train.py
    python src/train.py --data-dir data/ --output-dir models/ --report-dir reports/
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")          # non-interactive backend; safe on headless servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_splits(data_dir: str):
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test  = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()

    print(f"Loaded splits — train: {len(X_train):,}  test: {len(X_test):,}")
    print(f"Fraud cases   — train: {y_train.sum()}  test: {y_test.sum()}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Stage 1: Isolation Forest anomaly score
# ---------------------------------------------------------------------------

def add_anomaly_scores(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Fit an Isolation Forest on training data and append the anomaly score as a
    new feature to both splits.  The score from decision_function() is
    proportional to how 'normal' a point is: lower (more negative) = more anomalous.
    contamination=0.05 is intentionally set slightly above the true 0.17% rate to
    give the forest a wider anomaly budget during unsupervised fitting.
    """
    # Amount was scaled but we drop it before fitting IF to avoid leakage
    # (Amount is already captured as scaled_amount)
    X_tr = X_train.drop("scaled_amount", axis=1, errors="ignore").copy()
    X_te = X_test.drop("scaled_amount",  axis=1, errors="ignore").copy()

    iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    iso.fit(X_tr)

    X_train = X_train.copy()
    X_test  = X_test.copy()
    X_train["anomaly_score"] = iso.decision_function(X_tr)
    X_test["anomaly_score"]  = iso.decision_function(X_te)

    print(f"Anomaly score added — train range: [{X_train['anomaly_score'].min():.3f}, "
          f"{X_train['anomaly_score'].max():.3f}]")
    return X_train, X_test


# ---------------------------------------------------------------------------
# Business metrics
# ---------------------------------------------------------------------------

def calculate_vdr_tdr(y_true, y_prob, amounts, epsilon: float = 0.01):
    """
    Value Detection Rate (VDR) and Transaction Detection Rate (TDR) at a given
    alert budget epsilon (fraction of total transactions inspected).

    VDR policy — rank by  P(fraud) × Amount  (capture the most dollar value).
    TDR policy — rank by  P(fraud)            (capture the most fraud cases).
    """
    n_inspect = max(1, int(len(y_true) * epsilon))
    total_fraud_value = np.sum(amounts[y_true == 1])
    total_fraud_count = np.sum(y_true == 1)

    # VDR: sort by expected value (probability × transaction amount)
    psi = y_prob * amounts
    vdr_threshold = np.sort(psi)[-n_inspect]
    vdr_flagged   = psi >= vdr_threshold
    vdr_score     = np.sum(amounts[(y_true == 1) & vdr_flagged]) / total_fraud_value

    # TDR: sort by raw fraud probability (equivalent to Recall@k)
    tdr_threshold = np.sort(y_prob)[-n_inspect]
    tdr_flagged   = y_prob >= tdr_threshold
    tdr_score     = np.sum((y_true == 1) & tdr_flagged) / total_fraud_count

    vdr_dollars = np.sum(amounts[(y_true == 1) & vdr_flagged])
    tdr_dollars = np.sum(amounts[(y_true == 1) & tdr_flagged])
    return vdr_score, tdr_score, vdr_dollars, tdr_dollars


def eval_at_alert_cap(y_true, y_prob, epsilon: float = 0.01):
    """Return Precision, Recall, FPR, and AUPRC at the epsilon alert cap."""
    n_inspect  = max(1, int(len(y_true) * epsilon))
    threshold  = np.sort(y_prob)[-n_inspect]
    y_pred     = (y_prob >= threshold).astype(int)

    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    fp    = np.sum((y_pred == 1) & (y_true == 0))
    fpr   = fp / max(1, np.sum(y_true == 0))
    auprc = average_precision_score(y_true, y_prob)
    return {"threshold": threshold, "precision": prec, "recall": rec,
            "fpr": fpr, "auprc": auprc}


# ---------------------------------------------------------------------------
# Stage 2: Model tournament
# ---------------------------------------------------------------------------

def get_model_configs():
    return {
        "XGBoost":      xgb.XGBClassifier(
                            n_estimators=100, learning_rate=0.1, max_depth=6,
                            eval_metric="aucpr", random_state=42, verbosity=0),
        "RandomForest": RandomForestClassifier(
                            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "LogisticReg":  LogisticRegression(
                            max_iter=1000, solver="liblinear", random_state=42),
    }


def build_weight_strategies(y_train, amounts_train):
    """
    Three sample-weight strategies that encode different business priorities.
    All strategies up-weight fraud cases; they differ in how they treat transaction size.

    Frequency  — flat class ratio weight (treats all fraud equally regardless of amount).
    RawValue   — weight = transaction amount (maximise dollar recovery).
    Balanced   — log-dampened amount × class multiplier (compromise between the two).
    """
    class_ratio = int(np.sum(y_train == 0) / np.sum(y_train == 1))
    return {
        "Frequency": np.where(y_train == 1, class_ratio, 1),
        "RawValue":  amounts_train,
        "Balanced":  np.where(y_train == 1,
                               np.log1p(amounts_train) * 500,
                               np.log1p(amounts_train)),
    }


def run_tournament(X_train, X_test, y_train, y_test, amounts_train, amounts_test):
    strategies = build_weight_strategies(y_train, amounts_train)
    results = []

    print(f"\n{'Strategy':<12} {'Model':<14} {'Recall@1%':>9} {'VDR':>8} {'AUPRC':>7}")
    print("-" * 54)

    for strat_name, weights in strategies.items():
        for model_name, model in get_model_configs().items():
            model.fit(X_train, y_train, sample_weight=weights)
            y_prob = model.predict_proba(X_test)[:, 1]

            metrics = eval_at_alert_cap(y_test, y_prob)
            vdr, tdr, vdr_d, _ = calculate_vdr_tdr(y_test, y_prob, amounts_test)

            row = {"strategy": strat_name, "model": model_name, **metrics,
                   "vdr": vdr, "vdr_dollars": vdr_d}
            results.append(row)

            print(f"{strat_name:<12} {model_name:<14} "
                  f"{metrics['recall']:>8.1%} {vdr:>7.1%} {metrics['auprc']:>7.3f}")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Champion model: SHAP + threshold tuning + serialisation
# ---------------------------------------------------------------------------

def train_champion(X_train, X_test, y_train, y_test, amounts_test,
                   output_dir: str, report_dir: str):
    """
    Retrain the winning configuration (XGBoost + Frequency weights) with SHAP
    analysis, then calibrate the operating threshold to a strict 1% alert cap.
    """
    class_ratio = int(np.sum(y_train == 0) / np.sum(y_train == 1))
    freq_weights = np.where(y_train == 1, class_ratio, 1)

    champion = xgb.XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=6,
        eval_metric="aucpr", random_state=42, verbosity=0,
    )
    champion.fit(X_train, y_train, sample_weight=freq_weights)
    y_prob = champion.predict_proba(X_test)[:, 1]

    # --- Threshold tuning at 1% alert cap ---
    metrics = eval_at_alert_cap(y_test, y_prob, epsilon=0.01)
    vdr, tdr, vdr_d, _ = calculate_vdr_tdr(y_test, y_prob, amounts_test)

    print("\n── Champion Results (XGBoost + Frequency weights) ──")
    print(f"  AUPRC:              {metrics['auprc']:.3f}")
    print(f"  Recall@1%:          {metrics['recall']:.1%}")
    print(f"  Precision@1%:       {metrics['precision']:.1%}  "
          f"(~{metrics['precision'] / 0.0017:.0f}× lift over base rate)")
    print(f"  FPR@1%:             {metrics['fpr']:.2%}")
    print(f"  VDR:                {vdr:.1%}")
    print(f"  Operating threshold: {metrics['threshold']:.4f}")

    # --- SHAP analysis (enriched sample for contrast) ---
    print("\nComputing SHAP values …")
    fraud_idx = np.where(y_test == 1)[0]
    legit_idx = np.where(y_test == 0)[0]
    sample_idx = np.concatenate([
        np.random.RandomState(42).choice(fraud_idx, min(50, len(fraud_idx)), replace=False),
        np.random.RandomState(42).choice(legit_idx, 50, replace=False),
    ])
    X_sample = X_test.iloc[sample_idx] if hasattr(X_test, "iloc") else X_test[sample_idx]

    # Use logit-margin output for SHAP so values are on an additive log-odds scale
    booster     = champion.get_booster()
    model_fn    = lambda x: booster.predict(xgb.DMatrix(x), output_margin=True)
    masker      = shap.maskers.Independent(X_train, max_samples=100)
    explainer   = shap.Explainer(model_fn, masker)
    shap_values = explainer(X_sample)

    os.makedirs(report_dir, exist_ok=True)

    plt.figure()
    shap.plots.bar(shap_values, show=False)
    bar_path = os.path.join(report_dir, "shap_bar_importance.png")
    plt.savefig(bar_path, bbox_inches="tight")
    plt.close()
    print(f"SHAP bar chart saved → {bar_path}")

    # --- Threshold tuning plot ---
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    t = metrics["threshold"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, precisions[:-1], "b-", label="Precision (Hit Rate)", linewidth=2)
    ax.plot(thresholds, recalls[:-1],    "g-", label="Recall (Capture Rate)", linewidth=2)
    ax.axvline(t, color="red", linestyle="--", alpha=0.7,
               label=f"Operational threshold ({t:.4f})")
    ax.axvspan(t, 1.0, color="red", alpha=0.1, label="Active Alert Zone (1% cap)")
    ax.annotate(f"Recall: {metrics['recall']:.1%}",
                xy=(t, metrics["recall"]), xytext=(t + 0.05, metrics["recall"] + 0.05),
                arrowprops=dict(arrowstyle="->", color="black"))
    ax.annotate(f"Precision: {metrics['precision']:.1%}",
                xy=(t, metrics["precision"]), xytext=(t + 0.05, metrics["precision"] - 0.1),
                arrowprops=dict(arrowstyle="->", color="black"))
    ax.set_xlabel("Model Probability Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Threshold Tuning: Fraud Capture vs. Operational Capacity", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    thresh_path = os.path.join(report_dir, "threshold_tuning.png")
    fig.savefig(thresh_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Threshold tuning plot saved → {thresh_path}")

    # --- Serialise model + business config ---
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "champion_xgb.json")
    champion.save_model(model_path)

    config_path = os.path.join(output_dir, "business_config.json")
    with open(config_path, "w") as f:
        json.dump({"threshold": float(metrics["threshold"]),
                   "alert_cap_pct": 1.0,
                   "recall_at_cap": float(metrics["recall"]),
                   "precision_at_cap": float(metrics["precision"]),
                   "auprc": float(metrics["auprc"]),
                   "vdr": float(vdr)}, f, indent=2)

    print(f"\nModel saved      → {model_path}")
    print(f"Business config  → {config_path}")
    return champion, metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train the two-stage fraud detection pipeline.")
    parser.add_argument("--data-dir",   default="data/",    help="Directory with train/test CSVs")
    parser.add_argument("--output-dir", default="models/",  help="Where to write model artefacts")
    parser.add_argument("--report-dir", default="reports/", help="Where to write plots")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_splits(args.data_dir)

    # The raw scaled_amount column carries transaction size — needed for VDR/TDR.
    # We keep a copy before the Isolation Forest stage modifies the DataFrames.
    amounts_train = X_train["scaled_amount"].values.copy()
    amounts_test  = X_test["scaled_amount"].values.copy()

    # Stage 1 — inject anomaly score
    print("\n── Stage 1: Isolation Forest ──")
    X_train, X_test = add_anomaly_scores(X_train, X_test)

    # Stage 2 — model tournament
    print("\n── Stage 2: Model Tournament (3 models × 3 weight strategies) ──")
    leaderboard = run_tournament(X_train, X_test, y_train, y_test,
                                 amounts_train, amounts_test)

    lb_path = os.path.join(args.report_dir, "leaderboard.csv")
    os.makedirs(args.report_dir, exist_ok=True)
    leaderboard.to_csv(lb_path, index=False)
    print(f"\nFull leaderboard saved → {lb_path}")

    # Champion — retrain, SHAP, serialise
    print("\n── Champion Model ──")
    train_champion(X_train, X_test, y_train, y_test, amounts_test,
                   args.output_dir, args.report_dir)


if __name__ == "__main__":
    main()
