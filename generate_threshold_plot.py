"""
generate_threshold_plot.py
--------------------------
Standalone script to reproduce the threshold tuning plot from notebook 03.
No Azure, no MLflow, no SHAP required.

Usage:
    python generate_threshold_plot.py --data path/to/creditcard.csv
    python generate_threshold_plot.py --data path/to/creditcard.csv --two-panel

Output:
    threshold_tuning_plot.png         — zoomed operating region (0–10%)
    threshold_tuning_plot_twopanel.png — full range + zoom side-by-side (--two-panel)
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use('Agg')

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
import xgboost as xgb


# ── 1. Args ─────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, help="Path to creditcard.csv")
parser.add_argument("--two-panel", action="store_true",
                    help="Also save a full-range + zoomed two-panel figure")
args = parser.parse_args()


# ── 2. Load & preprocess  (mirrors notebook 02) ─────────────────────────────

print("Loading data...")
df = pd.read_csv(args.data)

# Feature engineering
df['Hour'] = df['Time'].apply(lambda x: (x / 3600) % 24)
robust_scaler = RobustScaler()
df['scaled_amount'] = robust_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df.drop(['Time', 'Amount'], axis=1, inplace=True)

X = df.drop('Class', axis=1)
y = df['Class']

# Stratified 80/20 split — same seed as notebook
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in sss.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print(f"Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")
print(f"Fraud rate — train: {y_train.mean():.3%}  test: {y_test.mean():.3%}")


# ── 3. IsolationForest anomaly score  (mirrors notebook 03, cell 5) ─────────

print("Training IsolationForest...")
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(X_train)
X_train = X_train.copy()
X_test  = X_test.copy()
X_train['anomaly_score'] = iso_forest.decision_function(X_train)
X_test['anomaly_score']  = iso_forest.decision_function(X_test)


# ── 4. XGBoost champion model  (mirrors notebook 03, cell 12) ───────────────

print("Training XGBoost champion model...")
freq_weights = np.where(y_train == 1, 578, 1)

best_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    eval_metric='aucpr',
    random_state=42
)
best_model.fit(X_train, y_train, sample_weight=freq_weights)


# ── 5. Business threshold tuning  (mirrors notebook 03, cell 14) ────────────

y_probs = best_model.predict_proba(X_test)[:, 1]

alert_rate_goal    = 0.01
business_threshold = np.percentile(y_probs, 100 * (1 - alert_rate_goal))

y_pred_business = (y_probs >= business_threshold).astype(int)
prec = precision_score(y_test, y_pred_business)
rec  = recall_score(y_test, y_pred_business)

print(f"\n--- Business-Constraint Tuning (1% Alert Cap) ---")
print(f"Operational Threshold : {business_threshold:.4f}")
print(f"Precision (Hit Rate)  : {prec:.2%}")
print(f"Recall (Fraud Captured): {rec:.2%}")


# ── 6. Plot  (mirrors notebook 03, cell 15) ──────────────────────────────────

print("\nGenerating plot...")
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

n_total          = len(y_probs)
alert_rates      = np.array([np.sum(y_probs >= t) / n_total for t in thresholds])
business_alert_rate = np.sum(y_probs >= business_threshold) / n_total

# Reverse so x-axis reads low → high alert rate (left to right)
alert_rates_plot = alert_rates[::-1]
precisions_plot  = precisions[:-1][::-1]
recalls_plot     = recalls[:-1][::-1]

# Look up curve values at operating point for accurate arrow tips
op_prec = float(np.interp(business_alert_rate, alert_rates_plot, precisions_plot))
op_rec  = float(np.interp(business_alert_rate, alert_rates_plot, recalls_plot))

plt.figure(figsize=(10, 6))
plt.plot(alert_rates_plot, precisions_plot, "b-", label="Precision (Hit Rate)",  linewidth=2)
plt.plot(alert_rates_plot, recalls_plot,    "g-", label="Recall (Capture Rate)", linewidth=2)

plt.axvline(business_alert_rate, color="red", linestyle="--", alpha=0.7,
            label=f"Operating Point ({business_alert_rate:.1%} alert rate)")
plt.axvspan(0, business_alert_rate, color='red', alpha=0.1,
            label='Active Alert Zone (≤1% cap)')

plt.xlabel("Alert Rate (% of transactions flagged)", fontsize=12)
plt.ylabel("Score (0.0 - 1.0)", fontsize=12)
plt.title("Threshold Tuning: Balancing Fraud Capture vs. Operational Capacity", fontsize=14)
plt.legend(loc="center right", frameon=True)
plt.grid(True, linestyle=':', alpha=0.6)
plt.xlim([0, 0.10])
plt.ylim([0, 1.05])
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

plt.annotate(f'Recall: {rec:.1%}',
             xy=(business_alert_rate, op_rec),
             xytext=(business_alert_rate + 0.005, op_rec + 0.05),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
plt.annotate(f'Precision: {prec:.1%}',
             xy=(business_alert_rate, op_prec),
             xytext=(business_alert_rate + 0.005, op_prec - 0.10),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

out = "threshold_tuning_plot.png"
plt.tight_layout()
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved → {out}")


# ── 7. Two-panel figure (optional) ──────────────────────────────────────────

if args.two_panel:
    print("Generating two-panel figure...")

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 6),
                                            gridspec_kw={'wspace': 0.35})

    # ── Left: full 0–100% ───────────────────────────────────────────────────
    ax_full.plot(alert_rates_plot, precisions_plot, "b-", linewidth=2,
                 label="Precision (Hit Rate)")
    ax_full.plot(alert_rates_plot, recalls_plot, "g-", linewidth=2,
                 label="Recall (Capture Rate)")
    ax_full.axvline(business_alert_rate, color="red", linestyle="--", alpha=0.7,
                    label=f"Operating Point ({business_alert_rate:.1%} cap)")

    # Highlight the zoomed region with an orange box
    zoom_x = 0.10
    rect = mpatches.FancyBboxPatch(
        (0, 0), zoom_x, 1.05,
        boxstyle="square,pad=0", linewidth=2,
        edgecolor="orange", facecolor="orange", alpha=0.12, zorder=3
    )
    ax_full.add_patch(rect)
    ax_full.text(zoom_x / 2, 0.35, "Zoomed\nregion →",
                 ha='center', fontsize=9, color='darkorange', fontweight='bold')

    ax_full.set_xlim([0, 1])
    ax_full.set_ylim([0, 1.05])
    ax_full.set_xlabel("Alert Rate (% of transactions flagged)", fontsize=11)
    ax_full.set_ylabel("Score (0.0 – 1.0)", fontsize=11)
    ax_full.set_title("Full Range (0–100%)", fontsize=13)
    ax_full.legend(loc="center right", fontsize=9, frameon=True)
    ax_full.grid(True, linestyle=':', alpha=0.6)
    ax_full.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

    # ── Right: zoomed 0–10% ─────────────────────────────────────────────────
    ax_zoom.plot(alert_rates_plot, precisions_plot, "b-", linewidth=2,
                 label="Precision (Hit Rate)")
    ax_zoom.plot(alert_rates_plot, recalls_plot, "g-", linewidth=2,
                 label="Recall (Capture Rate)")
    ax_zoom.axvline(business_alert_rate, color="red", linestyle="--", alpha=0.7,
                    label=f"Operating Point ({business_alert_rate:.1%} alert rate)")
    ax_zoom.axvspan(0, business_alert_rate, color='red', alpha=0.10,
                    label='Active Alert Zone (≤1% cap)')

    ax_zoom.annotate(f'Recall: {rec:.1%}',
                     xy=(business_alert_rate, op_rec),
                     xytext=(business_alert_rate + 0.012, op_rec + 0.04),
                     fontsize=10,
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
    ax_zoom.annotate(f'Precision: {prec:.1%}',
                     xy=(business_alert_rate, op_prec),
                     xytext=(business_alert_rate + 0.012, op_prec - 0.10),
                     fontsize=10,
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    ax_zoom.set_xlim([0, 0.10])
    ax_zoom.set_ylim([0, 1.05])
    ax_zoom.set_xlabel("Alert Rate (% of transactions flagged)", fontsize=11)
    ax_zoom.set_title("Operating Region (0–10%)", fontsize=13)
    ax_zoom.legend(loc="upper right", fontsize=9, frameon=True)
    ax_zoom.grid(True, linestyle=':', alpha=0.6)
    ax_zoom.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

    fig.text(0.5, -0.03,
             "In practice, the alert rate often falls below 1% depending on the touchpoint in the user journey.",
             ha='center', fontsize=9, style='italic', color='dimgray')
    fig.suptitle("Threshold Tuning: Balancing Fraud Capture vs. Operational Capacity",
                 fontsize=14, fontweight='bold', y=1.02)

    out2 = "threshold_tuning_plot_twopanel.png"
    fig.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out2}")
