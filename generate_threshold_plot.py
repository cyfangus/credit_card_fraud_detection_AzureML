"""
generate_threshold_plot.py
--------------------------
Standalone script to reproduce the threshold tuning plot from notebook 03.
No Azure, no MLflow, no SHAP required.

Usage:
    python generate_threshold_plot.py --data path/to/creditcard.csv
    python generate_threshold_plot.py --data path/to/creditcard.csv --two-panel

Output:
    threshold_tuning_plot.png          — zoomed operating region (0–10%)
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


# ── 1. Args ──────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, help="Path to creditcard.csv")
parser.add_argument("--two-panel", action="store_true",
                    help="Also save a full-range + zoomed two-panel figure")
args = parser.parse_args()


# ── 2. Load & preprocess  (mirrors notebook 02) ──────────────────────────────

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


# ── 3. IsolationForest anomaly score  (mirrors notebook 03, cell 5) ──────────

print("Training IsolationForest...")
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(X_train)
X_train = X_train.copy()
X_test  = X_test.copy()
X_train['anomaly_score'] = iso_forest.decision_function(X_train)
X_test['anomaly_score']  = iso_forest.decision_function(X_test)


# ── 4. XGBoost champion model  (mirrors notebook 03, cell 12) ────────────────

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


# ── 5. Business threshold tuning  (mirrors notebook 03, cell 14) ─────────────

y_probs = best_model.predict_proba(X_test)[:, 1]

alert_rate_goal    = 0.01
business_threshold = np.percentile(y_probs, 100 * (1 - alert_rate_goal))

y_pred_business = (y_probs >= business_threshold).astype(int)
prec = precision_score(y_test, y_pred_business)
rec  = recall_score(y_test, y_pred_business)

print(f"\n--- Business-Constraint Tuning (1% Alert Cap) ---")
print(f"Operational Threshold  : {business_threshold:.4f}")
print(f"Precision (Hit Rate)   : {prec:.2%}")
print(f"Recall (Fraud Captured): {rec:.2%}")


# ── 6. Curve data & key points ───────────────────────────────────────────────

precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

n_total             = len(y_probs)
alert_rates         = np.array([np.sum(y_probs >= t) / n_total for t in thresholds])
business_alert_rate = np.sum(y_probs >= business_threshold) / n_total

# Reverse so x-axis reads low → high alert rate (left to right)
alert_rates_plot = alert_rates[::-1]
precisions_plot  = precisions[:-1][::-1]
recalls_plot     = recalls[:-1][::-1]

# F1-max point — where precision == recall (balanced sweet spot)
f1_scores    = 2 * precisions_plot * recalls_plot / (precisions_plot + recalls_plot + 1e-10)
f1_max_idx   = np.argmax(f1_scores)
f1_alert_rate = alert_rates_plot[f1_max_idx]
f1_prec      = precisions_plot[f1_max_idx]
f1_rec       = recalls_plot[f1_max_idx]
f1_score_val = f1_scores[f1_max_idx]

# Interpolated values at the 1% operating point (for accurate arrow tips)
op_prec = float(np.interp(business_alert_rate, alert_rates_plot, precisions_plot))
op_rec  = float(np.interp(business_alert_rate, alert_rates_plot, recalls_plot))

print(f"\n--- F1-Max Point (Model Sweet Spot) ---")
print(f"Alert Rate : {f1_alert_rate:.2%}")
print(f"Precision  : {f1_prec:.2%}")
print(f"Recall     : {f1_rec:.2%}")
print(f"F1 Score   : {f1_score_val:.2%}")


# ── 7. Panel drawing functions ───────────────────────────────────────────────

def draw_full_panel(ax):
    """Left panel: curves only — clean overview, no markers."""
    ax.plot(alert_rates_plot, precisions_plot, color='steelblue',
            linewidth=2, label="Precision (Hit Rate)")
    ax.plot(alert_rates_plot, recalls_plot, color='seagreen',
            linewidth=2, label="Recall (Capture Rate)")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("Alert Rate (% of transactions flagged)", fontsize=11)
    ax.set_ylabel("Score (0.0 – 1.0)", fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.legend(loc='upper right', fontsize=9, frameon=True)


def draw_zoom_panel(ax, xlim=0.03, fontsize=10):
    """Right panel: zoomed view with F1-max and operational cap markers."""
    ax.plot(alert_rates_plot, precisions_plot, color='steelblue',
            linewidth=2, label="Precision (Hit Rate)")
    ax.plot(alert_rates_plot, recalls_plot, color='seagreen',
            linewidth=2, label="Recall (Capture Rate)")

    # Zone 1: 0 → F1-max  (model sweet spot)
    ax.axvspan(0, f1_alert_rate, color='mediumpurple', alpha=0.10,
               label=f'Model Sweet Spot (≤{f1_alert_rate:.1%})')

    # Zone 2: F1-max → 1% cap  (operational buffer)
    ax.axvspan(f1_alert_rate, business_alert_rate, color='salmon', alpha=0.12,
               label=f'Operational Buffer ({f1_alert_rate:.1%}–{business_alert_rate:.1%})')

    # Vertical lines
    ax.axvline(f1_alert_rate, color='mediumpurple', linestyle='--', linewidth=1.5,
               label=f'F1 Max ({f1_alert_rate:.1%})')
    ax.axvline(business_alert_rate, color='crimson', linestyle='--', linewidth=1.5,
               label='Operational Cap (1%)')

    # Dot markers
    ax.plot(f1_alert_rate,       f1_prec, 'o', color='mediumpurple', markersize=7, zorder=5)
    ax.plot(f1_alert_rate,       f1_rec,  'o', color='mediumpurple', markersize=7, zorder=5)
    ax.plot(business_alert_rate, op_prec, 'o', color='crimson',      markersize=7, zorder=5)
    ax.plot(business_alert_rate, op_rec,  'o', color='crimson',      markersize=7, zorder=5)

    # Annotation — F1-max
    ax.annotate(
        f'F1 Max\nP={f1_prec:.0%}  R={f1_rec:.0%}\nF1={f1_score_val:.0%}',
        xy=(f1_alert_rate, (f1_prec + f1_rec) / 2),
        xytext=(f1_alert_rate + xlim * 0.18, 0.55),
        fontsize=fontsize - 1, color='mediumpurple',
        arrowprops=dict(arrowstyle='->', color='mediumpurple', lw=1.2),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='mediumpurple', alpha=0.85)
    )

    # Annotation — 1% cap
    ax.annotate(
        f'Operational Cap\nP={prec:.0%}  R={rec:.0%}',
        xy=(business_alert_rate, (op_prec + op_rec) / 2),
        xytext=(business_alert_rate + xlim * 0.18, 0.30),
        fontsize=fontsize - 1, color='crimson',
        arrowprops=dict(arrowstyle='->', color='crimson', lw=1.2),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='crimson', alpha=0.85)
    )

    ax.set_xlim([0, xlim])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("Alert Rate (% of transactions flagged)", fontsize=fontsize + 1)
    ax.set_ylabel("Score (0.0 – 1.0)", fontsize=fontsize + 1)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
    ax.legend(loc='upper right', fontsize=fontsize - 2, frameon=True)


# ── 8. Single plot (zoomed 0–3%) ─────────────────────────────────────────────

print("\nGenerating single plot...")
fig, ax = plt.subplots(figsize=(10, 6))
draw_zoom_panel(ax, xlim=0.03)
ax.set_title("Threshold Tuning: Balancing Fraud Capture vs. Operational Capacity",
             fontsize=13, fontweight='bold')
fig.text(0.5, -0.02,
         "In practice, the alert rate often falls below 1% depending on the touchpoint in the user journey.",
         ha='center', fontsize=9, style='italic', color='dimgray')

out = "threshold_tuning_plot.png"
fig.tight_layout()
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved → {out}")


# ── 9. Two-panel figure (optional) ───────────────────────────────────────────

if args.two_panel:
    print("Generating two-panel figure...")

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(16, 6),
                                            gridspec_kw={'wspace': 0.40})

    # Left: clean full range — curves only + zoom box
    draw_full_panel(ax_full)
    ax_full.set_title("Full Range (0–100%)", fontsize=12)

    # Orange outline box marking the zoomed region (0–3%)
    rect = mpatches.FancyBboxPatch(
        (0, 0), 0.03, 1.05,
        boxstyle="square,pad=0", linewidth=2,
        edgecolor="darkorange", facecolor="darkorange", alpha=0.12, zorder=4
    )
    ax_full.add_patch(rect)
    ax_full.text(0.015, 0.12, "Zoomed →", ha='center', fontsize=8,
                 color='darkorange', fontweight='bold')

    # Right: zoomed 0–3%
    draw_zoom_panel(ax_zoom, xlim=0.03, fontsize=10)
    ax_zoom.set_title("Operating Region (0–3%)", fontsize=12)
    ax_zoom.set_ylabel("")   # remove duplicate y-label on right panel

    fig.suptitle("Threshold Tuning: Balancing Fraud Capture vs. Operational Capacity",
                 fontsize=14, fontweight='bold', y=1.02)
    fig.text(0.5, -0.03,
             "In practice, the alert rate often falls below 1% depending on the touchpoint in the user journey.",
             ha='center', fontsize=9, style='italic', color='dimgray')

    out2 = "threshold_tuning_plot_twopanel.png"
    fig.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out2}")
