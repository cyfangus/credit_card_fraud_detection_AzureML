"""
generate_plots.py
-----------------
Generates every README graphic from raw creditcard.csv.
No Azure, no MLflow required.

Usage:
    python generate_plots.py --data path/to/creditcard.csv

Output (all saved to assets/):
    01_class_distribution.png   — EDA: fraud vs non-fraud
    02_shap_bar.png             — global SHAP feature importance
    03_shap_beeswarm.png        — SHAP beeswarm (per-transaction impact)
    04_threshold_tuning.png     — precision/recall vs alert rate (0–3%)
    05_threshold_tuning_twopanel.png — full range + zoomed side-by-side
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
import xgboost as xgb


# ── Args ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, help="Path to creditcard.csv")
args = parser.parse_args()

OUT = "assets"
os.makedirs(OUT, exist_ok=True)
print(f"Outputs → {OUT}/\n")


# ── Load & preprocess  (mirrors notebook 02) ─────────────────────────────────

print("── 1/5  Loading & preprocessing data...")
df_raw = pd.read_csv(args.data)

df = df_raw.copy()
df['Hour'] = df['Time'].apply(lambda x: (x / 3600) % 24)
robust_scaler = RobustScaler()
df['scaled_amount'] = robust_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df.drop(['Time', 'Amount'], axis=1, inplace=True)

X = df.drop('Class', axis=1)
y = df['Class']

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in sss.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print(f"   Train: {len(X_train):,}  |  Test: {len(X_test):,}  |  Fraud rate: {y_train.mean():.3%}\n")


# ── Plot 1: EDA class distribution ───────────────────────────────────────────

print("── 2/5  Generating class distribution plot...")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Bar chart (log scale)
counts = df_raw['Class'].value_counts().sort_index()
bars = axes[0].bar(['Non-Fraud (0)', 'Fraud (1)'], counts.values,
                   color=['steelblue', 'crimson'], edgecolor='white', width=0.5)
axes[0].set_yscale('log')
axes[0].set_ylabel('Count (log scale)', fontsize=11)
axes[0].set_title('Transaction Distribution', fontsize=13)
for bar, val in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2, val * 1.3,
                 f'{val:,}', ha='center', va='bottom', fontsize=10)
axes[0].grid(True, axis='y', linestyle=':', alpha=0.5)

# Pie chart
axes[1].pie(counts.values,
            labels=['Non-Fraud', 'Fraud'],
            autopct='%1.2f%%',
            colors=['steelblue', 'crimson'],
            startangle=90,
            wedgeprops=dict(edgecolor='white'))
axes[1].set_title('Class Balance', fontsize=13)

fig.suptitle('Extreme Class Imbalance: 0.17% Fraud Rate', fontsize=14,
             fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/01_class_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved → {OUT}/01_class_distribution.png\n")


# ── Train models ─────────────────────────────────────────────────────────────

print("── Training IsolationForest...")
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(X_train)
X_train = X_train.copy()
X_test  = X_test.copy()
X_train['anomaly_score'] = iso_forest.decision_function(X_train)
X_test['anomaly_score']  = iso_forest.decision_function(X_test)

print("── Training XGBoost champion model...")
freq_weights = np.where(y_train == 1, 578, 1)
best_model = xgb.XGBClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=6,
    eval_metric='aucpr', random_state=42
)
best_model.fit(X_train, y_train, sample_weight=freq_weights)
print()


# ── Plot 2: SHAP bar importance ───────────────────────────────────────────────

print("── 3/5  Generating SHAP bar plot (this may take a minute)...")

X_test_fraud   = X_test[y_test == 1].sample(min(50, int(y_test.sum())), random_state=42)
X_test_legit   = X_test[y_test == 0].sample(50, random_state=42)
X_test_enriched = pd.concat([X_test_fraud, X_test_legit])

model_func_logit = lambda x: best_model.get_booster().predict(
    xgb.DMatrix(x), output_margin=True)
masker = shap.maskers.Independent(X_train, max_samples=100)
explainer = shap.Explainer(model_func_logit, masker)
shap_values = explainer(X_test_enriched)

shap.plots.bar(shap_values, show=False)
plt.title("Global Feature Importance (SHAP)", fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(f"{OUT}/02_shap_bar.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved → {OUT}/02_shap_bar.png\n")


# ── Plot 3: SHAP beeswarm ─────────────────────────────────────────────────────

print("── 4/5  Generating SHAP beeswarm plot...")

shap.plots.beeswarm(shap_values, show=False)
plt.title("Feature Impact Distribution (SHAP Beeswarm)", fontsize=13,
          fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(f"{OUT}/03_shap_beeswarm.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved → {OUT}/03_shap_beeswarm.png\n")


# ── Threshold curve data ──────────────────────────────────────────────────────

print("── 5/5  Generating threshold tuning plots...")

y_probs = best_model.predict_proba(X_test)[:, 1]

business_threshold  = np.percentile(y_probs, 99)   # top 1%
y_pred_business     = (y_probs >= business_threshold).astype(int)
prec                = precision_score(y_test, y_pred_business)
rec                 = recall_score(y_test, y_pred_business)
business_alert_rate = np.mean(y_probs >= business_threshold)

precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
n_total          = len(y_probs)
alert_rates      = np.array([np.sum(y_probs >= t) / n_total for t in thresholds])

# Reverse for left-to-right reading
alert_rates_plot = alert_rates[::-1]
precisions_plot  = precisions[:-1][::-1]
recalls_plot     = recalls[:-1][::-1]

# F1-max point
f1_scores     = 2 * precisions_plot * recalls_plot / (precisions_plot + recalls_plot + 1e-10)
f1_max_idx    = np.argmax(f1_scores)
f1_alert_rate = alert_rates_plot[f1_max_idx]
f1_prec       = precisions_plot[f1_max_idx]
f1_rec        = recalls_plot[f1_max_idx]
f1_score_val  = f1_scores[f1_max_idx]

op_prec = float(np.interp(business_alert_rate, alert_rates_plot, precisions_plot))
op_rec  = float(np.interp(business_alert_rate, alert_rates_plot, recalls_plot))

print(f"   Operational Cap  → Alert Rate: {business_alert_rate:.2%} | P: {prec:.2%} | R: {rec:.2%}")
print(f"   F1-Max           → Alert Rate: {f1_alert_rate:.2%} | P: {f1_prec:.2%} | R: {f1_rec:.2%} | F1: {f1_score_val:.2%}")


# ── Shared zoom-panel drawing helper ─────────────────────────────────────────

def draw_full_panel(ax):
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
    ax.plot(alert_rates_plot, precisions_plot, color='steelblue',
            linewidth=2, label="Precision (Hit Rate)")
    ax.plot(alert_rates_plot, recalls_plot, color='seagreen',
            linewidth=2, label="Recall (Capture Rate)")

    ax.axvspan(0, f1_alert_rate, color='mediumpurple', alpha=0.10,
               label=f'Model Sweet Spot (≤{f1_alert_rate:.1%})')
    ax.axvspan(f1_alert_rate, business_alert_rate, color='salmon', alpha=0.12,
               label=f'Operational Buffer ({f1_alert_rate:.1%}–{business_alert_rate:.1%})')
    ax.axvline(f1_alert_rate, color='mediumpurple', linestyle='--',
               linewidth=1.5, label=f'F1 Max ({f1_alert_rate:.1%})')
    ax.axvline(business_alert_rate, color='crimson', linestyle='--',
               linewidth=1.5, label='Operational Cap (1%)')

    ax.plot(f1_alert_rate,       f1_prec, 'o', color='mediumpurple', markersize=7, zorder=5)
    ax.plot(f1_alert_rate,       f1_rec,  'o', color='mediumpurple', markersize=7, zorder=5)
    ax.plot(business_alert_rate, op_prec, 'o', color='crimson',      markersize=7, zorder=5)
    ax.plot(business_alert_rate, op_rec,  'o', color='crimson',      markersize=7, zorder=5)

    ax.annotate(
        f'F1 Max\nP={f1_prec:.0%}  R={f1_rec:.0%}\nF1={f1_score_val:.0%}',
        xy=(f1_alert_rate, (f1_prec + f1_rec) / 2),
        xytext=(f1_alert_rate + xlim * 0.18, 0.55),
        fontsize=fontsize - 1, color='mediumpurple',
        arrowprops=dict(arrowstyle='->', color='mediumpurple', lw=1.2),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='mediumpurple', alpha=0.85)
    )
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


# ── Plot 4: single zoomed threshold plot ─────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))
draw_zoom_panel(ax, xlim=0.03)
ax.set_title("Threshold Tuning: Balancing Fraud Capture vs. Operational Capacity",
             fontsize=13, fontweight='bold')
fig.text(0.5, -0.02,
         "In practice, the alert rate often falls below 1% depending on the touchpoint in the user journey.",
         ha='center', fontsize=9, style='italic', color='dimgray')
fig.tight_layout()
fig.savefig(f"{OUT}/04_threshold_tuning.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved → {OUT}/04_threshold_tuning.png")


# ── Plot 5: two-panel threshold plot ─────────────────────────────────────────

fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(16, 6),
                                        gridspec_kw={'wspace': 0.40})

draw_full_panel(ax_full)
ax_full.set_title("Full Range (0–100%)", fontsize=12)
rect = mpatches.FancyBboxPatch(
    (0, 0), 0.03, 1.05, boxstyle="square,pad=0", linewidth=2,
    edgecolor="darkorange", facecolor="darkorange", alpha=0.12, zorder=4
)
ax_full.add_patch(rect)
ax_full.text(0.015, 0.12, "Zoomed →", ha='center', fontsize=8,
             color='darkorange', fontweight='bold')

draw_zoom_panel(ax_zoom, xlim=0.03, fontsize=10)
ax_zoom.set_title("Operating Region (0–3%)", fontsize=12)
ax_zoom.set_ylabel("")

fig.suptitle("Threshold Tuning: Balancing Fraud Capture vs. Operational Capacity",
             fontsize=14, fontweight='bold', y=1.02)
fig.text(0.5, -0.03,
         "In practice, the alert rate often falls below 1% depending on the touchpoint in the user journey.",
         ha='center', fontsize=9, style='italic', color='dimgray')
fig.savefig(f"{OUT}/05_threshold_tuning_twopanel.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved → {OUT}/05_threshold_tuning_twopanel.png")

print("\n✅  All plots saved to assets/")
