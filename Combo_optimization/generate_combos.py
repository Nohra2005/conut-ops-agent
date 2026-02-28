import json
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib

# ── Config ──────────────────────────────────────────────────────────────────
CHARTS_DIR = "charts"
MODEL_PATH  = "models/combo_pricing_model.pkl"
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs("models",   exist_ok=True)

PALETTE = {
    "bg":      "#0F1117",
    "panel":   "#1A1D27",
    "accent":  "#F5C842",
    "accent2": "#FF6B6B",
    "accent3": "#4ECDC4",
    "text":    "#E8E8E8",
    "muted":   "#6B7280",
}

def apply_dark_style(fig, ax_list):
    fig.patch.set_facecolor(PALETTE["bg"])
    for ax in (ax_list if isinstance(ax_list, list) else [ax_list]):
        ax.set_facecolor(PALETTE["panel"])
        ax.tick_params(colors=PALETTE["text"], labelsize=9)
        ax.xaxis.label.set_color(PALETTE["text"])
        ax.yaxis.label.set_color(PALETTE["text"])
        ax.title.set_color(PALETTE["accent"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["muted"])

# ── 1. Load Data ─────────────────────────────────────────────────────────────
def load_data():
    with open('cleaned_items_prices.json', 'r') as f:
        price_dict = {item['item_name']: item['unit_price'] for item in json.load(f)}

    with open('combo_results.json', 'r') as f:
        combo_data = json.load(f)['combo_recommendations']

    df = pd.DataFrame(combo_data)
    df['Base Item']  = df['if_buys'].apply(lambda x: x[0])
    df['Add-on Item']= df['also_buys'].apply(lambda x: x[0])
    df['Combo Name'] = df['Base Item'] + " + " + df['Add-on Item']
    df['total_orig_price'] = df.apply(
        lambda x: price_dict.get(x['Base Item'], 0) + price_dict.get(x['Add-on Item'], 0), axis=1
    )
    return df

# ── 2. Feature Engineering + Training Labels ─────────────────────────────────
def build_features_labels(df):
    X = df[['confidence_pct', 'lift', 'n_customers', 'total_orig_price']]
    y = 0.25 - (df['confidence_pct'] / 1000) - (df['lift'] / 50)
    y = np.clip(y, 0.05, 0.25)
    return X, y

# ── 3. Train & Save Model ────────────────────────────────────────────────────
def train_and_save(X, y, path=MODEL_PATH):
    print("🧠 Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, path)
    print(f"✅ Model saved → {path}")
    return model

# ── 4. Load Existing Model (for inference without retraining) ─────────────────
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model at '{path}'. Run train_and_save() first.")
    model = joblib.load(path)
    print(f"📦 Model loaded from {path}")
    return model

# ── 5. Predict ────────────────────────────────────────────────────────────────
def predict(model, df, X):
    df = df.copy()
    df['ai_suggested_discount'] = model.predict(X)
    df['promo_price']    = df['total_orig_price'] * (1 - df['ai_suggested_discount'])
    df['target_volume']  = np.ceil(df['n_customers'] / (1 - df['ai_suggested_discount']))
    return df

# ── 6. Charts ─────────────────────────────────────────────────────────────────
def generate_charts(df):
    short_names = [n[:22] + "…" if len(n) > 24 else n for n in df['Combo Name']]

    # ── Chart 1: Discount % per Combo ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    apply_dark_style(fig, ax)
    bars = ax.barh(short_names, df['ai_suggested_discount'] * 100,
                   color=PALETTE["accent"], edgecolor="none", height=0.6)
    ax.bar_label(bars, fmt="%.1f%%", color=PALETTE["text"], fontsize=9, padding=4)
    ax.set_xlabel("AI Suggested Discount (%)")
    ax.set_title("Combo Discount Recommendations", fontsize=13, fontweight="bold", pad=12)
    ax.invert_yaxis()
    plt.tight_layout()
    path1 = f"{CHARTS_DIR}/chart_discounts.png"
    plt.savefig(path1, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved {path1}")

    # ── Chart 2: Original vs Promo Price ─────────────────────────────────────
    x = np.arange(len(short_names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    apply_dark_style(fig, ax)
    ax.bar(x - w/2, df['total_orig_price'], width=w, label="Original",
           color=PALETTE["muted"], edgecolor="none")
    ax.bar(x + w/2, df['promo_price'],      width=w, label="Promo",
           color=PALETTE["accent3"], edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel("Price (units)")
    ax.set_title("Original vs Promo Prices per Combo", fontsize=13, fontweight="bold", pad=12)
    ax.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"], framealpha=0.8)
    plt.tight_layout()
    path2 = f"{CHARTS_DIR}/chart_prices.png"
    plt.savefig(path2, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved {path2}")

    # ── Chart 3: Target Volume to Break Even ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    apply_dark_style(fig, ax)
    ax.plot(short_names, df['target_volume'], marker='o', color=PALETTE["accent2"],
            linewidth=2, markersize=7)
    ax.fill_between(range(len(short_names)), df['target_volume'],
                    alpha=0.15, color=PALETTE["accent2"])
    ax.set_xticklabels(short_names, rotation=35, ha='right', fontsize=8)
    ax.set_xticks(range(len(short_names)))
    ax.set_ylabel("Customers Needed")
    ax.set_title("Break-even Customer Volume per Combo", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    path3 = f"{CHARTS_DIR}/chart_breakeven.png"
    plt.savefig(path3, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved {path3}")

    # ── Chart 4: Feature Importance ───────────────────────────────────────────
    return [path1, path2, path3]

def generate_feature_importance_chart(model, feature_names):
    importances = model.feature_importances_
    idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(7, 4))
    apply_dark_style(fig, ax)
    ax.barh([feature_names[i] for i in idx], importances[idx],
            color=PALETTE["accent"], edgecolor="none")
    ax.set_title("Feature Importance (What drives discount?)", fontsize=12,
                 fontweight="bold", pad=10)
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    path4 = f"{CHARTS_DIR}/chart_feature_importance.png"
    plt.savefig(path4, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved {path4}")
    return path4

# ── 7. Save Results CSV ───────────────────────────────────────────────────────
def save_results(df):
    out = df[['Combo Name', 'total_orig_price', 'ai_suggested_discount',
              'promo_price', 'target_volume']].copy()
    out['ai_suggested_discount'] = (out['ai_suggested_discount'] * 100).round(1).astype(str) + '%'
    out['promo_price']     = out['promo_price'].round(2)
    out['total_orig_price']= out['total_orig_price'].round(2)
    out.to_csv('ai_combo_pricing_results.csv', index=False)
    print("✅ Results saved → ai_combo_pricing_results.csv")

# ── Main Pipeline ─────────────────────────────────────────────────────────────
def train_predict_and_visualize():
    print("🚀 Initializing AI Pricing Model...")
    df         = load_data()
    X, y       = build_features_labels(df)
    model      = train_and_save(X, y)        # trains & persists
    df         = predict(model, df, X)
    chart_paths= generate_charts(df)
    fi_path    = generate_feature_importance_chart(model, X.columns.tolist())
    save_results(df)
    print(f"\n🎉 Done! Charts saved in ./{CHARTS_DIR}/")
    return df, model

if __name__ == "__main__":
    train_predict_and_visualize()