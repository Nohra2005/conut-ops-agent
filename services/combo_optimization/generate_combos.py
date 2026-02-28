import json, os, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR  = os.path.join(BASE_DIR, "charts")
MODEL_PATH  = os.path.join(BASE_DIR, "models", "combo_pricing_model.pkl")
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

PALETTE = {"bg":"#0F1117","panel":"#1A1D27","accent":"#F5C842",
           "accent2":"#FF6B6B","accent3":"#4ECDC4","text":"#E8E8E8","muted":"#6B7280"}

def apply_dark_style(fig, ax_list):
    fig.patch.set_facecolor(PALETTE["bg"])
    for ax in (ax_list if isinstance(ax_list, list) else [ax_list]):
        ax.set_facecolor(PALETTE["panel"]); ax.tick_params(colors=PALETTE["text"])
        ax.xaxis.label.set_color(PALETTE["text"]); ax.yaxis.label.set_color(PALETTE["text"])
        ax.title.set_color(PALETTE["accent"])
        for spine in ax.spines.values(): spine.set_edgecolor(PALETTE["muted"])

def load_data():
    with open(os.path.join(BASE_DIR, 'cleaned_items_prices.json')) as f:
        price_dict = {item['item_name']: item['unit_price'] for item in json.load(f)}
    with open(os.path.join(BASE_DIR, 'combo_results.json')) as f:
        combo_data = json.load(f)['combo_recommendations']
    df = pd.DataFrame(combo_data)
    df['Base Item']   = df['if_buys'].apply(lambda x: x[0])
    df['Add-on Item'] = df['also_buys'].apply(lambda x: x[0])
    df['Combo Name']  = df['Base Item'] + " + " + df['Add-on Item']
    df['total_orig_price'] = df.apply(
        lambda x: price_dict.get(x['Base Item'],0) + price_dict.get(x['Add-on Item'],0), axis=1)
    return df

def build_features_labels(df):
    X = df[['confidence_pct','lift','n_customers','total_orig_price']]
    y = np.clip(0.25 - (df['confidence_pct']/1000) - (df['lift']/50), 0.05, 0.25)
    return X, y

def train_and_save(X, y):
    print("Training Random Forest...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model

def predict(model, df, X):
    df = df.copy()
    df['ai_suggested_discount'] = model.predict(X)
    df['promo_price']   = df['total_orig_price'] * (1 - df['ai_suggested_discount'])
    df['target_volume'] = np.ceil(df['n_customers'] / (1 - df['ai_suggested_discount']))
    return df

def generate_charts(df):
    short_names = [n[:22]+"…" if len(n)>24 else n for n in df['Combo Name']]
    fig, ax = plt.subplots(figsize=(10,5)); apply_dark_style(fig,ax)
    bars = ax.barh(short_names, df['ai_suggested_discount']*100, color=PALETTE["accent"], edgecolor="none", height=0.6)
    ax.bar_label(bars, fmt="%.1f%%", color=PALETTE["text"], fontsize=9, padding=4)
    ax.set_xlabel("AI Suggested Discount (%)"); ax.set_title("Combo Discount Recommendations", fontsize=13, fontweight="bold")
    ax.invert_yaxis(); plt.tight_layout()
    p1 = os.path.join(CHARTS_DIR,"chart_discounts.png"); plt.savefig(p1, dpi=130, bbox_inches='tight'); plt.close()
    print(f"Saved {p1}")

    x=np.arange(len(short_names)); w=0.38
    fig,ax=plt.subplots(figsize=(11,5)); apply_dark_style(fig,ax)
    ax.bar(x-w/2, df['total_orig_price'], width=w, label="Original", color=PALETTE["muted"], edgecolor="none")
    ax.bar(x+w/2, df['promo_price'],      width=w, label="Promo",    color=PALETTE["accent3"], edgecolor="none")
    ax.set_xticks(x); ax.set_xticklabels(short_names, rotation=35, ha='right', fontsize=8)
    ax.set_title("Original vs Promo Prices", fontsize=13, fontweight="bold")
    ax.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"]); plt.tight_layout()
    p2 = os.path.join(CHARTS_DIR,"chart_prices.png"); plt.savefig(p2, dpi=130, bbox_inches='tight'); plt.close()
    print(f"Saved {p2}")

def save_results(df):
    out = df[['Combo Name','total_orig_price','ai_suggested_discount','promo_price','target_volume']].copy()
    out['ai_suggested_discount'] = (out['ai_suggested_discount']*100).round(1).astype(str)+'%'
    out.to_csv(os.path.join(BASE_DIR,'ai_combo_pricing_results.csv'), index=False)
    print("Results saved to ai_combo_pricing_results.csv")

def train_predict_and_visualize():
    df = load_data(); X, y = build_features_labels(df)
    model = train_and_save(X, y); df = predict(model, df, X)
    generate_charts(df); save_results(df)
    return df, model

if __name__ == "__main__":
    train_predict_and_visualize()