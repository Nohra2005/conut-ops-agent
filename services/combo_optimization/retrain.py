"""
retrain.py  —  Combo Pricing Model Retraining Script
═════════════════════════════════════════════════════
  python retrain.py              # retrain with default data files
  python retrain.py --tune       # retrain + multi-model hyperparameter search (RF, GB, ET, Ridge)
  python retrain.py --dry-run    # validate data only
  python retrain.py --force      # replace even if new model is worse
"""

import argparse, json, os, shutil, sys
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
CHARTS_DIR = os.path.join(BASE_DIR, "charts")
MODEL_PATH   = os.path.join(MODEL_DIR, "combo_pricing_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "training_metrics.json")
DEFAULT_PRICES_PATH = os.path.join(BASE_DIR, "cleaned_items_prices.json")
DEFAULT_COMBOS_PATH = os.path.join(BASE_DIR, "combo_results.json")

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

BG, PANEL, GOLD, TEAL = "#0F1117", "#1A1D27", "#F5C842", "#4ECDC4"

def _style(fig, axes):
    fig.patch.set_facecolor(BG)
    for ax in (axes if isinstance(axes, list) else [axes]):
        ax.set_facecolor(PANEL); ax.tick_params(colors="#E8E8E8")
        for label in [ax.xaxis.label, ax.yaxis.label, ax.title]: label.set_color("#E8E8E8")
        for sp in ax.spines.values(): sp.set_edgecolor("#6B7280")

def load_and_validate(prices_path, combos_path):
    print(f"\n[1/5] Loading data\n      prices: {prices_path}\n      combos: {combos_path}")
    for path in [prices_path, combos_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
    with open(prices_path) as f: raw_prices = json.load(f)
    price_dict = {item["item_name"]: item["unit_price"] for item in raw_prices}
    print(f"      ✓  {len(price_dict)} price entries loaded")
    with open(combos_path) as f: combo_data = json.load(f)
    if "combo_recommendations" not in combo_data:
        raise ValueError("combos file missing 'combo_recommendations' key")
    combos = combo_data["combo_recommendations"]
    if not combos: raise ValueError("combo_recommendations is empty")
    print(f"      ✓  {len(combos)} combos loaded")
    df = pd.DataFrame(combos)
    df["Base"]   = df["if_buys"].apply(lambda x: x[0])
    df["Add-on"] = df["also_buys"].apply(lambda x: x[0])
    df["total_orig_price"] = df.apply(
        lambda row: price_dict.get(row["Base"], 0) + price_dict.get(row["Add-on"], 0), axis=1)
    missing = df[df["total_orig_price"] == 0]["Base"].tolist()
    if missing: print(f"      ⚠  {len(missing)} combos have no price data")
    X = df[["confidence_pct", "lift", "n_customers", "total_orig_price"]]
    y = np.clip(0.25 - (df["confidence_pct"]/1000) - (df["lift"]/50), 0.05, 0.25)
    print(f"      ✓  Target range: [{y.min():.3f}, {y.max():.3f}]")
    return X, y

def train(X, y, params=None):
    params = params or {"n_estimators": 100, "random_state": 42}
    print(f"\n[2/5] Training model  (params: {params})")
    model = RandomForestRegressor(**params); model.fit(X, y)
    print(f"      ✓  Training complete"); return model

MODEL_CANDIDATES = [
    (
        "RandomForest",
        RandomForestRegressor(random_state=42),
        {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10], "min_samples_leaf": [1, 2, 5]},
    ),
    (
        "GradientBoosting",
        GradientBoostingRegressor(random_state=42),
        {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1, 0.2]},
    ),
    (
        "ExtraTrees",
        ExtraTreesRegressor(random_state=42),
        {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10], "min_samples_leaf": [1, 2, 5]},
    ),
    (
        "Ridge",
        Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())]),
        {"ridge__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    ),
]

def tune_all_models(X, y):
    print(f"\n[2/5] Multi-model hyperparameter search ({len(MODEL_CANDIDATES)} candidates)...")
    cv = min(5, len(X))
    best_name, best_estimator, best_params, best_score = None, None, None, float("inf")

    results = []
    for name, estimator, param_grid in MODEL_CANDIDATES:
        print(f"      → {name}...", end=" ", flush=True)
        gs = GridSearchCV(estimator, param_grid, cv=cv,
                          scoring="neg_mean_absolute_error", n_jobs=-1)
        gs.fit(X, y)
        mae = -gs.best_score_
        results.append((name, mae, gs.best_params_))
        print(f"MAE {mae:.5f}  params: {gs.best_params_}")
        if mae < best_score:
            best_score = mae
            best_name, best_estimator, best_params = name, gs.best_estimator_, gs.best_params_

    print(f"\n      ┌─ Model comparison ({'─'*40})")
    for name, mae, params in sorted(results, key=lambda r: r[1]):
        marker = "◀ BEST" if name == best_name else ""
        print(f"      │  {name:<20}  MAE {mae:.5f}  {marker}")
    print(f"      └{'─'*46}")
    print(f"      ✓  Winner: {best_name}  MAE: {best_score:.5f}")
    return best_estimator, best_name, best_params

def evaluate(model, X, y):
    n_splits = min(5, len(X))
    print(f"\n[3/5] Evaluating ({n_splits}-fold CV, {len(X)} samples)...")
    scores = cross_val_score(model, X, y, cv=n_splits, scoring="neg_mean_absolute_error")
    mae = -scores
    metrics = {"cv_folds": n_splits, "cv_mae_mean": round(float(mae.mean()),5),
               "cv_mae_std": round(float(mae.std()),5), "trained_at": datetime.now().isoformat()}
    print(f"      ✓  CV MAE: {metrics['cv_mae_mean']:.5f} ± {metrics['cv_mae_std']:.5f}")
    if n_splits < 5: print(f"      ⚠  Only {len(X)} combos — less reliable CV scores")
    return metrics

def load_previous_mae():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f: return json.load(f).get("cv_mae_mean")
    return None

def backup_current_model():
    if os.path.exists(MODEL_PATH):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = MODEL_PATH.replace(".pkl", f"_backup_{ts}.pkl")
        shutil.copy2(MODEL_PATH, backup); print(f"      ✓  Backed up → {backup}")

def save_model(model, metrics, best_params, force):
    print(f"\n[4/5] Comparing against deployed model...")
    old_mae = load_previous_mae(); new_mae = metrics["cv_mae_mean"]
    if old_mae is not None:
        delta = new_mae - old_mae; sign = "+" if delta > 0 else ""
        print(f"      Old MAE: {old_mae:.5f}  New MAE: {new_mae:.5f}  ({sign}{delta:.5f})")
        if new_mae > old_mae and not force:
            print(f"      ✗  New model is worse. Use --force to override."); return False
        elif new_mae > old_mae and force:
            print(f"      ⚠  Worse but --force passed. Replacing anyway.")
        else:
            print(f"      ✓  Equal or better. Replacing.")
    else:
        print(f"      No deployed model found. Saving new model.")
    backup_current_model()
    joblib.dump(model, MODEL_PATH)
    metrics["best_params"] = best_params
    metrics["model_name"]  = getattr(model, "__class__", type(model)).__name__
    with open(METRICS_PATH, "w") as f: json.dump(metrics, f, indent=2)
    print(f"      ✓  Saved → {MODEL_PATH}"); return True

def save_learning_curve(model, X, y):
    print(f"\n[5/5] Generating charts...")
    n_splits = min(5, len(X))
    sizes, train_s, val_s = learning_curve(model, X, y, cv=n_splits,
        scoring="neg_mean_absolute_error", train_sizes=np.linspace(0.5,1.0,min(6,len(X))))
    fig, ax = plt.subplots(figsize=(8,4)); _style(fig, ax)
    ax.plot(sizes, -train_s.mean(axis=1), "o-", color=GOLD, label="Train MAE")
    ax.plot(sizes, -val_s.mean(axis=1),   "o-", color=TEAL, label="Validation MAE")
    ax.set_xlabel("Training Samples"); ax.set_ylabel("MAE")
    ax.set_title("Learning Curve — Combo Pricing Model", fontsize=12, fontweight="bold")
    ax.legend(facecolor=PANEL, labelcolor="#E8E8E8"); plt.tight_layout()
    out = os.path.join(MODEL_DIR, "learning_curve.png")
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    print(f"      ✓  Saved → {out}")

def save_feature_importance(model, feature_names):
    # Unwrap Pipeline to get the final estimator
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    if not hasattr(estimator, "feature_importances_"):
        # Linear model — use absolute coefficients instead
        if hasattr(estimator, "coef_"):
            imp = np.abs(estimator.coef_)
            chart_title = "Feature Coefficients (abs)"
            xlabel = "Absolute Coefficient"
        else:
            print("      ⚠  Model has no importances or coefs — skipping chart.")
            return
    else:
        imp = estimator.feature_importances_
        chart_title = "Feature Importance"
        xlabel = "Importance Score"

    idx = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(7, 4)); _style(fig, ax)
    ax.barh([feature_names[i] for i in idx], imp[idx], color=GOLD, edgecolor="none")
    ax.set_title(chart_title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel); plt.tight_layout()
    out = os.path.join(MODEL_DIR, "feature_importance.png")
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    print(f"      ✓  Saved → {out}")

def main():
    parser = argparse.ArgumentParser(description="Retrain the Conut combo pricing model.")
    parser.add_argument("--prices",  default=DEFAULT_PRICES_PATH)
    parser.add_argument("--data",    default=DEFAULT_COMBOS_PATH)
    parser.add_argument("--tune",    action="store_true")
    parser.add_argument("--force",   action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("═"*60)
    print(f"  Conut — Combo Pricing Model Retrainer  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═"*60)

    try: X, y = load_and_validate(args.prices, args.data)
    except (FileNotFoundError, ValueError) as e: print(f"\n  ERROR: {e}"); sys.exit(1)

    if args.dry_run: print("\n  --dry-run: data looks good. Exiting."); sys.exit(0)

    mlflow.set_experiment("combo")

    with mlflow.start_run():
        mlflow.set_tags({
            "trained_at": datetime.now().isoformat(),
            "prices_path": args.prices,
            "combos_path": args.data,
            "tuned": str(args.tune),
            "forced": str(args.force),
        })

        mlflow.log_param("n_samples", len(X))
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("feature_names", X.columns.tolist())

        best_params = None
        model_name = "RandomForest"
        if args.tune:
            model, model_name, best_params = tune_all_models(X, y)
            mlflow.log_param("winning_model", model_name)
            mlflow.log_params(best_params)
        else:
            default_params = {"n_estimators": 100, "random_state": 42}
            model = train(X, y, default_params)
            mlflow.log_params(default_params)

        metrics = evaluate(model, X, y)
        mlflow.log_metric("cv_mae_mean", metrics["cv_mae_mean"])
        mlflow.log_metric("cv_mae_std",  metrics["cv_mae_std"])
        mlflow.log_metric("cv_folds",    metrics["cv_folds"])

        saved = save_model(model, metrics, best_params, force=args.force)
        mlflow.log_param("model_deployed", saved)

        lc_path = os.path.join(MODEL_DIR, "learning_curve.png")
        fi_path = os.path.join(MODEL_DIR, "feature_importance.png")
        save_learning_curve(model, X, y)
        save_feature_importance(model, X.columns.tolist())

        mlflow.log_artifact(lc_path, artifact_path="charts")
        mlflow.log_artifact(fi_path, artifact_path="charts")
        if os.path.exists(METRICS_PATH):
            mlflow.log_artifact(METRICS_PATH, artifact_path="metrics")

        mlflow.sklearn.log_model(model, artifact_path="model")

    print("\n" + "═"*60)
    print("  ✓  Done. Model deployed." if saved else "  ✗  Done. Model NOT replaced (use --force).")
    print("═"*60 + "\n")

if __name__ == "__main__":
    main()