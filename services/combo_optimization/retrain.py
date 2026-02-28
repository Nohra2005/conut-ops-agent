"""
retrain.py  —  Combo Pricing Model Retraining Script
═════════════════════════════════════════════════════
Retrain the combo discount model safely in production.

QUICK START
───────────
  python retrain.py                        # retrain with default data files
  python retrain.py --tune                 # retrain + run hyperparameter search
  python retrain.py --dry-run              # validate data only, don't save anything
  python retrain.py --prices my_prices.json --data my_combos.json   # custom paths

WHAT THIS SCRIPT DOES
─────────────────────
  1. Validates input data (checks files exist, required fields present, no bad prices)
  2. Trains a new model (or tunes it if --tune is passed)
  3. Evaluates it with 5-fold cross-validation
  4. Compares against the currently deployed model — only replaces it if the new
     model is equal or better (override with --force)
  5. Backs up the old model before replacing it
  6. Saves metrics + a learning curve chart
"""

import argparse
import json
import os
import shutil
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve

# ── Paths (override with CLI args if needed) ──────────────────────────────────
DEFAULT_PRICES_PATH = "cleaned_items_prices.json"
DEFAULT_COMBOS_PATH = "combo_results.json"
MODEL_DIR           = "models"
MODEL_PATH          = os.path.join(MODEL_DIR, "combo_pricing_model.pkl")
METRICS_PATH        = os.path.join(MODEL_DIR, "training_metrics.json")
CHARTS_DIR          = "charts"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def load_and_validate(prices_path: str, combos_path: str):
    """
    Load both data files, validate them, and return feature matrix X and targets y.
    Raises ValueError with a clear message if anything looks wrong.
    """
    print(f"\n[1/5] Loading data")
    print(f"      prices : {prices_path}")
    print(f"      combos : {combos_path}")

    # ── Check files exist ────────────────────────────────────────────────────
    for path in [prices_path, combos_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"  ✗  File not found: {path}")

    # ── Load prices ──────────────────────────────────────────────────────────
    with open(prices_path) as f:
        raw_prices = json.load(f)

    required_price_fields = {"item_name", "unit_price"}
    for i, entry in enumerate(raw_prices):
        missing = required_price_fields - entry.keys()
        if missing:
            raise ValueError(f"  ✗  prices[{i}] is missing fields: {missing}")

    price_dict = {item["item_name"]: item["unit_price"] for item in raw_prices}
    print(f"      ✓  {len(price_dict)} price entries loaded")

    # ── Load combos ──────────────────────────────────────────────────────────
    with open(combos_path) as f:
        combo_data = json.load(f)

    if "combo_recommendations" not in combo_data:
        raise ValueError("  ✗  combos file is missing 'combo_recommendations' key")

    combos = combo_data["combo_recommendations"]
    if not combos:
        raise ValueError("  ✗  combo_recommendations list is empty — nothing to train on")

    required_combo_fields = {"if_buys", "also_buys", "confidence_pct", "lift", "n_customers"}
    for i, c in enumerate(combos):
        missing = required_combo_fields - c.keys()
        if missing:
            raise ValueError(f"  ✗  combo[{i}] is missing fields: {missing}")

    print(f"      ✓  {len(combos)} combos loaded")

    # ── Build feature matrix ─────────────────────────────────────────────────
    df = pd.DataFrame(combos)
    df["Base"]   = df["if_buys"].apply(lambda x: x[0])
    df["Add-on"] = df["also_buys"].apply(lambda x: x[0])

    df["total_orig_price"] = df.apply(
        lambda row: price_dict.get(row["Base"], 0) + price_dict.get(row["Add-on"], 0),
        axis=1,
    )

    # Warn about any combos whose items have no price data
    missing_prices = df[df["total_orig_price"] == 0]["Base"].tolist()
    if missing_prices:
        print(f"      ⚠  {len(missing_prices)} combo(s) have no price data "
              f"(will default to 0): {missing_prices}")

    X = df[["confidence_pct", "lift", "n_customers", "total_orig_price"]]

    # Target: discount rate between 5% and 25%, inversely proportional to
    # confidence and lift (stronger associations → smaller discount needed)
    y = 0.25 - (df["confidence_pct"] / 1000) - (df["lift"] / 50)
    y = np.clip(y, 0.05, 0.25)

    print(f"      ✓  Features: {list(X.columns)}")
    print(f"      ✓  Target range: [{y.min():.3f}, {y.max():.3f}]")

    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# 2. TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train(X: pd.DataFrame, y: pd.Series, params: dict = None) -> RandomForestRegressor:
    """Train with fixed or provided hyperparameters."""
    params = params or {"n_estimators": 100, "random_state": 42}
    print(f"\n[2/5] Training model  (params: {params})")
    model = RandomForestRegressor(**params)
    model.fit(X, y)
    print(f"      ✓  Training complete")
    return model


def tune(X: pd.DataFrame, y: pd.Series):
    """Run GridSearchCV and return the best estimator + its params."""
    print(f"\n[2/5] Hyperparameter tuning (GridSearchCV, cv=5) ...")
    param_grid = {
        "n_estimators":     [50, 100, 200],
        "max_depth":        [None, 5, 10],
        "min_samples_leaf": [1, 2, 5],
    }
    n_splits = min(5, len(X))
    gs = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=n_splits,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(X, y)
    print(f"      ✓  Best params : {gs.best_params_}")
    print(f"      ✓  Best CV MAE : {-gs.best_score_:.5f}")
    return gs.best_estimator_, gs.best_params_


# ══════════════════════════════════════════════════════════════════════════════
# 3. EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """Cross-validation. Folds are capped at the number of samples."""
    n_splits = min(5, len(X))
    print(f"\n[3/5] Evaluating model ({n_splits}-fold CV, {len(X)} samples) ...")
    scores  = cross_val_score(model, X, y, cv=n_splits, scoring="neg_mean_absolute_error")
    mae     = -scores
    metrics = {
        "cv_folds":      n_splits,
        "cv_mae_mean":   round(float(mae.mean()), 5),
        "cv_mae_std":    round(float(mae.std()),  5),
        "cv_mae_scores": [round(float(s), 5) for s in mae],
        "trained_at":    datetime.now().isoformat(timespec="seconds"),
    }
    print(f"      ✓  CV MAE : {metrics['cv_mae_mean']:.5f} ± {metrics['cv_mae_std']:.5f}")
    if n_splits < 5:
        print(f"      ⚠  Only {len(X)} combos available — using {n_splits}-fold CV instead of 5.")
        print(f"         CV scores are less reliable with this few samples.")
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 4. COMPARE & REPLACE
# ══════════════════════════════════════════════════════════════════════════════

def load_previous_mae() -> float | None:
    """Return the CV MAE of the currently deployed model (or None if none exists)."""
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            old = json.load(f)
        return old.get("cv_mae_mean")
    return None


def backup_current_model():
    """Rename the existing model file so it's not lost."""
    if os.path.exists(MODEL_PATH):
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup  = MODEL_PATH.replace(".pkl", f"_backup_{ts}.pkl")
        shutil.copy2(MODEL_PATH, backup)
        print(f"      ✓  Old model backed up → {backup}")


def save_model(model, metrics: dict, best_params: dict, force: bool):
    """
    Compare new model against deployed model.
    Saves only if new model is equal/better — or if --force is passed.
    """
    print(f"\n[4/5] Comparing against deployed model ...")
    old_mae = load_previous_mae()
    new_mae = metrics["cv_mae_mean"]

    if old_mae is not None:
        delta = new_mae - old_mae
        sign  = "+" if delta > 0 else ""
        print(f"      Old model MAE : {old_mae:.5f}")
        print(f"      New model MAE : {new_mae:.5f}  ({sign}{delta:.5f})")

        if new_mae > old_mae and not force:
            print(f"\n      ✗  New model is WORSE than deployed model.")
            print(f"         Skipping replacement. Use --force to override.")
            return False
        elif new_mae > old_mae and force:
            print(f"      ⚠  New model is worse but --force was passed. Replacing anyway.")
        else:
            print(f"      ✓  New model is equal or better. Replacing.")
    else:
        print(f"      No deployed model found. Saving new model.")

    backup_current_model()
    joblib.dump(model, MODEL_PATH)

    metrics["best_params"] = best_params
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"      ✓  Model saved   → {MODEL_PATH}")
    print(f"      ✓  Metrics saved → {METRICS_PATH}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# 5. CHARTS
# ══════════════════════════════════════════════════════════════════════════════

BG, PANEL, GOLD, TEAL = "#0F1117", "#1A1D27", "#F5C842", "#4ECDC4"

def _style(fig, axes):
    fig.patch.set_facecolor(BG)
    for ax in (axes if isinstance(axes, list) else [axes]):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors="#E8E8E8")
        for label in [ax.xaxis.label, ax.yaxis.label, ax.title]:
            label.set_color("#E8E8E8")
        for sp in ax.spines.values():
            sp.set_edgecolor("#6B7280")


def save_learning_curve(model, X, y):
    print(f"\n[5/5] Generating learning curve chart ...")
    n_splits = min(5, len(X))
    sizes, train_s, val_s = learning_curve(
        model, X, y, cv=n_splits,
        scoring="neg_mean_absolute_error",
        train_sizes=np.linspace(0.5, 1.0, min(6, len(X))),
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    _style(fig, ax)
    ax.plot(sizes, -train_s.mean(axis=1), "o-", color=GOLD,  label="Train MAE")
    ax.plot(sizes, -val_s.mean(axis=1),   "o-", color=TEAL,  label="Validation MAE")
    ax.set_xlabel("Training Samples")
    ax.set_ylabel("MAE")
    ax.set_title("Learning Curve — Combo Pricing Model", fontsize=12, fontweight="bold")
    ax.legend(facecolor=PANEL, labelcolor="#E8E8E8")
    plt.tight_layout()
    out = os.path.join(MODEL_DIR, "learning_curve.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"      ✓  Saved → {out}")


def save_feature_importance(model, feature_names):
    imp = model.feature_importances_
    idx = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(7, 4))
    _style(fig, ax)
    ax.barh([feature_names[i] for i in idx], imp[idx], color=GOLD, edgecolor="none")
    ax.set_title("Feature Importance — What Drives the Discount?",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    out = os.path.join(MODEL_DIR, "feature_importance.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"      ✓  Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Retrain the Conut combo pricing model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--prices",  default=DEFAULT_PRICES_PATH,
                        help=f"Path to prices JSON  (default: {DEFAULT_PRICES_PATH})")
    parser.add_argument("--data",    default=DEFAULT_COMBOS_PATH,
                        help=f"Path to combos JSON  (default: {DEFAULT_COMBOS_PATH})")
    parser.add_argument("--tune",    action="store_true",
                        help="Run GridSearchCV hyperparameter search")
    parser.add_argument("--force",   action="store_true",
                        help="Replace deployed model even if new model is worse")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate data only — do not train or save anything")
    args = parser.parse_args()

    print("═" * 60)
    print("  Conut — Combo Pricing Model Retrainer")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 60)

    # Step 1: Load + validate
    try:
        X, y = load_and_validate(args.prices, args.data)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)

    if args.dry_run:
        print("\n  --dry-run passed. Data looks good. Exiting without training.")
        sys.exit(0)

    # Step 2: Train (or tune)
    best_params = None
    if args.tune:
        model, best_params = tune(X, y)
    else:
        model = train(X, y)

    # Step 3: Evaluate
    metrics = evaluate(model, X, y)

    # Step 4: Compare + save (skipped if new model is worse and --force not set)
    saved = save_model(model, metrics, best_params, force=args.force)

    # Step 5: Charts (always generated, even if model wasn't saved)
    save_learning_curve(model, X, y)
    save_feature_importance(model, X.columns.tolist())

    print("\n" + "═" * 60)
    if saved:
        print("  ✓  Retraining complete. New model is deployed.")
    else:
        print("  ✗  Retraining complete. Model was NOT replaced (use --force to override).")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()