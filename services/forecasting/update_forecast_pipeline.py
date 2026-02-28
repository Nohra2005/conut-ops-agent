import os
import sys

def run_pipeline():
    forecast_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, forecast_dir)

    print("=" * 40)
    print("FORECASTING PIPELINE")
    print("=" * 40)

    print("\n[1/1] Running preprocessing...")
    from preprocessing import run_preprocessing
    data_dir = os.path.normpath(os.path.join(forecast_dir, "..", "..", "data"))
    run_preprocessing(data_dir, forecast_dir)

    print("\n" + "=" * 40)
    print("PIPELINE COMPLETE")
    print("master_daily_inventory.csv updated.")
    print("Restart api.py to reload the new data.")
    print("=" * 40)

if __name__ == "__main__":
    run_pipeline()