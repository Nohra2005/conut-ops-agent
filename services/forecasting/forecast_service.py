import pandas as pd
import os
from datetime import datetime, timedelta


# UPGRADE PATH: When real daily sales data is available per item,
# replace predict_single_day() with an XGBoost model trained on 
# actual transactions. The API interface stays identical.
class InventoryForecaster:
    def __init__(self, data_dir):
        """Loads master_daily_inventory.csv into memory once at startup."""
        self.data_dir = data_dir
        master_path = os.path.join(data_dir, "master_daily_inventory.csv")
        if not os.path.exists(master_path):
            raise FileNotFoundError("master_daily_inventory.csv not found. Run preprocessing.py first.")
        self.df = pd.read_csv(master_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        # Normalize for case-insensitive lookup
        self.df['branch_upper'] = self.df['branch'].str.upper().str.strip()
        self.df['item_upper']   = self.df['item_name'].str.upper().str.strip()
        print(f"Forecaster loaded: {len(self.df)} rows, "
              f"{self.df['branch'].nunique()} branches, "
              f"{self.df['item_name'].nunique()} items.")

    def predict_single_day(self, branch: str, item: str, date_str: str):
        """Returns predicted qty for a branch/item on a specific date."""
        try:
            target = pd.to_datetime(date_str)
        except Exception:
            return f"Error: invalid date format '{date_str}'. Use YYYY-MM-DD."

        mask = (
            (self.df['date'] == target) &
            (self.df['branch_upper'] == branch.upper().strip()) &
            (self.df['item_upper']   == item.upper().strip())
        )
        rows = self.df[mask]

        if rows.empty:
            # Date might be outside 2026 — fall back to same DOW + month from nearest year
            return self._fallback_predict(branch, item, target)

        return round(float(rows.iloc[0]['predicted_qty']), 4)

    def _fallback_predict(self, branch: str, item: str, target: datetime):
        """
        If date is outside the precomputed range, find the same
        month + day-of-week combination and return the average predicted qty.
        This ensures the service never fails for future dates.
        """
        month_name = target.strftime('%B')
        dow_name   = target.strftime('%A')

        mask = (
            (self.df['branch_upper'] == branch.upper().strip()) &
            (self.df['item_upper']   == item.upper().strip()) &
            (self.df['date'].dt.month_name() == month_name) &
            (self.df['date'].dt.day_name()   == dow_name)
        )
        rows = self.df[mask]

        if rows.empty:
            # Last resort — average for that branch/item across all dates
            mask2 = (
                (self.df['branch_upper'] == branch.upper().strip()) &
                (self.df['item_upper']   == item.upper().strip())
            )
            rows = self.df[mask2]
            if rows.empty:
                return f"Error: branch '{branch}' or item '{item}' not found in data."
            return round(float(rows['predicted_qty'].mean()), 4)

        return round(float(rows['predicted_qty'].mean()), 4)

    def predict_date_range(self, branch: str, item: str, start_str: str, end_str: str):
        """Returns total and daily breakdown for a date range."""
        try:
            start = datetime.strptime(start_str, '%Y-%m-%d')
            end   = datetime.strptime(end_str,   '%Y-%m-%d')
        except Exception:
            return "Error: use YYYY-MM-DD format for dates."

        if end < start:
            return "Error: end_date must be after start_date."

        daily = {}
        current = start
        while current <= end:
            date_str = current.strftime('%Y-%m-%d')
            qty = self.predict_single_day(branch, item, date_str)
            if isinstance(qty, str) and qty.startswith("Error"):
                return qty
            daily[date_str] = qty
            current += timedelta(days=1)

        return {
            "total_predicted": round(sum(daily.values()), 4),
            "daily_breakdown": daily
        }

    def list_items(self, branch: str = None):
        """Returns all available items, optionally filtered by branch."""
        df = self.df
        if branch:
            df = df[df['branch_upper'] == branch.upper().strip()]
        return sorted(df['item_name'].unique().tolist())


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    f = InventoryForecaster(script_dir)

    print("\n=== Test 1: Single day ===")
    print(f.predict_single_day("Conut - Tyre", "FULL FAT MILK", "2026-03-12"))

    print("\n=== Test 2: Full week ===")
    print(f.predict_date_range("Conut - Tyre", "FULL FAT MILK", "2026-03-12", "2026-03-18"))

    print("\n=== Test 3: Seasonality (March vs December at Jnah) ===")
    print("March:   ", f.predict_single_day("Conut Jnah", "NUTELLA SPREAD CONUT", "2026-03-12"))
    print("December:", f.predict_single_day("Conut Jnah", "NUTELLA SPREAD CONUT", "2026-12-12"))

    print("\n=== Test 4: Weekend lift ===")
    print("Tuesday:  ", f.predict_single_day("Conut", "CLASSIC CHIMNEY", "2026-03-10"))
    print("Saturday: ", f.predict_single_day("Conut", "CLASSIC CHIMNEY", "2026-03-15"))