import pandas as pd
import os
from datetime import datetime, timedelta

class InventoryForecaster:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        master_path = os.path.join(data_dir, "master_daily_inventory.csv")
        if not os.path.exists(master_path):
            raise FileNotFoundError("master_daily_inventory.csv not found. Run preprocessing.py first.")
        self.df = pd.read_csv(master_path)
        self.df['date'] = pd.to_datetime(self.df['date'])

        # ── Auto-detect quantity column ──────────────────────────────────
        print(f"CSV columns found: {list(self.df.columns)}")
        qty_candidates = [c for c in self.df.columns if any(k in c.lower() for k in ['qty', 'quantity', 'predicted', 'units'])]
        if qty_candidates:
            self.qty_col = qty_candidates[0]
        else:
            # Last resort: use the last numeric column
            numeric_cols = self.df.select_dtypes(include='number').columns.tolist()
            self.qty_col = numeric_cols[-1] if numeric_cols else self.df.columns[-1]
        print(f"Using quantity column: '{self.qty_col}'")

        # Normalize for case-insensitive lookup
        self.df['branch_upper'] = self.df['branch'].str.upper().str.strip()
        self.df['item_upper']   = self.df['item_name'].str.upper().str.strip()
        print(f"Forecaster loaded: {len(self.df)} rows, "
              f"{self.df['branch'].nunique()} branches, "
              f"{self.df['item_name'].nunique()} items.")

    def predict_single_day(self, branch: str, item: str, date_str: str):
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
            return self._fallback_predict(branch, item, target)

        return round(float(rows.iloc[0][self.qty_col]), 4)

    def _fallback_predict(self, branch: str, item: str, target):
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
            mask2 = (
                (self.df['branch_upper'] == branch.upper().strip()) &
                (self.df['item_upper']   == item.upper().strip())
            )
            rows = self.df[mask2]
            if rows.empty:
                return f"Error: branch '{branch}' or item '{item}' not found in data."

        return round(float(rows[self.qty_col].mean()), 4)

    def predict_date_range(self, branch: str, item: str, start_str: str, end_str: str):
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
        df = self.df
        if branch:
            df = df[df['branch_upper'] == branch.upper().strip()]
        return sorted(df['item_name'].unique().tolist())


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    f = InventoryForecaster(script_dir)
    print("\n=== Test ===")
    print(f.predict_single_day("Conut - Tyre", "FULL FAT MILK", "2026-03-12"))