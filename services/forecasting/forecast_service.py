import pandas as pd
import xgboost as xgb
import pickle
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class InventoryForecaster:
    def __init__(self, model_dir):
        """Loads the trained model and translators into memory."""
        self.model = xgb.XGBRegressor()
        self.model.load_model(os.path.join(model_dir, "xgb_inventory_model.json"))
        
        with open(os.path.join(model_dir, "branch_encoder.pkl"), "rb") as f:
            self.branch_encoder = pickle.load(f)
            
        with open(os.path.join(model_dir, "item_encoder.pkl"), "rb") as f:
            self.item_encoder = pickle.load(f)
            
    def predict_single_day(self, branch_name, item_name, target_date_str):
        """Predicts inventory for a single specific date (YYYY-MM-DD)."""
        try:
            target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
            
            # Translate text to numbers
            branch_encoded = self.branch_encoder.transform([branch_name])[0]
            item_encoded = self.item_encoder.transform([item_name])[0]
            
            # Extract date features the model needs
            month = target_date.month
            day_of_week = target_date.weekday()
            day_of_month = target_date.day
            
            # Create the exact feature structure the model expects
            X_infer = pd.DataFrame(
                [[branch_encoded, item_encoded, month, day_of_week, day_of_month]], 
                columns=['branch_encoded', 'item_encoded', 'month', 'day_of_week', 'day_of_month']
            )
            
            # Make the prediction
            prediction = self.model.predict(X_infer)[0]
            
            # Ensure we don't return negative inventory
            return max(0.0, round(float(prediction), 2))
            
        except ValueError as e:
            return f"Error: Make sure the branch and item exist in the data. Details: {e}"
        except Exception as e:
            return f"Prediction error: {e}"

    def predict_date_range(self, branch_name, item_name, start_date_str, end_date_str):
        """Predicts total inventory needed over a span of days (like 'next week')."""
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
            
            current_date = start_date
            total_qty = 0.0
            daily_breakdown = {}
            
            # Loop through each day in the range and predict
            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                qty = self.predict_single_day(branch_name, item_name, date_str)
                
                if isinstance(qty, str):  # If it returned an error string
                    return qty
                    
                daily_breakdown[date_str] = qty
                total_qty += qty
                current_date += timedelta(days=1)
                
            return {
                "total_requested": round(total_qty, 2),
                "daily_breakdown": daily_breakdown
            }
            
        except Exception as e:
            return f"Range prediction error: {e}"
        
    def visualize_forecast(self, forecast_data, item_name, branch_name):
        """Generates a bar chart for the daily breakdown."""
        dates = list(forecast_data['daily_breakdown'].keys())
        values = list(forecast_data['daily_breakdown'].values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(dates, values, color='skyblue', edgecolor='navy')
        
        # Add labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, yval, ha='center', va='bottom')

        plt.title(f"Inventory forecast: {item_name} at {branch_name}")
        plt.xlabel("Date")
        plt.ylabel("Units needed")
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the chart as an image for the agent/teammate
        chart_path = f"forecast_{branch_name.replace(' ', '_')}.png"
        plt.savefig(chart_path)
        print(f"\nChart saved successfully as {chart_path}")
        plt.show()

# --- Testing the service ---
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    forecaster = InventoryForecaster(script_dir)
    
    test_branch = "Conut - Tyre"
    test_item = "FULL FAT MILK"
    
    print("Test 1: Single day (March 12, 2026)")
    single_result = forecaster.predict_single_day(test_branch, test_item, "2026-03-12")
    print(f"Prediction: {single_result} units\n")
    
    print("Test 2: A full week (March 12 to March 18)")
    week_result = forecaster.predict_date_range(test_branch, test_item, "2026-03-12", "2026-03-18")
    print(f"Total needed for the week: {week_result['total_requested']} units")
    print(f"Daily breakdown: {week_result['daily_breakdown']}")
    
    # Test Case: High vs Low Season at Jnah
    print(f"Jnah March 12: {forecaster.predict_single_day('Conut Jnah', 'NUTELLA SPREAD CONUT', '2026-03-12')}")
    print(f"Jnah Dec 12:   {forecaster.predict_single_day('Conut Jnah', 'NUTELLA SPREAD CONUT', '2026-12-12')}")

    # Test Case: Weekend Lift
    print(f"Tuesday March 10:  {forecaster.predict_single_day('Conut', 'CLASSIC CHIMNEY', '2026-03-10')}")
    print(f"Saturday March 14: {forecaster.predict_single_day('Conut', 'CLASSIC CHIMNEY', '2026-03-14')}")
    forecaster.visualize_forecast(week_result, test_item, test_branch)