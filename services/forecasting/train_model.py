import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import os
import numpy as np

def train_forecasting_model(data_path, model_dir):
    print("Loading master dataset...")
    df = pd.read_csv(data_path)
    
    # 1. Feature engineering
    print("Engineering date features...")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date') # Crucial for chronological splitting
    
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    
    # 2. Encode categorical variables
    print("Encoding branches and items...")
    branch_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    df['branch_encoded'] = branch_encoder.fit_transform(df['branch'])
    df['item_encoded'] = item_encoder.fit_transform(df['item_name'])
    
    # 3. Dynamic Chronological Split
    # We take the unique sorted dates and find the 80% mark
    print("Calculating dynamic split point...")
    unique_dates = sorted(df['date'].unique())
    split_index = int(len(unique_dates) * 0.8)
    split_date = unique_dates[split_index]
    
    print(f"Training on data before: {split_date.date()}")
    print(f"Testing on data after:  {split_date.date()}")
    
    train_df = df[df['date'] < split_date]
    test_df = df[df['date'] >= split_date]
    
    features = ['branch_encoded', 'item_encoded', 'month', 'day_of_week', 'day_of_month']
    target = 'daily_qty_rounded'
    
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    # 4. Train the XGBoost model
    print("Training cumulative XGBoost regressor...")
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        objective='reg:squarederror'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    
    # 5. Evaluate
    print("\nEvaluating model on final 20% of timeline...")
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"Mean absolute error (MAE): {mae:.2f} units")
    print(f"Root mean squared error (RMSE): {rmse:.2f} units")
    
    # 6. Save the model and encoders
    print("Saving model and encoders...")
    os.makedirs(model_dir, exist_ok=True)
    
    model.save_model(os.path.join(model_dir, "xgb_inventory_model.json"))
    
    with open(os.path.join(model_dir, "branch_encoder.pkl"), "wb") as f:
        pickle.dump(branch_encoder, f)
        
    with open(os.path.join(model_dir, "item_encoder.pkl"), "wb") as f:
        pickle.dump(item_encoder, f)
        
    print(f"Success! Model files updated in {model_dir}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "master_daily_inventory.csv")
    
    try:
        train_forecasting_model(data_path, script_dir)
    except Exception as e:
        print(f"Error training model: {e}")