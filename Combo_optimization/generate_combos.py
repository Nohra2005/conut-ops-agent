import json
import pandas as pd
import numpy as np

def generate_combo_pricing_strategy():
    # 1. Load the pricing data
    with open('cleaned_items_prices.json', 'r') as f:
        prices_data = json.load(f)
    df_prices = pd.DataFrame(prices_data)

    # Create a dictionary for quick price lookups: { 'item_name': unit_price }
    price_dict = df_prices.set_index('item_name')['unit_price'].to_dict()

    # 2. Load the Market Basket Analysis (Combo) results
    with open('combo_results.json', 'r') as f:
        combo_data = json.load(f)

    # Extract the list of combo recommendations
    df_combos = pd.DataFrame(combo_data['combo_recommendations'])

    # 3. Define the pricing and discount logic
    def process_combo(row):
        # Extract base item and add-on item from the lists
        item_a = row['if_buys'][0]
        item_b = row['also_buys'][0]
        
        # Look up prices (default to 0 if not found)
        price_a = price_dict.get(item_a, 0)
        price_b = price_dict.get(item_b, 0)
        sum_price = price_a + price_b
        
        # Convert confidence from percentage to decimal
        conf = row['confidence_pct'] / 100.0
        
        # Discount logic based on Cannibalization Risk
        if conf >= 0.40:
            discount = 0.05
            risk_level = "High Risk (5% off)"
        elif conf >= 0.20:
            discount = 0.12
            risk_level = "Medium Risk (12% off)"
        else:
            discount = 0.18
            risk_level = "Low Risk (18% off)"
            
        combo_price = sum_price * (1 - discount)
        
        # Calculate target breakeven volume using current customer count as a baseline proxy
        current_volume = row['n_customers']
        breakeven_volume = np.ceil(current_volume / (1 - discount)).astype(int)
        
        return pd.Series({
            'Base Item': item_a,
            'Add-on Item': item_b,
            'Confidence (%)': row['confidence_pct'],
            'Lift': row['lift'],
            'Orig Price': sum_price,
            'Discount %': int(discount * 100),
            'Promo Price': combo_price,
            'Risk Level': risk_level,
            'Current Vol': current_volume,
            'Target Breakeven': breakeven_volume
        })

    # 4. Apply the logic to the dataframe
    df_analysis = df_combos.apply(process_combo, axis=1)

    # Sort by Lift to prioritize the combos with the strongest natural relationships
    df_analysis = df_analysis.sort_values(by='Lift', ascending=False)

    # Print the clean output
    print("--- 🚀 Conut Combo Pricing Strategy ---")
    print(df_analysis.to_string(index=False))

    # Save to CSV for the OpenClaw Agent to use later
    df_analysis.to_csv('final_combo_pricing_strategy.csv', index=False)
    print("\n✅ Saved to 'final_combo_pricing_strategy.csv'")

if __name__ == "__main__":
    generate_combo_pricing_strategy()