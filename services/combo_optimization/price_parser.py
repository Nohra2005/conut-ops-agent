import pandas as pd
import os

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "rep_s_00191_SMRY.csv")
json_path = os.path.join(BASE_DIR, "cleaned_items_prices.json")

df = pd.read_csv(file_path, skiprows=3)
df.columns = df.columns.str.strip()
df = df.dropna(subset=['Description'])

prefixes_to_remove = ('Branch:', 'Division:', 'Group:', 'Total', 'Page', 'Sales')
df = df[~df['Description'].str.strip().str.startswith(prefixes_to_remove)]
df = df[df['Qty'] != 'Qty']
df = df[df['Total Amount'] != 'Total Amount']
df = df.dropna(subset=['Qty', 'Total Amount'])

df['Total Amount'] = df['Total Amount'].astype(str).str.replace(',', '').astype(float)
df['Qty'] = df['Qty'].astype(float)

agg_df = df.groupby('Description').agg(
    total_quantity=('Qty', 'sum'),
    total_amount=('Total Amount', 'sum')
).reset_index()

agg_df['unit_price'] = (agg_df['total_amount'] / agg_df['total_quantity']).round(2)
final_df = agg_df[['Description', 'unit_price', 'total_quantity']].copy()
final_df.columns = ['item_name', 'unit_price', 'quantity_sold']
final_df = final_df.sort_values(by='quantity_sold', ascending=False)
final_df.to_json(json_path, orient='records', indent=4)
print(f"Saved to {json_path}")