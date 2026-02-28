import pandas as pd

# Load the file, skipping the first 3 rows which contain report metadata
file_path = "rep_s_00191_SMRY.csv"
df = pd.read_csv(file_path, skiprows=3)

# Strip whitespace from column names 
df.columns = df.columns.str.strip()

# Remove rows with empty Descriptions
df = df.dropna(subset=['Description'])

# Remove structural report rows (Branch, Division, Group, Total)
prefixes_to_remove = ('Branch:', 'Division:', 'Group:', 'Total', 'Page', 'Sales')
df = df[~df['Description'].str.strip().str.startswith(prefixes_to_remove)]

# Filter out repeated header rows and missing quantity data
df = df[df['Qty'] != 'Qty']
df = df[df['Total Amount'] != 'Total Amount']
df = df.dropna(subset=['Qty', 'Total Amount'])

# Clean Total Amount by removing commas and casting to float
df['Total Amount'] = df['Total Amount'].astype(str).str.replace(',', '').astype(float)
df['Qty'] = df['Qty'].astype(float)

# Group by Item Name in case the same item is sold across different branches
agg_df = df.groupby('Description').agg(
    total_quantity=('Qty', 'sum'),
    total_amount=('Total Amount', 'sum')
).reset_index()

# Compute the unit price: Total Amount / Quantity Ordered
agg_df['unit_price'] = (agg_df['total_amount'] / agg_df['total_quantity']).round(2)

# Select and rename final columns for the JSON output
final_df = agg_df[['Description', 'unit_price', 'total_quantity']].copy()
final_df.columns = ['item_name', 'unit_price', 'quantity_sold']

# Sort by quantity sold descending
final_df = final_df.sort_values(by='quantity_sold', ascending=False)

# Save to JSON
json_path = "cleaned_items_prices.json"
final_df.to_json(json_path, orient='records', indent=4)

print(f"Data successfully cleaned and saved to {json_path}")