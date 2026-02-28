import pandas as pd
import numpy as np
import os
import calendar

def clean_item_name(name):
    """standardizes item names by removing inconsistent punctuation and spacing."""
    if not isinstance(name, str): 
        return name
    # remove trailing dots, commas, and extra spaces
    cleaned = name.strip().replace(' .', '.').replace(' ,', ',')
    cleaned = cleaned.rstrip('.').rstrip(',').strip()
    return cleaned.upper()

def run_preprocessing(data_dir, output_dir):
    print("starting unified preprocessing...")

    # --- 1. annual anchor (file 191) ---
    path_191 = os.path.join(data_dir, "rep_s_00191_SMRY.csv")
    if not os.path.exists(path_191):
        raise FileNotFoundError(f"could not find file: {path_191}")
        
    df_191 = pd.read_csv(path_191, dtype=str, on_bad_lines='skip', header=None)
    
    # use object dtype to prevent LossySetitemError/TypeError
    df_191['branch'] = pd.Series([None] * len(df_191), dtype='object')
    
    branch_mask = df_191[0].str.contains('Branch:', na=False, case=False)
    df_191.loc[branch_mask, 'branch'] = df_191.loc[branch_mask, 0].str.replace('Branch:', '', case=False).str.strip()
    df_191['branch'] = df_191['branch'].ffill()
    
    df_191['qty'] = pd.to_numeric(df_191[2].str.replace(',', '').str.replace('"', ''), errors='coerce')
    
    anchor = df_191.dropna(subset=['qty']).copy()
    anchor = anchor[~anchor[0].str.contains('Total|Group:|Division:|Page|Description|Branch', na=False, case=False)]
    anchor.rename(columns={0: 'item_name'}, inplace=True)
    anchor['item_name'] = anchor['item_name'].apply(clean_item_name)
    
    # group by cleaned names to merge duplicates (e.g. "STRAWBERRY." and "STRAWBERRY")
    anchor = anchor.groupby(['branch', 'item_name'])['qty'].sum().reset_index()
    anchor.rename(columns={'qty': 'annual_qty'}, inplace=True)

    # --- 2. monthly weights (file 334) ---
    path_334 = os.path.join(data_dir, "rep_s_00334_1_SMRY.csv")
    if not os.path.exists(path_334):
        raise FileNotFoundError(f"could not find file: {path_334}")
        
    df_334 = pd.read_csv(path_334, dtype=str, on_bad_lines='skip', header=None)
    
    df_334['branch'] = pd.Series([None] * len(df_334), dtype='object')
    branch_name_mask = df_334[0].str.startswith('Branch Name:', na=False)
    df_334.loc[branch_name_mask, 'branch'] = df_334.loc[branch_name_mask, 0].str.replace('Branch Name:', '', case=False).str.strip()
    df_334['branch'] = df_334['branch'].ffill()
    
    months = list(calendar.month_name)[1:]
    m_weights = df_334[df_334[0].isin(months)].copy()
    m_weights.rename(columns={0: 'month', 3: 'revenue'}, inplace=True)
    m_weights['revenue'] = pd.to_numeric(m_weights['revenue'].str.replace(',', '').str.replace('"', ''), errors='coerce')
    
    branch_totals = m_weights.groupby('branch')['revenue'].sum().reset_index(name='annual_rev')
    m_weights = pd.merge(m_weights, branch_totals, on='branch')
    m_weights['monthly_weight'] = m_weights['revenue'] / m_weights['annual_rev']

    # --- 3. dow seasonality (file 461) ---
    path_461 = os.path.join(data_dir, "REP_S_00461.csv")
    if not os.path.exists(path_461):
        raise FileNotFoundError(f"could not find file: {path_461}")
        
    df_461 = pd.read_csv(path_461, dtype=str, on_bad_lines='skip', header=None)
    date_mask = df_461[0].str.match(r'\d{2}-[a-zA-Z]{3}-\d{2}', na=False)
    staff_df = df_461[date_mask].copy()
    staff_df['date'] = pd.to_datetime(staff_df[0], format='%d-%b-%y')
    
    def parse_dur(ts):
        try: 
            h, m, s = map(int, str(ts).split('.'))
            return h + m/60 + s/3600
        except: return 0.0
    
    staff_df['hrs'] = staff_df[5].apply(parse_dur)
    dow_counts = staff_df.groupby(staff_df['date'].dt.day_name())['hrs'].sum()
    dow_weights = (dow_counts / dow_counts.sum()).to_dict()

    # --- 4. final master assembly (2026 forecast) ---
    merged = pd.merge(anchor, m_weights[['branch', 'month', 'monthly_weight']], on='branch')
    date_rng = pd.date_range(start='2026-01-01', end='2026-12-31')
    master_cal = pd.DataFrame({'date': date_rng})
    master_cal['month'] = master_cal['date'].dt.month_name()
    master_cal['day_name'] = master_cal['date'].dt.day_name()
    master_cal['raw_w'] = master_cal['day_name'].map(dow_weights)
    
    m_sum = master_cal.groupby('month')['raw_w'].transform('sum')
    master_cal['norm_w'] = master_cal['raw_w'] / m_sum
    
    final = pd.merge(master_cal, merged, on='month')
    final['daily_qty_rounded'] = (final['annual_qty'] * final['monthly_weight'] * final['norm_w']).round(2)
    
    output_file = os.path.join(output_dir, "master_daily_inventory.csv")
    final[['date', 'branch', 'item_name', 'daily_qty_rounded']].to_csv(output_file, index=False)
    print(f"preprocessing complete. master_daily_inventory.csv generated at {output_file}")

if __name__ == "__main__":
    # dynamic path resolution
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # navigates from services/forecasting to the root data folder
    DATA_FOLDER = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "data"))
    OUTPUT_FOLDER = BASE_DIR
    
    try:
        run_preprocessing(DATA_FOLDER, OUTPUT_FOLDER)
    except Exception as e:
        print(f"error during preprocessing: {e}")