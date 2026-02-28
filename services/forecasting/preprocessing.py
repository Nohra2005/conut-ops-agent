import pandas as pd
import os
import calendar

def clean_item_name(name):
    if not isinstance(name, str): return name
    cleaned = name.strip().replace(' .', '.').replace(' ,', ',')
    cleaned = cleaned.rstrip('.').rstrip(',').strip()
    return cleaned.upper()

def run_preprocessing(data_dir, output_dir):
    print("Starting preprocessing...")

    # ── 1. Annual anchor (file 191) ──────────────────────────────────────────
    df_191 = pd.read_csv(os.path.join(data_dir, "rep_s_00191_SMRY.csv"),
                         dtype=str, on_bad_lines='skip', header=None)
    df_191['branch'] = pd.Series([None] * len(df_191), dtype='object')
    branch_mask = df_191[0].str.contains('Branch:', na=False, case=False)
    df_191.loc[branch_mask, 'branch'] = df_191.loc[branch_mask, 0].str.replace('Branch:', '', case=False).str.strip()
    df_191['branch'] = df_191['branch'].ffill()
    df_191['qty'] = pd.to_numeric(df_191[2].str.replace(',','').str.replace('"',''), errors='coerce')
    anchor = df_191.dropna(subset=['qty']).copy()
    anchor = anchor[~anchor[0].str.contains('Total|Group:|Division:|Page|Description|Branch', na=False, case=False)]
    anchor.rename(columns={0: 'item_name'}, inplace=True)
    anchor['item_name'] = anchor['item_name'].apply(clean_item_name)
    anchor = anchor.groupby(['branch', 'item_name'])['qty'].sum().reset_index()
    anchor.rename(columns={'qty': 'annual_qty'}, inplace=True)
    print(f"  Anchor: {len(anchor)} branch-item pairs loaded.")

    # ── 2. Monthly weights (file 334) ────────────────────────────────────────
    df_334 = pd.read_csv(os.path.join(data_dir, "rep_s_00334_1_SMRY.csv"),
                         dtype=str, on_bad_lines='skip', header=None)
    df_334['branch'] = pd.Series([None] * len(df_334), dtype='object')
    bname_mask = df_334[0].str.startswith('Branch Name:', na=False)
    df_334.loc[bname_mask, 'branch'] = df_334.loc[bname_mask, 0].str.replace('Branch Name:', '', case=False).str.strip()
    df_334['branch'] = df_334['branch'].ffill()
    months = list(calendar.month_name)[1:]
    m_weights = df_334[df_334[0].isin(months)].copy()
    m_weights.rename(columns={0: 'month', 3: 'revenue'}, inplace=True)
    m_weights['revenue'] = pd.to_numeric(m_weights['revenue'].str.replace(',','').str.replace('"',''), errors='coerce')

    # Remove anomalous months — below 10% of branch average (catches Conut Dec crash)
    branch_avg = m_weights.groupby('branch')['revenue'].transform('mean')
    m_weights = m_weights[m_weights['revenue'] >= branch_avg * 0.1].copy()

    # Compute monthly weight = this month's revenue / branch total revenue
    branch_totals = m_weights.groupby('branch')['revenue'].sum().reset_index(name='annual_rev')
    m_weights = pd.merge(m_weights, branch_totals, on='branch')
    m_weights['monthly_weight'] = m_weights['revenue'] / m_weights['annual_rev']

    # Fill missing months with branch average so all 12 months are covered
    filled = []
    for branch, g in m_weights.groupby('branch'):
        avg_w = g['monthly_weight'].mean()
        g_idx = g.set_index('month')
        for month in months:
            w = g_idx.loc[month, 'monthly_weight'] if month in g_idx.index else avg_w
            filled.append({'branch': branch, 'month': month, 'monthly_weight': w})
    m_weights_full = pd.DataFrame(filled)
    print(f"  Monthly weights: {len(m_weights_full)} rows (all 12 months × all branches).")

    # ── 3. DOW seasonality (file 461) ────────────────────────────────────────
    df_461 = pd.read_csv(os.path.join(data_dir, "REP_S_00461.csv"),
                         dtype=str, on_bad_lines='skip', header=None)
    date_mask = df_461[0].str.match(r'\d{2}-[a-zA-Z]{3}-\d{2}', na=False)
    staff_df = df_461[date_mask].copy()
    staff_df['date'] = pd.to_datetime(staff_df[0], format='%d-%b-%y')

    def parse_dur(ts):
        try:
            h, m, s = map(int, str(ts).split('.')); return h + m/60 + s/3600
        except: return 0.0

    staff_df['hrs'] = staff_df[5].apply(parse_dur)
    dow_counts = staff_df.groupby(staff_df['date'].dt.day_name())['hrs'].sum()
    all_days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    for day in all_days:
        if day not in dow_counts.index: dow_counts[day] = dow_counts.mean()
    dow_weights = (dow_counts / dow_counts.sum()).to_dict()
    print(f"  DOW weights: {', '.join(f'{d}: {v:.3f}' for d, v in sorted(dow_weights.items()))}")

    # ── 4. Build full year daily forecast ────────────────────────────────────
    # Formula: daily_qty = annual_qty × monthly_weight × normalized_dow_weight
    # normalized_dow_weight ensures all days in the month sum to exactly 1.0
    merged = pd.merge(anchor, m_weights_full, on='branch')

    date_rng = pd.date_range(start='2026-01-01', end='2026-12-31')
    master_cal = pd.DataFrame({'date': date_rng})
    master_cal['month']    = master_cal['date'].dt.month_name()
    master_cal['day_name'] = master_cal['date'].dt.day_name()
    master_cal['raw_w']    = master_cal['day_name'].map(dow_weights)

    # Normalize per month so weights sum to 1 within each month
    m_sum = master_cal.groupby('month')['raw_w'].transform('sum')
    master_cal['norm_w'] = master_cal['raw_w'] / m_sum

    final = pd.merge(master_cal, merged, on='month')
    final['predicted_qty'] = (final['annual_qty'] * final['monthly_weight'] * final['norm_w']).round(4)
    final['predicted_qty'] = final['predicted_qty'].clip(lower=0)

    new_data = final[['date','branch','item_name','predicted_qty']].copy()
    new_data['date'] = new_data['date'].dt.strftime('%Y-%m-%d')

    # ── 5. Append and deduplicate ─────────────────────────────────────────────
    output_file = os.path.join(output_dir, "master_daily_inventory.csv")
    if os.path.exists(output_file):
        print("  Existing master file found — merging...")
        existing = pd.read_csv(output_file)
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=['date','branch','item_name'], keep='last')
        combined = combined.sort_values('date')
        combined.to_csv(output_file, index=False)
        print(f"  Updated. Total rows: {len(combined)}")
    else:
        new_data.to_csv(output_file, index=False)
        print(f"  Created master_daily_inventory.csv with {len(new_data)} rows.")

    print("Preprocessing complete.")
    return new_data


if __name__ == "__main__":
    BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
    DATA_FOLDER   = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "data"))
    OUTPUT_FOLDER = BASE_DIR
    try:
        run_preprocessing(DATA_FOLDER, OUTPUT_FOLDER)
    except Exception as e:
        print(f"Error: {e}")
        raise