import os
import pandas as pd
import mysql.connector
import streamlit as st
import io

# ------------------------
# 🔧 Config
# ------------------------
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'B@ndung40175',
    'database': 'epigenomicdb'
}

WINDOW_SIZE = 256
BATCH_SIZE = 500

# ------------------------
# 🧠 Connect to MySQL
# ------------------------
conn = mysql.connector.connect(**DB_CONFIG)
cursor = conn.cursor()

# ------------------------
# 🔁 Cache Feature IDs
# ------------------------
feature_id_cache = {}

def get_or_insert_feature(feature_name):
    if feature_name in feature_id_cache:
        return feature_id_cache[feature_name]

    cursor.execute("SELECT feature_id FROM feature WHERE feature_name=%s", (feature_name,))
    result = cursor.fetchone()
    if result:
        feature_id_cache[feature_name] = result[0]
        return result[0]

    cursor.execute("INSERT INTO feature (feature_name) VALUES (%s)", (feature_name,))
    conn.commit()
    feature_id = cursor.lastrowid
    feature_id_cache[feature_name] = feature_id
    return feature_id

def parse_filename(filename):
    base = os.path.basename(filename)
    parts = base.split('_')
    return parts[0], int(parts[1]), parts[2].replace('.csv', '')  # cell_line, window_size, region_type

# ------------------------
# 📦 Streamlit UI
# ------------------------
st.title("📥 Epigenomic CSV Upload to MySQL")

uploaded_files = st.file_uploader("Upload one or more processed CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Extract metadata from filename
        cell_line, window_size, region_type = parse_filename(uploaded_file.name)
        if window_size != WINDOW_SIZE:
            st.warning(f"Skipping {uploaded_file.name}: window size not {WINDOW_SIZE}")
            continue

        st.info(f"Processing: {uploaded_file.name}")
        df = pd.read_csv(uploaded_file)
        feature_cols = [col for col in df.columns if col not in ['chrom', 'start', 'end', 'strand', 'TPM']]

        total_rows = len(df)
        for start_idx in range(0, total_rows, BATCH_SIZE):
            batch = df.iloc[start_idx:start_idx + BATCH_SIZE]
            region_data = []

            for _, row in batch.iterrows():
                tpm_value = int(row['TPM']) if pd.notna(row['TPM']) else None
                region_data.append((
                    row['chrom'], int(row['start']), int(row['end']), row['strand'],
                    cell_line, region_type.capitalize(), window_size, tpm_value
                ))

            cursor.executemany("""
                INSERT INTO region (chrom, start, end, strand, cell_line, region_type, window_size, TPM)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, region_data)
            conn.commit()

            # Get the starting region_id of inserted batch
            cursor.execute("SELECT LAST_INSERT_ID()")
            start_region_id = cursor.fetchone()[0]
            region_ids = list(range(start_region_id, start_region_id + len(region_data)))

            region_feature_data = []
            for row_idx, region_id in enumerate(region_ids):
                for feature in feature_cols:
                    value = batch.iloc[row_idx][feature]
                    if pd.isna(value):
                        continue
                    feature_id = get_or_insert_feature(feature)
                    region_feature_data.append((region_id, feature_id, float(value)))

            if region_feature_data:
                cursor.executemany("""
                    INSERT INTO region_feature (region_id, feature_id, value)
                    VALUES (%s, %s, %s)
                """, region_feature_data)
                conn.commit()

        st.success(f"✅ Successfully ingested {uploaded_file.name}")

# ------------------------
# ✅ Clean up
# ------------------------
cursor.close()
conn.close()
