import streamlit as st
from pymongo import MongoClient
import urllib.parse
import pandas as pd
import time  # ⏱️ Timing

# Use quote_plus to safely escape special characters
username = urllib.parse.quote_plus("kaonasty")
password = urllib.parse.quote_plus("B@ndung40175")

# Construct the URI safely
MONGO_URI = f"mongodb+srv://{username}:{password}@epigenomic.fygtvjs.mongodb.net/epigenomic_db?retryWrites=true&w=majority"

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client["epigenomic_db"]
regions_col = db["Region"]
features_col = db["EpigenomicFeatures"]

st.title("🧬 Epigenomic Data Viewer")

# -------------------------------
# 🔎 SEARCH REGION TOOL
# -------------------------------
st.subheader("🔎 Search Region by Chromosome & Position")

search_start = time.time()

input_chrom = st.text_input("Chromosome (e.g. chr10)", "chr10")
input_start = st.number_input("Start position (e.g. 100007762)", value=100007762)
input_type = st.selectbox("Region Type", ["promoter", "enhancer"])
input_size = st.selectbox("Window Size", [1024, 512, 256, 128, 64], key="search_window")

query = {
    "chrom": input_chrom,
    "region_type": input_type,
    "window_size": input_size
}

closest_region = regions_col.find(query).sort([("start", 1)]).limit(1000)
closest_match = min(closest_region, key=lambda x: abs(x["start"] - input_start), default=None)

if closest_match:
    st.success("🎯 Closest region found:")
    st.json(closest_match)

    st.subheader("📊 Epigenomic Features for this Region")
    feature = features_col.find_one({"region_id": closest_match["id"]})
    if feature:
        st.json(feature)
    else:
        st.warning("No matching features found for this region.")
else:
    st.error("No region found matching your criteria.")

search_end = time.time()
st.info(f"⏱️ Search completed in {search_end - search_start:.2f} seconds")

# -------------------------------
# 📦 EXTRACT TOOL
# -------------------------------
st.subheader("📦 Extract X and y Dataset")

extract_start = time.time()

model_type = st.selectbox("Model Type", [
    "Active Promoter vs Inactive Promoter",
    "Active Enhancer vs Inactive Enhancer",
    "Active Enhancer vs Active Promoter",
    "Inactive Enhancer vs Inactive Promoter"
], key="extract_model")

window_size = st.selectbox("Window Size", [1024, 512, 256, 128, 64], key="extract_window")

ALL_FEATURES = [
    "CTCF", "CTCFL", "FOXA1", "FOXA2", "FOXA3", "GATA1", "GATA2", "GATA3",
    "NR3C1", "ESR1", "ETS1", "ETS2", "CHD1", "CHD2", "CHD4", "CHD7", "ARID1B", "ARID2", "ARID3A", "ARID3B",
    "ARID4A", "ARID4B", "ARID5B", "SMARCA4", "SMARCB1", "EZH2", "EED", "SUZ12", "KDM1A", "KDM2A", "KDM3A", "KDM4B", "KDM5A",
    "KDM5B", "KDM6A", "KAT2A", "KAT2B", "KAT7", "KAT8", "HDAC1", "HDAC2",
    "HDAC3", "HDAC6", "HDAC8", "BRD4", "DNMT1", "DNMT3B", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2", "H3K4me3",
    "H3K79me2", "H3K9ac", "H3K9me2", "H3K9me3", "H4K20me1"
]

def get_conditions(model_type):
    if model_type == "Active Promoter vs Inactive Promoter":
        return [("promoter", 1), ("promoter", 0)]
    elif model_type == "Active Enhancer vs Inactive Enhancer":
        return [("enhancer", 1), ("enhancer", 0)]
    elif model_type == "Active Enhancer vs Active Promoter":
        return [("enhancer", 1), ("promoter", 1)]
    elif model_type == "Inactive Enhancer vs Inactive Promoter":
        return [("enhancer", 0), ("promoter", 0)]
    return []

all_data = []

for region_type, tpm_val in get_conditions(model_type):
    query = {
        "id": {"$regex": f"{region_type}_{window_size}"},
        "TPM": tpm_val
    }
    projection = {"TPM": 1, **{feat: 1 for feat in ALL_FEATURES}}
    docs = features_col.find(query, projection).limit(1000)

    for doc in docs:
        row = {"TPM": doc.get("TPM", 0)}
        for feat in ALL_FEATURES:
            row[feat] = doc.get(feat, float("nan"))
        all_data.append(row)

if all_data:
    df = pd.DataFrame(all_data)
    st.write(f"✅ Extracted {len(df)} rows, {len(df.columns)-1} features")
    st.dataframe(df.head())

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Dataset CSV",
        data=csv,
        file_name=f"dataset_{model_type.replace(' ', '_')}_{window_size}.csv",
        mime="text/csv"
    )
else:
    st.warning("⚠️ No matching documents found.")

extract_end = time.time()
st.info(f"⏱️ Extraction completed in {extract_end - extract_start:.2f} seconds")
