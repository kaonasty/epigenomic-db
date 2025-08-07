# streamlit_mongo_ingest.py

import os
import pandas as pd
import streamlit as st
from pymongo import MongoClient
import uuid

# --- Streamlit UI ---
st.title("CSV to MongoDB Ingestion")

# Select CSV folder
csv_folder = st.text_input("Enter the path to the folder containing CSV files:", "E:/Final2/Max Data/Data insert")

if st.button("Start Ingestion"):

    if not os.path.exists(csv_folder):
        st.error("Folder does not exist.")
    else:
        # MongoDB setup
        client = MongoClient("mongodb://localhost:27017/")
        db = client["epigenomic_reduction"]
        region_collection = db["Region"]
        features_collection = db["EpigenomicFeatures"]

        inserted_files = 0

        for file in os.listdir(csv_folder):
            if file.endswith(".csv"):
                path = os.path.join(csv_folder, file)

                try:
                    parts = file.replace(".csv", "").split("_")
                    cell_line, window_size, region_type = parts[0], int(parts[1]), parts[2]
                except Exception as e:
                    st.warning(f"Skipped {file}: Filename format incorrect. Expected format: CellLine_WindowSize_RegionType.csv")
                    continue

                df = pd.read_csv(path)
                region_docs = []
                feature_docs = []

                for _, row in df.iterrows():
                    region_id = str(uuid.uuid4())  # Unique ID for this region

                    # Region document
                    region_doc = {
                        "id": region_id,
                        "chrom": row["chrom"],
                        "start": int(row["start"]),
                        "end": int(row["end"]),
                        "strand": row["strand"],
                        "cell_line": cell_line,
                        "region_type": region_type,
                        "window_size": int(window_size)
                    }
                    region_docs.append(region_doc)

                    # Feature document
                    feature_doc = {
                        "id": region_id + "_f",
                        "region_id": region_id,
                        "TPM": float(row["TPM"])
                    }

                    # Optional: Add all other epigenomic features dynamically
                    features_only = row.drop(["chrom", "start", "end", "strand", "TPM"])
                    feature_doc.update(features_only.to_dict())

                    feature_docs.append(feature_doc)

                # Insert in bulk
                if region_docs:
                    region_collection.insert_many(region_docs)
                    features_collection.insert_many(feature_docs)
                    inserted_files += 1
                    st.success(f"Inserted {file} ({len(region_docs)} regions)")

        st.info(f"✅ Done. Processed and inserted data from {inserted_files} file(s).")
