import os
import pandas as pd
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["epigenomic_db"]
collection = db["regions"]

# Path to CSVs
csv_folder = "E:/Final2/Max Data/Data insert"

# Ingest all CSVs
for file in os.listdir(csv_folder):
    if file.endswith(".csv"):
        path = os.path.join(csv_folder, file)

        # Extract meta info from filename
        parts = file.replace(".csv", "").split("_")
        cell_line, window_size, region_type = parts[0], int(parts[1]), parts[2]

        df = pd.read_csv(path)

        docs = []
        for _, row in df.iterrows():
            doc = {
                "cell_line": cell_line,
                "window_size": window_size,
                "region_type": region_type,
                "chrom": row["chrom"],
                "start": int(row["start"]),
                "end": int(row["end"]),
                "strand": row["strand"],
                "TPM": row["TPM"],
                "features": row.drop(["chrom", "start", "end", "strand", "TPM"]).to_dict()
            }
            docs.append(doc)

        # Insert in bulk
        if docs:
            collection.insert_many(docs)
