# streamlit_app.py
import tempfile
import joblib
import numpy as np
from sklearn.metrics import accuracy_score
import streamlit as st
import pandas as pd
import os
import time
from sqlalchemy import (
    text,
    insert,
    Table,
    Column,
    Integer,
    Float,
    String,
    DateTime,
    MetaData
)
from data_loader import (
    get_filtered_data,
    get_all_cell_lines,
    get_region_types,
    get_all_features,
    count_matching_rows,
    get_sqlalchemy_engine
)
from dataset_utils import prepare_X_y
from train_model import train_ffnn_streaming

st.set_page_config(page_title="Epigenomic FFNN", layout="wide")
st.title("🧬 Epigenomic FFNN Training WebApp")

# === Section 1: SQL Query ===
st.header("🔍 Explore Database")

engine = get_sqlalchemy_engine()
metadata = MetaData()

model_log_table = Table(
    "model_training_log",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("timestamp", DateTime),
    Column("task", String(255)),
    Column("selected_cell_lines", String),
    Column("selected_features", String),
    Column("row_limit", Integer),
    Column("batch_size", Integer),
    Column("train_duration", Float),
    Column("data_duration", Float),
    Column("accuracy", Float),
    autoload_with=engine  # Automatically maps to existing table schema
)

with engine.connect() as conn:
    # Show number of tables and their names
    table_list = conn.execute(text("SHOW TABLES")).fetchall()
    table_names = [row[0] for row in table_list]
    st.subheader("📦 Database Tables")
    st.write(f"Total tables: {len(table_names)}")
    st.dataframe(pd.DataFrame(table_names, columns=["Table Name"]))

    # SQL query input from user
    st.subheader("📝 Custom SQL Query")
    user_sql = st.text_area("Enter your SQL query below (e.g., SELECT * FROM region LIMIT 5)", height=150)
    if st.button("Run Query"):
        try:
            result_df = pd.read_sql(user_sql, conn)
            st.success("Query executed successfully!")
            st.dataframe(result_df)
        except Exception as e:
            st.error(f"Error running query: {e}")

# === Section 2: User Selection for FFNN Tasks ===
st.header("🤖 Train FFNN Model")
region_choices = {
    "IE vs IP": ("enhancer", "promoter", 0.0, 0.0),
    "AP vs IP": ("promoter", "promoter", 1.0, 0.0),
    "AE vs IE": ("enhancer", "enhancer", 1.0, 0.0),
    "AE vs AP": ("enhancer", "promoter", 1.0, 1.0),
}

selected_task = st.selectbox("Select Classification Task", list(region_choices.keys()))

region1, region2, tpm1, tpm2 = region_choices[selected_task]

cell_lines = get_all_cell_lines()
selected_cell_lines = st.multiselect("Select Cell Lines", cell_lines)

features = get_all_features()
selected_features = st.multiselect("Select Features (leave empty to use all)", features)

# Default to all features if none selected
if not selected_features:
    selected_features = features

row_limit = st.number_input("Total rows to fetch (dataset size)", value=50000, step=1000)
batch_size = st.number_input("Batch size for training", value=5000, step=1000)

if st.button("Start Training from MySQL"):
    if not selected_cell_lines:
        st.warning("Please select at least one cell line.")
    else:
        start_data_time = time.time()
        full_df = get_filtered_data(
            cell_lines=selected_cell_lines,
            feature_names=selected_features,
            region_type=[region1, region2],
            TPM=min(tpm1, tpm2),
            limit=row_limit
        )
        data_duration = time.time() - start_data_time

        X, y = prepare_X_y(full_df)
        st.write(f"Dataset shape: X={X.shape}, y={y.shape}")

        start_train_time = time.time()
        with st.spinner("Training FFNN model in batches..."):
            model, history = train_ffnn_streaming(X, y, batch_size=batch_size)
        train_duration = time.time() - start_train_time

        y_pred = (model.predict(X) > 0.5).astype(int)
        accuracy = accuracy_score(y, y_pred)
        st.metric(label="🔍 Model Accuracy", value=f"{accuracy:.4f}")

        # Save log to database
        log_entry = {
            "timestamp": pd.Timestamp.now(),
            "task": selected_task,
            "selected_cell_lines": ", ".join(selected_cell_lines),
            "selected_features": ", ".join(selected_features),
            "row_limit": row_limit,
            "batch_size": batch_size,
            "train_duration": train_duration,
            "data_duration": data_duration,
            "accuracy": accuracy
        }

        with engine.connect() as conn:
            conn.execute(
                insert(model_log_table),
                [log_entry]
            )
            conn.commit()

st.header("📊 Training Log History")

if st.button("Load Training Logs"):
    with engine.connect() as conn:
        logs_df = pd.read_sql("SELECT * FROM model_training_log ORDER BY timestamp DESC", conn)
        st.dataframe(logs_df)

st.caption("📘 Marco William Langi · 23523033 · Thesis 2025")