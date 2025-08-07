import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import joblib
import time
from datetime import datetime
from sklearn.metrics import accuracy_score
from pymongo import MongoClient
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# --- MongoDB Setup ---
client = MongoClient("mongodb://localhost:27017/")
db = client["epigenomic_db"]
region_collection = db["regions"]
feature_collection = db["features"]
model_log_collection = db["model_logs"]

# --- UI ---
st.set_page_config(page_title="Epigenomic FFNN", layout="wide")
st.title("🧬 Epigenomic FFNN Training WebApp (MongoDB Version)")

# --- Section: MongoDB Query Interface ---
st.header("🔍 Explore MongoDB")

# Allow selection between collections
collection_options = {
    "Regions": region_collection,
    "Training Logs": db["model_logs"]
}
selected_collection_name = st.selectbox("Choose Collection to Query", list(collection_options.keys()))
selected_collection = collection_options[selected_collection_name]

st.markdown(f"Total documents: `{selected_collection.estimated_document_count():,}`")

# Default query depending on collection
default_query = (
    '{ "cell_line": "H1", "TPM": { "$gt": 0 } }'
    if selected_collection_name == "Regions"
    else '{}'
)

user_query_str = st.text_area("Enter MongoDB query (JSON format)", value=default_query, height=150)

if st.button("Run MongoDB Query"):
    try:
        import ast
        user_query = ast.literal_eval(user_query_str)
        result_docs = list(selected_collection.find(user_query).limit(50))
        if result_docs:
            result_df = pd.DataFrame(result_docs).drop(columns=["_id"], errors='ignore')
            st.dataframe(result_df)
        else:
            st.info("No documents matched the query.")
    except Exception as e:
        st.error(f"Error parsing or executing query: {e}")

# --- Fetch dynamic dropdown options ---
def get_all_cell_lines():
    return sorted(region_collection.distinct("cell_line"))

def get_all_features(selected_cell_lines=None, sample_limit=1000):
    query = {}
    if selected_cell_lines:
        query["cell_line"] = {"$in": selected_cell_lines}
    cursor = region_collection.find(query, {"features": 1}).limit(sample_limit)
    all_keys = set()
    for doc in cursor:
        if "features" in doc:
            all_keys.update(doc["features"].keys())
    return sorted(all_keys)

# --- Data Loader ---
def load_filtered_data(cell_lines, features, region_types, tpm_values, sample_limit=50000):
    query = {
        "cell_line": {"$in": cell_lines},
        "region_type": {"$in": region_types},
        "TPM": {"$in": tpm_values},
    }
    projection = {f"features.{f}": 1 for f in features}
    projection.update({"TPM": 1})
    cursor = region_collection.find(query, projection).limit(sample_limit)
    X, y = [], []
    for doc in cursor:
        row = [doc['features'].get(f, 0) for f in features]
        X.append(row)
        y.append(int(doc['TPM'] > 0))
    return np.array(X), np.array(y)

# --- Model Trainer ---
def build_model(input_dim):
    return Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

def train_ffnn_streaming(X, y, batch_size=5000):
    model = build_model(X.shape[1])
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    history_all = {'loss': [], 'accuracy': []}
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        history = model.fit(X_batch, y_batch, epochs=10, verbose=0)
        history_all['loss'].extend(history.history['loss'])
        history_all['accuracy'].extend(history.history['accuracy'])
    return model, history_all

# --- UI Controls ---
st.header("🤖 Train FFNN Model")
region_choices = {
    "IE vs IP": (["enhancer", "promoter"], [0.0]),
    "AP vs IP": (["promoter"], [1.0, 0.0]),
    "AE vs IE": (["enhancer"], [1.0, 0.0]),
    "AE vs AP": (["enhancer", "promoter"], [1.0])
}
selected_task = st.selectbox("Select Classification Task", list(region_choices.keys()))
region_types, tpm_values = region_choices[selected_task]

cell_lines = get_all_cell_lines()
selected_cell_lines = st.multiselect("Select Cell Lines", cell_lines)

features = get_all_features(selected_cell_lines)
selected_features = st.multiselect("Select Features (leave empty to use all)", features)

sample_limit = st.number_input("Total documents to fetch (dataset size)", value=50000, step=1000)
batch_size = st.number_input("Batch size for training", value=5000, step=1000)

if st.button("Start Training from MongoDB"):
    if not selected_cell_lines:
        st.warning("Please select at least one cell line.")
    else:
        if not selected_features:
            selected_features = features  # Use all if none selected
            used_all_features = True
        else:
            used_all_features = False

        with st.spinner("⏳ Loading data and training model..."):
            start_data = time.time()
            X, y = load_filtered_data(selected_cell_lines, selected_features, region_types, tpm_values, sample_limit)
            data_load_time = time.time() - start_data

            start_train = time.time()
            model, history = train_ffnn_streaming(X, y, batch_size=batch_size)
            train_time = time.time() - start_train

            y_pred = (model.predict(X) > 0.5).astype(int)
            accuracy = accuracy_score(y, y_pred)
            st.metric(label="🔍 Model Accuracy", value=f"{accuracy:.4f}")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
                joblib.dump(model, tmp.name)
                st.download_button(
                    label="📥 Download Trained Model",
                    data=open(tmp.name, "rb").read(),
                    file_name="trained_ffnn_model.pkl",
                    mime="application/octet-stream"
                )

            # Save training metadata
            model_log = {
                "timestamp": datetime.utcnow(),
                "task": selected_task,
                "cell_lines": selected_cell_lines,
                "features_used": selected_features,
                "used_all_features": used_all_features,
                "sample_limit": sample_limit,
                "batch_size": batch_size,
                "data_load_time_sec": round(data_load_time, 2),
                "train_time_sec": round(train_time, 2),
                "accuracy": round(float(accuracy), 4)
            }
            model_log_collection.insert_one(model_log)

        st.success("✅ Model trained!")
        st.line_chart(pd.DataFrame(history['loss'], columns=["Loss"]))

# --- View Past Model Logs ---
st.header("📜 Past Training Logs")
if st.button("Show Recent Logs"):
    logs = list(model_log_collection.find().sort("timestamp", -1).limit(20))
    if logs:
        log_df = pd.DataFrame(logs).drop(columns=["_id"])
        st.dataframe(log_df)
    else:
        st.info("No logs found.")

st.caption("📘 Marco William Langi · 23523033 · Thesis 2025")
