import pandas as pd
from sqlalchemy import create_engine, text
import urllib.parse

# --- Configuration ---
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": urllib.parse.quote_plus("B@ndung40175"),  # Encode special characters like @
    "database": "epigenomicdb"
}

# --- SQLAlchemy Engine ---
def get_sqlalchemy_engine():
    return create_engine(
        f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
    )

# --- Helper to build WHERE clauses with parameterized IN ---
def build_in_clause(column, values, param_prefix, params):
    placeholders = []
    for i, val in enumerate(values):
        param_name = f"{param_prefix}_{i}"
        placeholders.append(f":{param_name}")
        params[param_name] = val
    return f"{column} IN ({', '.join(placeholders)})"

# --- Fetch filtered region-feature data ---
def get_filtered_data(cell_lines=None, region_type=None, TPM=None, feature_names=None, limit=10000):
    engine = get_sqlalchemy_engine()
    conn = engine.connect()

    where_clauses = []
    params = {}

    if cell_lines:
        where_clauses.append(build_in_clause("r.cell_line", cell_lines, "cell_line", params))
    if region_type:
        where_clauses.append(build_in_clause("r.region_type", region_type, "region_type", params))
    if TPM is not None:
        where_clauses.append("r.TPM >= :TPM")
        params["TPM"] = TPM
    if feature_names:
        where_clauses.append(build_in_clause("f.feature_name", feature_names, "feature", params))

    where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    query = f"""
        SELECT r.region_id, r.region_type, r.TPM, r.cell_line, f.feature_name, rf.value
        FROM region r
        JOIN region_feature rf ON r.region_id = rf.region_id
        JOIN feature f ON rf.feature_id = f.feature_id
        {where_clause}
        LIMIT {limit}
    """

    df = pd.read_sql(text(query), conn, params=params)
    conn.close()
    return df


# --- Count how many rows match current filters ---
def count_matching_rows(cell_lines=None, region_type=None, TPM=None, feature_names=None):
    engine = get_sqlalchemy_engine()
    conn = engine.connect()

    where_clauses = []
    params = {}

    if cell_lines:
        where_clauses.append(build_in_clause("r.cell_line", cell_lines, "cell_line", params))
    if region_type:
        where_clauses.append(build_in_clause("r.region_type", region_type, "region_type", params))
    if TPM:
        where_clauses.append(build_in_clause("r.TPM", TPM, "TPM", params))
    if feature_names:
        where_clauses.append(build_in_clause("f.feature_name", feature_names, "feature", params))

    where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    query = f"""
        SELECT COUNT(*) as count
        FROM region r
        JOIN region_feature rf ON r.region_id = rf.region_id
        JOIN feature f ON rf.feature_id = f.feature_id
        {where_clause}
    """

    df = pd.read_sql(text(query), conn, params=params)
    conn.close()
    return int(df["count"].iloc[0])

# --- Get all unique cell lines ---
def get_all_cell_lines():
    engine = get_sqlalchemy_engine()
    query = "SELECT DISTINCT cell_line FROM region"
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df['cell_line'].tolist()

# --- Get all unique region types ---
def get_region_types():
    engine = get_sqlalchemy_engine()
    query = "SELECT DISTINCT region_type FROM region"
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df['region_type'].tolist()

# --- Get all TPM values ---
def get_tpm_values():
    engine = get_sqlalchemy_engine()
    query = "SELECT DISTINCT TPM FROM region"
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return sorted(df['TPM'].unique().tolist())

# --- Get all features ---
def get_all_features():
    engine = get_sqlalchemy_engine()
    query = "SELECT DISTINCT feature_name FROM feature"
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df['feature_name'].tolist()
