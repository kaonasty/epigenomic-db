def prepare_X_y(df, source='db'):
    if source == 'csv':
        feature_cols = df.columns.difference(['chrom', 'start', 'end', 'strand', 'TPM'])
        X = df[feature_cols].values
        y = df['TPM'].values
        y = (y > 0).astype(int)
    else:
        X = df['value'].values.reshape(-1, 1)
        y = df['TPM'].values
        y = (y > 0).astype(int)
    return X, y
