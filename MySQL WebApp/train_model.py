import tensorflow as tf

def build_model(input_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

def train_ffnn_streaming(X, y, batch_size=5000):
    model = build_model(X.shape[1])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history_all = {'loss': [], 'accuracy': []}
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        hist = model.fit(X_batch, y_batch, epochs=10, verbose=0)
        history_all['loss'].extend(hist.history['loss'])
        history_all['accuracy'].extend(hist.history['accuracy'])

    return model, history_all