"""
06_dl_models.py -- Deep learning models for J1 (improved over C2).

CLAUDE.md Section 4.3:
  MODEL 6: ANN with residual connections + Huber loss + cosine-annealed Adam
  MODEL 7: Three-stream CNN-LSTM with multi-head self-attention (main DL novelty)
  MODEL 8: Transformer-based regressor (custom TransformerEncoder layer)
"""
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config

tf.random.set_seed(config.TF_SEED)
np.random.seed(config.RANDOM_STATE)

# By default TF grabs ~all free GPU memory on first use. On this project's
# 4 GB card that starves XGBoost/CatBoost (also GPU, config.USE_GPU) which
# run in the SAME process later in the pipeline -- enable memory growth so
# TF only allocates what it actually needs.
for _gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass  # must be set before GPUs are initialized; harmless if already done


# ---------------------------------------------------------------------------
# MODEL 6: ANN with residual connection
# ---------------------------------------------------------------------------
def build_ann_residual(n_features):
    inputs = layers.Input(shape=(n_features,), name="features")

    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    block1 = layers.Dense(128, activation='relu')(x)
    block1 = layers.BatchNormalization()(block1)
    block1 = layers.Dropout(0.25)(block1)

    input_proj = layers.Dense(128)(inputs)
    res = layers.Add()([input_proj, block1])

    x2 = layers.Dense(64, activation='relu')(res)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(0.2)(x2)

    x3 = layers.Dense(32, activation='relu')(x2)
    x3 = layers.Dropout(0.1)(x3)

    outputs = layers.Dense(1, activation='linear')(x3)
    model = keras.Model(inputs, outputs, name="ANN_Residual")
    return model


# ---------------------------------------------------------------------------
# MODEL 7: Three-stream CNN-LSTM with multi-head attention
# ---------------------------------------------------------------------------
def _cnn_lstm_stream(n_group_features, name):
    stream_in = layers.Input(shape=(n_group_features, 1), name=f"{name}_input")
    kernel = min(3, n_group_features)
    x = layers.Conv1D(64, kernel_size=kernel, activation='relu', padding='same')(stream_in)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
    x = layers.LSTM(64)(x)
    out = layers.Dense(32, activation='relu')(x)
    return stream_in, out


def build_cnn_lstm_attention(group_sizes):
    """group_sizes: dict {'RUSLE': n, 'Topographic': n, 'Spectral_Climate_Soil': n}."""
    inputs, outs = [], []
    for name, n in group_sizes.items():
        stream_in, out = _cnn_lstm_stream(n, name)
        inputs.append(stream_in)
        outs.append(out)

    stacked = layers.Lambda(lambda t: tf.stack(t, axis=1),
                             output_shape=(len(outs), 32))(outs)  # (batch, 3, 32)
    attn_out = layers.MultiHeadAttention(num_heads=4, key_dim=8)(stacked, stacked)
    flat = layers.Flatten()(attn_out)

    x = layers.Dense(64, activation='relu')(flat)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='linear')(x)

    model = keras.Model(inputs, outputs, name="CNN_LSTM_Attention")
    return model


# ---------------------------------------------------------------------------
# MODEL 8: Transformer-based regressor
# ---------------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable(package="j1")
class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim=64, num_heads=4, ff_dim=128, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim, self.num_heads, self.ff_dim, self.rate = embed_dim, num_heads, ff_dim, rate
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(rate)
        self.drop2 = layers.Dropout(rate)

    def call(self, x, training=False):
        attn_out = self.att(x, x)
        attn_out = self.drop1(attn_out, training=training)
        x1 = self.norm1(x + attn_out)
        ffn_out = self.ffn(x1)
        ffn_out = self.drop2(ffn_out, training=training)
        return self.norm2(x1 + ffn_out)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'embed_dim': self.embed_dim, 'num_heads': self.num_heads,
                    'ff_dim': self.ff_dim, 'rate': self.rate})
        return cfg


def build_transformer_regressor(n_features, num_layers=2, embed_dim=64):
    inputs = layers.Input(shape=(n_features,), name="features")
    x = layers.Reshape((n_features, 1))(inputs)
    x = layers.Dense(embed_dim)(x)  # feature embedding per "token"
    for _ in range(num_layers):
        x = TransformerEncoderBlock(embed_dim=embed_dim, num_heads=4, ff_dim=128)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='linear')(x)
    model = keras.Model(inputs, outputs, name="Transformer_Regressor")
    return model


# ---------------------------------------------------------------------------
# Training orchestration
# ---------------------------------------------------------------------------
def _split_by_group(X, features, group_sizes_features):
    """Returns list of arrays [n_samples, n_group_features, 1] per group, in
    the same order as group_sizes_features (dict name -> list of feature idx)."""
    return [X[:, idx][:, :, None] for idx in group_sizes_features.values()]


def train_all(features, X_train_mm, y_train_dl, X_val_mm, y_val, X_test_mm, y_test,
              epochs=None, batch_size=None, quick=False):
    epochs = epochs or config.EPOCHS
    batch_size = batch_size or config.BATCH_SIZE
    n_features = X_train_mm.shape[1]
    results = {}

    group_idx = {name: [i for i, f in enumerate(features) if f in flist]
                 for name, flist in config.FEATURE_GROUPS.items()}
    # Any feature not captured by the 3 named groups (e.g. engineered features)
    # falls into Spectral_Climate_Soil as a catch-all so every column is used.
    assigned = set(sum(group_idx.values(), []))
    leftover = [i for i in range(n_features) if i not in assigned]
    group_idx['Spectral_Climate_Soil'] = group_idx.get('Spectral_Climate_Soil', []) + leftover
    group_idx = {k: v for k, v in group_idx.items() if len(v) > 0}
    group_sizes = {k: len(v) for k, v in group_idx.items()}

    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=config.PATIENCE_ES,
                                        restore_best_weights=True)

    # -- MODEL 6: ANN-Residual -------------------------------------------------
    t0 = time.time()
    steps_per_epoch = max(1, len(X_train_mm) // batch_size)
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        config.LR_ANN, decay_steps=steps_per_epoch * epochs)
    ann = build_ann_residual(n_features)
    ann.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                loss=keras.losses.Huber())
    hist_ann = ann.fit(X_train_mm, y_train_dl, validation_data=(X_val_mm, y_val),
                        epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=0)
    y_pred_ann = ann.predict(X_test_mm, verbose=0).flatten()
    results['ANN-Residual'] = {'model': ann, 'y_pred': y_pred_ann,
                                'train_time': time.time() - t0, 'history': hist_ann.history}
    print(f"  ANN-Residual trained in {results['ANN-Residual']['train_time']:.1f}s "
          f"({len(hist_ann.history['loss'])} epochs)")

    # -- MODEL 7: CNN-LSTM-Attention -------------------------------------------
    t0 = time.time()
    X_train_groups = _split_by_group(X_train_mm, features, group_idx)
    X_val_groups = _split_by_group(X_val_mm, features, group_idx)
    X_test_groups = _split_by_group(X_test_mm, features, group_idx)

    cnn_lstm = build_cnn_lstm_attention(group_sizes)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=config.PATIENCE_LR,
                                                    min_lr=config.MIN_LR)
    cnn_lstm.compile(optimizer=keras.optimizers.Adam(learning_rate=config.LR_CNN_LSTM),
                      loss=keras.losses.Huber())
    hist_cnn = cnn_lstm.fit(X_train_groups, y_train_dl, validation_data=(X_val_groups, y_val),
                             epochs=epochs, batch_size=batch_size,
                             callbacks=[es, reduce_lr], verbose=0)
    y_pred_cnn = cnn_lstm.predict(X_test_groups, verbose=0).flatten()
    results['CNN-LSTM-Attention'] = {'model': cnn_lstm, 'y_pred': y_pred_cnn,
                                      'train_time': time.time() - t0, 'history': hist_cnn.history,
                                      'group_sizes': group_sizes}
    print(f"  CNN-LSTM-Attention trained in {results['CNN-LSTM-Attention']['train_time']:.1f}s "
          f"({len(hist_cnn.history['loss'])} epochs)")

    # -- MODEL 8: Transformer ---------------------------------------------------
    t0 = time.time()
    transformer = build_transformer_regressor(n_features)
    reduce_lr2 = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=config.PATIENCE_LR,
                                                     min_lr=config.MIN_LR)
    transformer.compile(optimizer=keras.optimizers.Adam(learning_rate=config.LR_TRANSFORMER),
                         loss=keras.losses.Huber())
    hist_tr = transformer.fit(X_train_mm, y_train_dl, validation_data=(X_val_mm, y_val),
                               epochs=epochs, batch_size=batch_size,
                               callbacks=[es, reduce_lr2], verbose=0)
    y_pred_tr = transformer.predict(X_test_mm, verbose=0).flatten()
    results['Transformer'] = {'model': transformer, 'y_pred': y_pred_tr,
                               'train_time': time.time() - t0, 'history': hist_tr.history}
    print(f"  Transformer trained in {results['Transformer']['train_time']:.1f}s "
          f"({len(hist_tr.history['loss'])} epochs)")

    return results, group_idx


def save_models(results):
    for name, key in [('ANN-Residual', 'ann_residual.keras'),
                       ('CNN-LSTM-Attention', 'cnn_lstm_attention.keras'),
                       ('Transformer', 'transformer.keras')]:
        path = os.path.join(config.MODELS_DIR, key)
        results[name]['model'].save(path)
        print(f"  Saved: {path}")


if __name__ == "__main__":
    pass
