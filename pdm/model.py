"""CPU-friendly fusion-specific temporal-attention reconstruction model."""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import platform
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .losses import clinical_pattern_loss

_LAST_SEED = None

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except RuntimeError:
    # TensorFlow may already be initialized by an embedding application.
    pass


@tf.keras.utils.register_keras_serializable(package="pdm")
class TemporalAttention(layers.Layer):
    """Learned shared temporal attention over the final encoder sequence."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score = layers.Dense(1, name="attention_score")

    def build(self, input_shape):
        self.score.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        logits = self.score(inputs)
        weights = tf.nn.softmax(logits, axis=1)
        context = tf.reduce_sum(inputs * weights, axis=1)
        return context, tf.squeeze(weights, axis=-1)


@tf.keras.utils.register_keras_serializable(package="pdm")
class RuntimeMetadata(layers.Layer):
    """Identity layer whose config embeds runtime provenance in checkpoints."""

    def __init__(self, metadata: dict, **kwargs):
        super().__init__(**kwargs)
        self.metadata = dict(metadata)

    def call(self, inputs):
        return inputs

    def get_config(self):
        config = super().get_config()
        config["metadata"] = self.metadata
        return config


def set_global_seed(seed: int) -> dict:
    """Seed Python, NumPy, and TensorFlow for a training entry point.

    TensorFlow operation ordering can still differ between builds and hardware,
    so this helper reports (rather than promises) deterministic execution.
    """

    global _LAST_SEED

    seed = int(seed)
    _LAST_SEED = seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    return get_runtime_metadata(seed)


def get_runtime_metadata(seed: int | None = None) -> dict:
    """Return versions and reproducibility caveat to persist with a checkpoint."""

    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
    }
    return {
        "seed": None if seed is None else int(seed),
        "python_version": versions["python"],
        "numpy_version": versions["numpy"],
        "tensorflow_version": versions["tensorflow"],
        "keras_version": versions["keras"],
        "library_versions": versions,
        "bitwise_reproducibility": False,
        "reproducibility_note": "Exact bitwise reproducibility is not guaranteed across TensorFlow builds or hardware.",
    }


class AttentionAutoencoder:
    """Fixed-contract two-stage BiLSTM TAAE with shared temporal attention.

    The Phase 2 widths are intentionally enforced: encoder units 32 then 16,
    latent dimension 16, and decoder units 16 then 32. Training controls such
    as epochs, batch size, and patience remain configurable for smoke tests and
    resource-constrained runs without changing the detector architecture.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_units: int = 32,
        epochs: int = 20,
        batch_size: int = 64,
        early_stopping_patience: int = 3,
    ):
        if int(latent_dim) != 16 or int(hidden_units) != 32:
            raise ValueError(
                "Phase 2 architecture is fixed at latent_dim=16 and hidden_units=32"
            )
        self.latent_dim = 16
        self.hidden_units = 32
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.early_stopping_patience = int(early_stopping_patience)
        self.network = None
        self.attention_model = None
        self.threshold = None
        self.history = None
        self.input_shape = None
        self.runtime_metadata = get_runtime_metadata(_LAST_SEED)

    def _build(self, shape):
        """Build the training model and its reconstruction/attention inference view."""

        shape = tuple(int(value) for value in shape)
        if len(shape) != 2 or shape[0] < 1 or shape[1] < 1:
            raise ValueError("shape must be (timesteps, channels)")
        self.input_shape = shape
        self.runtime_metadata = get_runtime_metadata(_LAST_SEED)

        inp = keras.Input(shape=shape, name="signals")
        enc = layers.Bidirectional(
            layers.LSTM(self.hidden_units, return_sequences=True),
            name="encoder_bilstm_32",
        )(inp)
        enc = layers.Bidirectional(
            layers.LSTM(16, return_sequences=True),
            name="encoder_bilstm_16",
        )(enc)
        context, attention = TemporalAttention(name="temporal_attention")(enc)
        latent = layers.Dense(self.latent_dim, activation="tanh", name="latent_projection")(context)
        latent = layers.LayerNormalization(name="latent_normalization")(latent)
        repeated = layers.RepeatVector(shape[0], name="latent_repeat")(latent)
        dec = layers.LSTM(16, return_sequences=True, name="decoder_lstm_16")(repeated)
        dec = layers.LSTM(32, return_sequences=True, name="decoder_lstm_32")(dec)
        reconstruction = layers.TimeDistributed(
            layers.Dense(shape[1], activation="linear"), name="reconstruction"
        )(dec)
        reconstruction = RuntimeMetadata(
            self.runtime_metadata, name="runtime_metadata"
        )(reconstruction)

        self.network = keras.Model(
            inp, reconstruction, name="bilstm_temporal_attention_autoencoder"
        )
        self.network.compile(
            optimizer=keras.optimizers.Adam(1e-3), loss=clinical_pattern_loss
        )
        self.attention_model = keras.Model(
            inp, [reconstruction, attention], name="attention_inference"
        )
        return self.network

    def fit(self, x, percentile=90, validation_split=0.1):
        x = np.asarray(x, dtype="float32")
        if x.ndim != 3 or x.shape[0] == 0:
            raise ValueError("x must have shape (samples, timesteps, channels)")
        self._build(tuple(x.shape[1:]))
        validation_split = float(validation_split)
        has_validation = 0 < validation_split < 1 and x.shape[0] > 1
        monitor = "val_loss" if has_validation else "loss"
        callback = keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=self.early_stopping_patience,
            restore_best_weights=True,
        )
        self.history = self.network.fit(
            x,
            x,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split if has_validation else 0,
            shuffle=True,
            callbacks=[callback],
            verbose=0,
        )
        self.runtime_metadata = get_runtime_metadata(_LAST_SEED)
        self.network.get_layer("runtime_metadata").metadata = dict(
            self.runtime_metadata
        )
        errors, _, _ = self.predict(x)
        self.threshold = float(np.percentile(errors, percentile))
        return self

    def predict(self, x):
        if self.attention_model is None:
            raise RuntimeError("build or fit the model before predicting")
        x = np.asarray(x, dtype="float32")
        reconstruction, attention = self.attention_model.predict(
            x, batch_size=self.batch_size, verbose=0
        )
        errors = np.mean((x - reconstruction) ** 2, axis=(1, 2))
        return errors, reconstruction, attention

    def results(self, x):
        if self.threshold is None:
            raise ValueError("anomaly threshold is not calibrated")
        threshold = float(self.threshold)
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("anomaly threshold must be a finite positive number")
        errors, reconstruction, attention = self.predict(x)
        anomaly_margin = errors - threshold
        normalized_margin = anomaly_margin / threshold
        is_anomaly = errors > threshold
        decisions = np.where(is_anomaly, "anomaly", "normal").tolist()
        return {
            "error": errors,
            "reconstruction_error": errors,
            "reconstruction": reconstruction,
            "attention": attention,
            "anomaly_threshold": threshold,
            "threshold": threshold,
            "anomaly_margin": anomaly_margin,
            "normalized_margin": normalized_margin,
            "is_anomaly": is_anomaly,
            "decision": decisions,
            "model_decision": decisions,
            "predicted_class": decisions,
        }

    def save(self, path):
        """Save a reloadable Keras checkpoint without caller custom objects."""

        if self.network is None:
            raise RuntimeError("build or fit the model before saving")
        # Optimizer slots are not needed for inference reloads and can be
        # absent on an untrained checkpoint. Save an uncompiled view so a
        # loader does not try to restore a partial optimizer state.
        checkpoint = keras.Model(
            self.network.inputs, self.network.outputs, name=self.network.name
        )
        checkpoint.get_layer("runtime_metadata").metadata = dict(
            self.runtime_metadata
        )
        checkpoint.set_weights(self.network.get_weights())
        checkpoint.save(path, include_optimizer=False)
