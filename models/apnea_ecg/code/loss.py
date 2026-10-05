"""
Loss Function Implementation for  MedAttnAID.

Extracted from the notebook section:
"Clinical Pattern Loss implementation and ablation losses".

Implementation note from the notebook:
The provided pseudo-algorithm defines PatternLoss using sign mismatches and
TrendLoss using sliding-window slopes. In this TensorFlow notebook, those
components are implemented as differentiable approximations of the same ideas
so they can be used safely during backpropagation.
"""

import tensorflow as tf
from tensorflow import keras


# Minimal CONFIG values needed by the loss functions.
# These values come from the notebook's global CONFIG.
CONFIG = {
    # 100-step morphology-bin representation.
    # Channels: mean, std, min, max, range, slope, energy.
    "morphology_feature_names": [
        "mean", "std", "min", "max", "range", "slope", "energy"
    ],
    "morphology_feature_weights": [
        0.10, 0.25, 0.10, 0.10, 0.25, 0.05, 0.15
    ],

    # Required PatternLoss ablation modes from the notebook.
    "medattnaid_loss_modes_to_run": [
        "mse_only",
        "mse_plus_pattern",
        "mse_plus_trend",
        "cploss_full",
    ],

    # Final official MedAttnAID loss mode used in the notebook.
    "official_medattnaid_loss_mode": "mse_plus_trend",
}


def channel_weights_for_tensor(y_true):
    """
    Return per-channel weights for the current model input.
    For morphology-bin input, emphasize std/range/energy because they preserve ECG morphology.
    """
    n_channels = y_true.shape[-1]

    if n_channels == 7:
        w = tf.constant(CONFIG["morphology_feature_weights"], dtype=tf.float32)
    else:
        w = tf.ones((int(n_channels),), dtype=tf.float32)

    return w / (tf.reduce_sum(w) + 1e-8)


def mse_component(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    se = tf.square(y_true - y_pred)
    per_channel = tf.reduce_mean(se, axis=1)  # (batch, channels)
    w = channel_weights_for_tensor(y_true)

    return tf.reduce_sum(per_channel * tf.reshape(w, (1, -1)), axis=1)


def pattern_component(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    dy_true = y_true[:, 1:, :] - y_true[:, :-1, :]
    dy_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]

    per_channel = tf.reduce_mean(tf.abs(dy_true - dy_pred), axis=1)
    w = channel_weights_for_tensor(y_true)

    return tf.reduce_sum(per_channel * tf.reshape(w, (1, -1)), axis=1)


def trend_component(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    d2_true = y_true[:, 2:, :] - 2.0 * y_true[:, 1:-1, :] + y_true[:, :-2, :]
    d2_pred = y_pred[:, 2:, :] - 2.0 * y_pred[:, 1:-1, :] + y_pred[:, :-2, :]

    per_channel = tf.reduce_mean(tf.abs(d2_true - d2_pred), axis=1)
    w = channel_weights_for_tensor(y_true)

    return tf.reduce_sum(per_channel * tf.reshape(w, (1, -1)), axis=1)


class ClinicalPatternLoss(keras.losses.Loss):
    def __init__(self, mode="cploss_full", name=None):
        super().__init__(name=name or mode)
        self.mode = mode

    def call(self, y_true, y_pred):
        fid = mse_component(y_true, y_pred)
        pat = pattern_component(y_true, y_pred)
        trd = trend_component(y_true, y_pred)

        # Normalize by detached batch means to keep components comparable but non-saturating.
        fid_n = fid / (tf.stop_gradient(tf.reduce_mean(fid)) + 1e-6)
        pat_n = pat / (tf.stop_gradient(tf.reduce_mean(pat)) + 1e-6)
        trd_n = trd / (tf.stop_gradient(tf.reduce_mean(trd)) + 1e-6)

        if self.mode == "mse_only":
            return fid
        if self.mode == "mse_plus_pattern":
            return 0.5 * fid_n + 0.5 * pat_n
        if self.mode == "mse_plus_trend":
            return 0.5 * fid_n + 0.5 * trd_n
        if self.mode == "cploss_full":
            return 0.3 * fid_n + 0.3 * pat_n + 0.4 * trd_n
        raise ValueError(f"Unknown loss mode: {self.mode}")


LOSS_DISPLAY_NAMES = {
    "mse_only": "MSE only",
    "mse_plus_pattern": "MSE + Pattern",
    "mse_plus_trend": "MSE + Trend",
    "cploss_full": "CPLoss (Full)",
}


def make_medattnaid_loss(loss_mode):
    return ClinicalPatternLoss(mode=loss_mode, name=loss_mode)



