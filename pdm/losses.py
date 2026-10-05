"""Differentiable reconstruction losses used by the fusion detector.

The detector uses a practical, differentiable approximation of the paper's
Clinical Pattern Loss (CPLoss). The approximation compares local temporal
patterns, window-level trends, and point-wise reconstruction fidelity. It is
implemented with TensorFlow operations so it remains usable in a
``GradientTape`` and in a serialized Keras model.
"""

from __future__ import annotations

import tensorflow as tf


def _float_pair(y_true, y_pred):
    """Convert loss inputs to a common floating dtype without breaking gradients."""

    true = tf.convert_to_tensor(y_true)
    pred = tf.convert_to_tensor(y_pred)
    dtype = tf.float32
    if true.dtype.is_floating and pred.dtype.is_floating:
        dtype = true.dtype if true.dtype == pred.dtype else tf.float32
    return tf.cast(true, dtype), tf.cast(pred, dtype)


def _safe_mean(values):
    values = tf.convert_to_tensor(values)
    return tf.math.divide_no_nan(
        tf.reduce_sum(values), tf.cast(tf.size(values), values.dtype)
    )


@tf.keras.utils.register_keras_serializable(package="pdm")
def pattern_loss(y_true, y_pred):
    """Mean absolute difference between first temporal differences."""

    true, pred = _float_pair(y_true, y_pred)
    true_diff = true[:, 1:, ...] - true[:, :-1, ...]
    pred_diff = pred[:, 1:, ...] - pred[:, :-1, ...]
    return _safe_mean(tf.abs(true_diff - pred_diff))


@tf.keras.utils.register_keras_serializable(package="pdm")
def trend_loss(y_true, y_pred):
    """Mean absolute difference between per-window/channel least-squares slopes."""

    true, pred = _float_pair(y_true, y_pred)
    length = tf.shape(true)[1]
    time = tf.cast(tf.range(length), true.dtype)
    centered_time = time - tf.reduce_mean(time)
    denominator = tf.reduce_sum(tf.square(centered_time))
    weights = centered_time[tf.newaxis, :, tf.newaxis]
    true_slope = tf.math.divide_no_nan(
        tf.reduce_sum(true * weights, axis=1), denominator
    )
    pred_slope = tf.math.divide_no_nan(
        tf.reduce_sum(pred * weights, axis=1), denominator
    )
    return _safe_mean(tf.abs(true_slope - pred_slope))


@tf.keras.utils.register_keras_serializable(package="pdm")
def fidelity_loss(y_true, y_pred):
    """Point-wise mean squared reconstruction error."""

    true, pred = _float_pair(y_true, y_pred)
    return _safe_mean(tf.square(true - pred))


@tf.keras.utils.register_keras_serializable(package="pdm")
def clinical_pattern_loss(y_true, y_pred):
    """Practical CPLoss approximation: 0.3 pattern + 0.4 trend + 0.3 fidelity."""

    return (
        0.3 * pattern_loss(y_true, y_pred)
        + 0.4 * trend_loss(y_true, y_pred)
        + 0.3 * fidelity_loss(y_true, y_pred)
    )
