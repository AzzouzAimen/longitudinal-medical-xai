"""
Model implementation for  MedAttnAID.

Adapted from the training notebook section:
"MedAttnAID model implementation: Temporal Attention Autoencoder".

This file is focused only on the model implementation and covers:
- TemporalAttention layer
- LearnedPositionEmbedding layer
- TimestepMask layer
- Global-context Temporal Attention Autoencoder
- Final sequence-masked MedAttnAID autoencoder
- build_medattnaid_model factory that selects the architecture from CONFIG

"""

import os

# Same quiet TensorFlow flags used in the notebook before importing TensorFlow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["AUTOGRAPH_VERBOSITY"] = "0"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Minimal model-related CONFIG values taken from the notebook.
# Other training/evaluation/loss settings belong in the other modules.
CONFIG = {
    "taae_steps": 100,
    "latent_dim": 16,
    "attention_dim": 64,
    "encoder_units": 64,
    "decoder_units": 64,

    # Final selected MedAttnAID architecture in the notebook:
    # sequence bottleneck + true timestep masking.
    "medattnaid_architecture": "sequence_masked",
    "timestep_mask_rate": 0.25,

    # Kept because the notebook model factory checks this older key as fallback.
    "enhanced_timestep_mask_rate": 0.0,
}

class TemporalAttention(layers.Layer):
    def __init__(self, attention_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.attention_dim = int(attention_dim)
        self.score_dense = layers.Dense(self.attention_dim, activation="tanh")
        self.score_out = layers.Dense(1)

    def call(self, h, return_weights=False):
        # h: (batch, L, hidden)
        score = self.score_out(self.score_dense(h))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * h, axis=1)
        if return_weights:
            return context, weights
        return context

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"attention_dim": self.attention_dim})
        return cfg

class LearnedPositionEmbedding(layers.Layer):
    def __init__(self, length, dim=16, **kwargs):
        super().__init__(**kwargs)
        self.length = int(length)
        self.dim = int(dim)
        self.embedding = layers.Embedding(self.length, self.dim)

    def call(self, x):
        batch = tf.shape(x)[0]
        positions = tf.range(self.length)
        pos = self.embedding(positions)
        pos = tf.expand_dims(pos, axis=0)
        return tf.tile(pos, [batch, 1, 1])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"length": self.length, "dim": self.dim})
        return cfg


class TimestepMask(layers.Layer):
    """
    True timestep masking for sequence-bottleneck denoising.

    Unlike SpatialDropout1D, this layer drops whole timesteps across all channels.
    It is active only during training, so evaluation uses the unmasked input.
    """
    def __init__(self, rate=0.25, **kwargs):
        super().__init__(**kwargs)
        self.rate = float(rate)
        self.dropout = None

    def build(self, input_shape):
        # noise_shape=(batch, timesteps, 1) means the same mask is shared
        # across all channels, so selected timesteps are fully dropped.
        self.dropout = layers.Dropout(
            self.rate,
            noise_shape=(None, input_shape[1], 1),
            name="timestep_dropout",
        )
        super().build(input_shape)

    def call(self, x, training=None):
        if self.rate <= 0:
            return x
        return self.dropout(x, training=training)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"rate": self.rate})
        return cfg

def build_temporal_attention_autoencoder(input_shape, latent_dim=None, attention_dim=None, name="MedAttnAID"):
    """
    Original global-context Temporal Attention Autoencoder.

    This is the morphology-bin model version that gave the stronger anomaly-loss
    separation. It uses attention context -> latent vector -> RepeatVector decoder.
    """
    L, C = int(input_shape[0]), int(input_shape[1])
    latent_dim = int(CONFIG.get("latent_dim", 32) if latent_dim is None else latent_dim)
    attention_dim = int(CONFIG.get("attention_dim", 64) if attention_dim is None else attention_dim)
    enc_units = int(CONFIG.get("encoder_units", 64))
    dec_units = int(CONFIG.get("decoder_units", 64))

    inputs = keras.Input(shape=(L, C), name="ecg_window")

    x = layers.Conv1D(
        32,
        kernel_size=7,
        padding="same",
        name="encoder_conv_1",
    )(inputs)
    x = layers.BatchNormalization(name="encoder_bn_1")(x)
    x = layers.Activation("swish", name="encoder_swish_1")(x)

    x = layers.Conv1D(
        64,
        kernel_size=5,
        padding="same",
        name="encoder_conv_2",
    )(x)
    x = layers.BatchNormalization(name="encoder_bn_2")(x)
    x = layers.Activation("swish", name="encoder_swish_2")(x)

    h = layers.Bidirectional(
        layers.GRU(enc_units, return_sequences=True, dropout=0.05),
        name="encoder_bigru",
    )(x)

    attn = TemporalAttention(
        attention_dim=attention_dim,
        name="temporal_attention",
    )
    context, attn_weights = attn(h, return_weights=True)

    latent = layers.Dense(
        latent_dim,
        activation="linear",
        name="latent_vector",
    )(context)
    latent = layers.LayerNormalization(name="latent_norm")(latent)

    repeated = layers.RepeatVector(
        L,
        name="repeat_latent",
    )(latent)

    pos = LearnedPositionEmbedding(
        L,
        dim=16,
        name="decoder_position_embedding",
    )(repeated)

    d = layers.Concatenate(name="decoder_latent_plus_position")(
        [repeated, pos]
    )

    d = layers.GRU(
        dec_units,
        return_sequences=True,
        dropout=0.05,
        name="decoder_gru_1",
    )(d)

    d = layers.Conv1D(
        64,
        kernel_size=5,
        padding="same",
        activation="swish",
        name="decoder_conv_1",
    )(d)

    d = layers.Conv1D(
        32,
        kernel_size=7,
        padding="same",
        activation="swish",
        name="decoder_conv_2",
    )(d)

    outputs = layers.Conv1D(
        C,
        kernel_size=1,
        padding="same",
        activation="sigmoid",
        dtype="float32",
        name="reconstruction",
    )(d)

    model = keras.Model(inputs, outputs, name=name)
    model.attention_model = keras.Model(inputs, attn_weights, name=f"{name}_attention")
    model.encoder_model = keras.Model(inputs, latent, name=f"{name}_encoder")
    return model


def build_sequence_bottleneck_autoencoder(input_shape, latent_dim=None, attention_dim=None, name="MedAttnAID_sequence_masked"):
    """
    Sequence-preserving MedAttnAID variant.

    This keeps a per-timestep latent sequence instead of using RepeatVector.
    When CONFIG["timestep_mask_rate"] > 0, true timestep masking is applied only
    during training, so the decoder must learn healthy temporal rules rather than
    copying every input timestep.
    """
    L, C = int(input_shape[0]), int(input_shape[1])
    latent_dim = int(CONFIG.get("latent_dim", 32) if latent_dim is None else latent_dim)
    attention_dim = int(CONFIG.get("attention_dim", 64) if attention_dim is None else attention_dim)
    enc_units = int(CONFIG.get("encoder_units", 64))
    dec_units = int(CONFIG.get("decoder_units", 64))
    mask_rate = float(CONFIG.get("timestep_mask_rate", CONFIG.get("enhanced_timestep_mask_rate", 0.0)))

    inputs = keras.Input(shape=(L, C), name="ecg_window")
    x_in = TimestepMask(rate=mask_rate, name="input_timestep_mask")(inputs) if mask_rate > 0 else inputs

    x = layers.Conv1D(
        32,
        kernel_size=7,
        padding="same",
        name="encoder_conv_1",
    )(x_in)
    x = layers.BatchNormalization(name="encoder_bn_1")(x)
    x = layers.Activation("swish", name="encoder_swish_1")(x)

    x = layers.Conv1D(
        64,
        kernel_size=5,
        padding="same",
        name="encoder_conv_2",
    )(x)
    x = layers.BatchNormalization(name="encoder_bn_2")(x)
    x = layers.Activation("swish", name="encoder_swish_2")(x)

    h = layers.Bidirectional(
        layers.GRU(enc_units, return_sequences=True, dropout=0.05),
        name="encoder_bigru",
    )(x)

    attn = TemporalAttention(
        attention_dim=attention_dim,
        name="temporal_attention",
    )
    context, attn_weights = attn(h, return_weights=True)

    latent_seq = layers.Dense(
        latent_dim,
        activation="linear",
        name="latent_sequence",
    )(h)
    latent_seq = layers.LayerNormalization(name="latent_sequence_norm")(latent_seq)

    latent_summary = layers.GlobalAveragePooling1D(name="latent_vector")(latent_seq)

    d = layers.GRU(
        dec_units,
        return_sequences=True,
        dropout=0.05,
        name="decoder_gru_1",
    )(latent_seq)

    d = layers.Conv1D(
        64,
        kernel_size=5,
        padding="same",
        activation="swish",
        name="decoder_conv_1",
    )(d)

    d = layers.Conv1D(
        32,
        kernel_size=7,
        padding="same",
        activation="swish",
        name="decoder_conv_2",
    )(d)

    outputs = layers.Conv1D(
        C,
        kernel_size=1,
        padding="same",
        activation="sigmoid",
        dtype="float32",
        name="reconstruction",
    )(d)

    model = keras.Model(inputs, outputs, name=name)
    model.attention_model = keras.Model(inputs, attn_weights, name=f"{name}_attention")
    model.encoder_model = keras.Model(inputs, latent_summary, name=f"{name}_encoder")
    model.latent_sequence_model = keras.Model(inputs, latent_seq, name=f"{name}_latent_sequence")
    return model

# Backward-compatible alias.

def build_medattnaid_model(input_shape, latent_dim=None, attention_dim=None, name="MedAttnAID"):
    """
    Backward-compatible model factory.

    Default CONFIG["medattnaid_architecture"] == "global" preserves the current
    working RepeatVector/global-context model. Enhanced experiments switch to
    "sequence_masked" inside temporary_config blocks.
    """
    architecture = str(CONFIG.get("medattnaid_architecture", "global")).lower()

    if architecture in {"global", "repeatvector", "morphglobal"}:
        return build_temporal_attention_autoencoder(
            input_shape=input_shape,
            latent_dim=latent_dim,
            attention_dim=attention_dim,
            name=name,
        )

    if architecture in {"sequence", "sequence_masked", "masked_sequence"}:
        return build_sequence_bottleneck_autoencoder(
            input_shape=input_shape,
            latent_dim=latent_dim,
            attention_dim=attention_dim,
            name=name,
        )

    raise ValueError(f"Unknown medattnaid_architecture: {architecture}")


__all__ = [
    "CONFIG",
    "TemporalAttention",
    "LearnedPositionEmbedding",
    "TimestepMask",
    "build_temporal_attention_autoencoder",
    "build_sequence_bottleneck_autoencoder",
    "build_medattnaid_model",
]
