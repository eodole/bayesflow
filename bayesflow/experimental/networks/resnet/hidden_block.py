
import keras
from keras import layers, regularizers
from keras.saving import (
    register_keras_serializable,
)

from bayesflow.experimental.types import Tensor


@register_keras_serializable(package="bayesflow.networks.resnet")
class ConfigurableHiddenBlock(keras.layers.Layer):
    def __init__(
        self,
        units: int = 256,
        activation: str = "gelu",
        kernel_regularizer: regularizers.Regularizer | None = None,
        bias_regularizer: regularizers.Regularizer | None = None,
        kernel_initializer: str = "he_uniform",
        residual: bool = True,
        dropout_rate: float = 0.05,
        spectral_normalization: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.activation_fn = keras.activations.get(activation)
        self.residual = residual
        self.dense = layers.Dense(
            units=units,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer,
            bias_regularizer=bias_regularizer
        )
        if spectral_normalization:
            self.dense = layers.SpectralNormalization(self.dense)
        self.dropout = keras.layers.Dropout(dropout_rate)

    def call(self, inputs: Tensor, training=False):
        x = self.dense(inputs, training=training)
        x = self.dropout(x, training=training)
        if self.residual:
            x = x + inputs
        return self.activation_fn(x)

    def build(self, input_shape):
        super().build(input_shape)
        self(keras.KerasTensor(input_shape))

    def get_config(self):
        config = super().get_config()
        config.update({
            "residual": self.residual,
            "activation_fn": keras.saving.serialize_keras_object(self.activation_fn),
            "dense": keras.saving.serialize_keras_object(self.dense),
            "dropout": keras.saving.serialize_keras_object(self.dropout)
        })
        return config
