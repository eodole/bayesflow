from keras import ops

from bayesflow.experimental.types import Tensor
from .invertible_layer import InvertibleLayer


class ActNorm(InvertibleLayer):
    """Implements an Activation Normalization (ActNorm) Layer.
    Activation Normalization is learned invertible normalization, using
    a Scale (s) and Bias (b) vector::

       y = s * x + b (forward)
       x = (y - b) / s (inverse)

    References
    ----------

    .. [1] Kingma, D. P., & Dhariwal, P. (2018). 
        Glow: Generative flow with invertible 1x1 convolutions. 
        Advances in Neural Information Processing Systems, 31.

    .. [2] Salimans, Tim, and Durk P. Kingma. (2016).
       Weight normalization: A simple reparameterization to accelerate
       training of deep neural networks.
       Advances in Neural Information Processing Systems, 29.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scale = None
        self.bias = None

    def build(self, input_shape):
        self.scale = self.add_weight(shape=(input_shape[-1],), initializer="ones", name="scale")
        self.bias = self.add_weight(shape=(input_shape[-1],), initializer="zeros", name="bias")

    def call(self, xz: Tensor, inverse: bool = False):
        if inverse:
            return self._inverse(xz)
        return self._forward(xz)

    def _forward(self, x: Tensor) -> (Tensor, Tensor):
        z = self.scale * x + self.bias
        log_det = ops.sum(ops.log(ops.abs(self.scale)), axis=-1)
        return z, log_det

    def _inverse(self, z: Tensor) -> (Tensor, Tensor):
        x = (z - self.bias) / self.scale
        log_det = -ops.sum(ops.log(ops.abs(self.scale)), axis=-1)
        return x, log_det