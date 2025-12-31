import keras

from bayesflow.types import Shape, Tensor
from bayesflow.utils.serialization import serializable

from bayesflow.networks.inference_network import InferenceNetwork
# from bayesflow.networks.coupling_flow import CouplingFlow

from bayesflow.networks.coupling_flow import CouplingFlow
# from bayesflow.networks.coupling_flow import DualCoupling

from .conditional_gaussian import ConditionalGaussian


# disable module check, use potential module after moving from experimental
@serializable("bayesflow.networks", disable_module_check=True)
class CIF(InferenceNetwork):
    """(IN) Implements a continuously indexed flow (CIF) with a `CouplingFlow`
    bijection and `ConditionalGaussian` distributions p and q. Improves on
    eliminating leaky sampling found topologically in normalizing flows.
    Built in reference to [1].

    [1] R. Cornish, A. Caterini, G. Deligiannidis, & A. Doucet (2021).
    Relaxing Bijectivity Constraints with Continuously Indexed Normalising
    Flows.
    arXiv:1909.13833.
    """

    def __init__(
        self, pq_depth: int = 4, pq_width: int = 128, pq_activation: str = "swish", layers_L: int = 3, **kwargs
    ):
        """Creates an instance of a `CIF` with configurable
        `ConditionalGaussian` distributions p and q, each containing MLP
        networks

        Parameters
        ----------
        pq_depth: int, optional, default: 4
            The number of MLP hidden layers (minimum: 1)
        pq_width: int, optional, default: 128
            The dimensionality of the MLP hidden layers
        pq_activation: str, optional, default: 'tanh'
            The MLP activation function
        layers_L: int, number of layers in CIF
        """

        super().__init__(base_distribution="normal", **kwargs)
        # self.bijection = CouplingFlow()
        # these are basically the neural net that we are learning params from, this implementation uses
        # a conditional gaussian.. how does this relate to the paper?
        # both p and q have a nn representing the means and standard dev.
        # self.base_dist = ConditionalGaussian(depth=pq_depth, width=pq_width, activation=pq_activation)
        self.p_dists = []
        self.q_dists = []
        self.layers = []
        self.layers_L = layers_L

        for i in range(layers_L):
            p_dist = ConditionalGaussian(depth=pq_depth, width=pq_width, activation=pq_activation)
            q_dist = ConditionalGaussian(depth=pq_depth, width=pq_width, activation=pq_activation)
            bijection = CouplingFlow()

            self.p_dists.append(p_dist)
            self.q_dists.append(q_dist)
            self.layers.append(bijection)

    def build(self, xz_shape: Shape, conditions_shape: Shape = None) -> None:
        for i in range(self.layers_L):
            self.p_dists[i].build(xz_shape)
            self.q_dists[i].build(xz_shape)
            self.layers[i].build(xz_shape, conditions_shape=conditions_shape)

        super().build(xz_shape)

    def call(
        self, xz: Tensor, conditions: Tensor = None, inverse: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        if inverse:
            return self._inverse(xz, conditions=conditions, **kwargs)
        return self._forward(xz, conditions=conditions, **kwargs)

    def _forward(  # distribution direction
        self, x: Tensor, conditions: Tensor = None, density: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        z = x
        elbo = 0.0
        for layer_idx in reversed(range(self.layers_L)):
            layer = self.layers[layer_idx]
            q_dist = self.q_dists[layer_idx]
            p_dist = self.p_dists[layer_idx]

            # Sample u ~ q(u | z_l) where z_l is current z
            u, log_qu = q_dist.sample(z, log_prob=True)

            # Bijection and log Jacobian x -> z
            # z_{l-1} = F^{-1}(z_l;u) and log Jac

            z_prev, log_jac = layer(z, conditions=keras.ops.concatenate([conditions, u], axis=-1), density=True)
            if log_jac.ndim > 1:
                log_jac = keras.ops.sum(log_jac, axis=1)

            # Log prob over p on u with conditions z
            # log p(u | z_{l-1})
            log_pu = p_dist.log_prob(u, z_prev)

            # we cannot compute an exact analytical density
            # Update ELBO: += log p(u|z_{l-1}) - log q(u|z_l) + log|det J|
            elbo = log_pu - log_qu + log_jac  # +

            # Update Z
            z = z_prev

        # Prior log prob
        log_prior = self.base_distribution.log_prob(z)
        if log_prior.ndim > 1:
            log_prior = keras.ops.sum(log_prior, axis=1)

        elbo = elbo + log_prior

        if density:
            return z, elbo

        return z

    def _inverse(  # sampling direction
        self, z: Tensor, conditions: Tensor = None, density: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        # compute bijection z -> x

        x = z
        log_prob_sum = 0.0

        for layer_index in range(len(self.layers)):
            layer = self.layers[layer_index]
            p_dist = self.p_dists[layer_index]

            # sample u ~ p(u | z_{l-1})
            u = p_dist.sample(x)

            # compute x_l = F(x_{l-1};u)
            x, log_jac = layer(x, conditions=keras.ops.concatenate([conditions, u], axis=-1), inverse=True)
            log_pu = p_dist.log_prob(u, x)
            log_prob_sum = log_prob_sum + log_pu + log_jac

        if not density:
            return x

        return x, log_prob_sum

    # def F(self, z,u):
    #     s = self.s_net
    #     t = self.t_net
    #     return self.bijection(keras.layers.Multiply()[keras.ops.exp(-s(u)), z - t(u)])

    def compute_metrics(self, x: Tensor, conditions: Tensor = None, stage: str = "training") -> dict[str, Tensor]:
        base_metrics = super().compute_metrics(x, conditions=conditions, stage=stage)

        elbo = self.log_prob(x, conditions=conditions)

        loss = -keras.ops.mean(elbo)

        return base_metrics | {"loss": loss}
