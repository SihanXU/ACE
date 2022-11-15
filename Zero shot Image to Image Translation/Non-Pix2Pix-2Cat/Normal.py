import paddle
import paddle.nn as nn
import math
import functools

class _SpectralNorm(nn.SpectralNorm):
    def __init__(self,
                 weight_shape,
                 dim=0,
                 power_iters=1,
                 eps=1e-12,
                 dtype='float32'):
        super(_SpectralNorm, self).__init__(weight_shape, dim, power_iters, eps,
                                            dtype)

    def forward(self, weight):
        inputs = {'Weight': weight, 'U': self.weight_u, 'V': self.weight_v}
        out = self._helper.create_variable_for_type_inference(self._dtype)
        _power_iters = self._power_iters if self.training else 0
        self._helper.append_op(type="spectral_norm",
                               inputs=inputs,
                               outputs={
                                   "Out": out,
                               },
                               attrs={
                                   "dim": self._dim,
                                   "power_iters": _power_iters,
                                   "eps": self._eps,
                               })

        return out


class Spectralnorm(paddle.nn.Layer):
    def __init__(self, layer, dim=None, power_iters=1, eps=1e-12, dtype='float32'):
        super(Spectralnorm, self).__init__()

        if dim is None: # conv: dim = 1, Linear: dim = 0
            if isinstance(layer, (nn.Conv1D, nn.Conv2D, nn.Conv3D, 
                    nn.Conv1DTranspose, nn.Conv2DTranspose, nn.Conv3DTranspose)):
                dim = 1
            else:
                dim = 0
        if layer.training == False: # don't do power iterations when infering
            power_iters = 0

        self.spectral_norm = _SpectralNorm(layer.weight.shape, dim, power_iters,
                                           eps, dtype)
        self.dim = dim
        self.power_iters = power_iters
        self.eps = eps
        self.layer = layer
        weight = layer._parameters['weight']
        del layer._parameters['weight']
        self.weight_orig = self.create_parameter(weight.shape,
                                                 dtype=weight.dtype)
        self.weight_orig.set_value(weight)

    def forward(self, x):
        weight = self.spectral_norm(self.weight_orig)
        self.layer.weight = weight
        out = self.layer(x)
        return out

def build_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Args:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we do not use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(
            nn.BatchNorm2D,
            weight_attr=False,
            bias_attr=False)
    elif norm_type == 'instance':
        norm_layer = functools.partial(
            nn.InstanceNorm2D,
            weight_attr=False,
            bias_attr=False)
    elif norm_type == 'spectral':
        norm_layer = functools.partial(Spectralnorm)

    else:
        raise NotImplementedError('normalization layer [%s] is not found' %
                                  norm_type)
    return norm_layer