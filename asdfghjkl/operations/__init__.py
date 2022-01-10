import warnings
from .operation import *
from .linear import Linear
from .conv import Conv2d
from .batchnorm import BatchNorm1d, BatchNorm2d
from .layernorm import LayerNorm
from .bias import Bias, BiasExt
from .scale import Scale, ScaleExt
from .embeddings import ViTEmbeddings
from ..utils import vit_check

__all__ = [
    'Linear',
    'Conv2d',
    'BatchNorm1d',
    'BatchNorm2d',
    'LayerNorm',
    'Bias',
    'Scale',
    'BiasExt',
    'ScaleExt',
    'ViTEmbeddings',
    'get_op_class',
    'Operation',
    'OP_FULL_COV',
    'OP_FULL_CVP',
    'OP_COV',
    'OP_CVP',
    'OP_COV_KRON',
    'OP_COV_DIAG',
    'OP_COV_UNIT_WISE',
    'OP_RFIM_RELU',
    'OP_RFIM_SOFTMAX',
    'OP_GRAM_DIRECT',
    'OP_GRAM_HADAMARD',
    'OP_BATCH_GRADS',
    'ALL_OPS',
    'OperationManager'
]


def get_op_class(module):
    if isinstance(module, nn.Linear):
        return Linear
    elif isinstance(module, nn.Conv2d):
        return Conv2d
    elif isinstance(module, nn.BatchNorm1d):
        return BatchNorm1d
    elif isinstance(module, nn.BatchNorm2d):
        return BatchNorm2d
    elif isinstance(module, nn.LayerNorm):
        return LayerNorm
    elif isinstance(module, Bias):
        return BiasExt
    elif isinstance(module, Scale):
        return ScaleExt
    elif vit_check(module):
        return ViTEmbeddings
    else:
        warnings.warn(f'Failed to lookup operations for Module {module}.')
        return None
