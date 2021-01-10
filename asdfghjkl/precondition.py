import torch
from torch import nn

from .matrices import FISHER_EXACT, SHAPE_FULL, SHAPE_BLOCK_DIAG, SHAPE_KRON, SHAPE_DIAG  # NOQA
from .fisher import fisher_for_cross_entropy
from .utils import add_value_to_diagonal

_supported_modules = (nn.Linear, nn.Conv2d, nn.BatchNorm1d, nn.BatchNorm2d)
_normalizations = (nn.BatchNorm1d, nn.BatchNorm2d)

__all__ = [
    'Precondition', 'NaturalGradient', 'LayerWiseNaturalGradient', 'KFAC',
    'DiagNaturalGradient'
]


class Precondition:
    def __init__(self):
        pass

    def update_curvature(self, inputs=None, targets=None, data_loader=None):
        raise NotImplementedError

    def accumulate_curvature(self):
        raise NotImplementedError

    def finalize_accumulation(self):
        raise NotImplementedError

    def reduce_curvature(self):
        raise NotImplementedError

    def update_inv(self, damping=None):
        raise NotImplementedError

    def precondition(self):
        raise NotImplementedError

    def precondition_vector(self, vec):
        raise NotImplementedError


class NaturalGradient(Precondition):
    def __init__(self,
                 model,
                 fisher_type=FISHER_EXACT,
                 pre_inv_postfix=None,
                 n_mc_samples=1,
                 damping=1e-5,
                 ):
        self.model = model
        self.modules = [model]
        self.fisher_type = fisher_type
        self.n_mc_samples = n_mc_samples
        self.damping = damping
        super().__init__()
        self.fisher_shape = SHAPE_FULL
        self.fisher_manager = None
        self._pre_inv_postfix = pre_inv_postfix

    def _get_fisher_attr(self, postfix=None):
        if postfix is None:
            return self.fisher_type
        else:
            return f'{self.fisher_type}_{postfix}'

    def _get_fisher(self, module, postfix=None):
        attr = self._get_fisher_attr(postfix)
        fisher = getattr(module, attr, None)
        return fisher

    @property
    def _pre_inv_attr(self):
        return self._get_fisher_attr(self._pre_inv_postfix)

    def _get_pre_inv_fisher(self, module):
        return getattr(module, self._pre_inv_attr, None)

    def _set_fisher(self, module, data, postfix=None):
        attr = self._get_fisher_attr(postfix)
        setattr(module, attr, data)

    def _clear_fisher(self, module, postfix=None):
        attr = self._get_fisher_attr(postfix)
        if hasattr(module, attr):
            delattr(module, attr)

    def update_curvature(self, inputs=None, targets=None, data_loader=None):
        rst = fisher_for_cross_entropy(self.model,
                                       inputs=inputs,
                                       targets=targets,
                                       data_loader=data_loader,
                                       fisher_types=self.fisher_type,
                                       fisher_shapes=self.fisher_shape,
                                       n_mc_samples=self.n_mc_samples)
        self.fisher_manager = rst

    def move_curvature(self, postfix, scale=1., to_pre_inv=False):
        self.accumulate_curvature(postfix, scale, to_pre_inv, replace=True)

    def accumulate_curvature(self, postfix='acc', scale=1., to_pre_inv=False, replace=False):
        if to_pre_inv:
            postfix = self._pre_inv_postfix
        for module in self.modules:
            fisher = self._get_fisher(module)
            if fisher is None:
                continue
            fisher.scaling(scale)
            fisher_acc = self._get_fisher(module, postfix)
            if fisher_acc is None or replace:
                self._set_fisher(module, fisher, postfix)
            else:
                self._set_fisher(module, fisher_acc + fisher, postfix)
            self._clear_fisher(module)

    def finalize_accumulation(self, postfix='acc'):
        for module in self.modules:
            fisher_acc = self._get_fisher(module, postfix)
            assert fisher_acc is not None
            self._set_fisher(module, fisher_acc)
            self._clear_fisher(module, postfix)

    def reduce_curvature(self, all_reduce=True):
        self.fisher_manager.reduce_matrices(all_reduce=all_reduce)

    def update_inv(self, damping=None):
        if damping is None:
            damping = self.damping

        for module in self.modules:
            fisher = self._get_pre_inv_fisher(module)
            if fisher is None:
                continue
            inv = _cholesky_inv(add_value_to_diagonal(fisher.data, damping))
            setattr(fisher, 'inv', inv)

    def precondition(self):
        grads = []
        for p in self.model.parameters():
            if p.requires_grad and p.grad is not None:
                grads.append(p.grad.flatten())
        g = torch.cat(grads)
        fisher = self._get_pre_inv_fisher(self.model)
        ng = torch.mv(fisher.inv, g)

        pointer = 0
        for p in self.model.parameters():
            if p.requires_grad and p.grad is not None:
                numel = p.grad.numel()
                val = ng[pointer:pointer + numel]
                p.grad.copy_(val.reshape_as(p.grad))
                pointer += numel

        assert pointer == ng.numel()


class LayerWiseNaturalGradient(NaturalGradient):
    def __init__(self,
                 model,
                 fisher_type=FISHER_EXACT,
                 pre_inv_postfix=None,
                 n_mc_samples=1,
                 damping=1e-5):
        super().__init__(model, fisher_type, pre_inv_postfix, n_mc_samples, damping)
        self.fisher_shape = SHAPE_BLOCK_DIAG
        self.modules = [
            m for m in model.modules() if isinstance(m, _supported_modules)
        ]

    def precondition(self):
        for module in self.modules:
            fisher = self._get_pre_inv_fisher(module)
            if fisher is None:
                continue
            g = module.weight.grad.flatten()
            if _bias_requires_grad(module):
                g = torch.cat([g, module.bias.grad.flatten()])
            ng = torch.mv(fisher.inv, g)

            if _bias_requires_grad(module):
                w_numel = module.weight.numel()
                grad_w = ng[:w_numel]
                module.bias.grad.copy_(ng[w_numel:])
            else:
                grad_w = ng
            module.weight.grad.copy_(grad_w.reshape_as(module.weight.grad))


class KFAC(NaturalGradient):
    def __init__(self,
                 model,
                 fisher_type=FISHER_EXACT,
                 pre_inv_postfix=None,
                 n_mc_samples=1,
                 damping=1e-5):
        super().__init__(model, fisher_type, pre_inv_postfix, n_mc_samples, damping)
        self.fisher_shape = SHAPE_KRON
        self.modules = [
            m for m in model.modules() if isinstance(m, _supported_modules)
        ]

    def update_inv(self, damping=None):
        if damping is None:
            damping = self.damping

        for module in self.modules:
            fisher = self._get_pre_inv_fisher(module)
            if fisher is None:
                continue
            if isinstance(module, _normalizations):
                pass
            else:
                A = fisher.kron.A
                B = fisher.kron.B
                A_eig_mean = A.trace() / A.shape[0]
                B_eig_mean = B.trace() / B.shape[0]
                pi = torch.sqrt(A_eig_mean / B_eig_mean)
                r = damping**0.5

                A_inv = _cholesky_inv(add_value_to_diagonal(A, r * pi))
                B_inv = _cholesky_inv(add_value_to_diagonal(B, r / pi))

                setattr(fisher.kron, 'A_inv', A_inv)
                setattr(fisher.kron, 'B_inv', B_inv)

    def precondition(self):
        for module in self.modules:
            fisher = self._get_pre_inv_fisher(module)
            if fisher is None:
                continue
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                pass
            else:
                A_inv = fisher.kron.A_inv
                B_inv = fisher.kron.B_inv
                grad2d = module.weight.grad.view(B_inv.shape[0], -1)
                if _bias_requires_grad(module):
                    grad2d = torch.cat(
                        [grad2d, module.bias.grad.unsqueeze(dim=1)], dim=1)
                ng = B_inv.mm(grad2d).mm(A_inv)
                if _bias_requires_grad(module):
                    grad_w = ng[:, :-1]
                    module.bias.grad.copy_(ng[:, -1])
                else:
                    grad_w = ng
                module.weight.grad.copy_(grad_w.reshape_as(module.weight.grad))


class DiagNaturalGradient(NaturalGradient):
    def __init__(self,
                 model,
                 fisher_type=FISHER_EXACT,
                 pre_inv_postfix=None,
                 n_mc_samples=1,
                 damping=1e-5):
        super().__init__(model, fisher_type, pre_inv_postfix, n_mc_samples, damping)
        self.fisher_shape = SHAPE_DIAG
        self.modules = [
            m for m in model.modules() if isinstance(m, _supported_modules)
        ]

    def update_inv(self, damping=None):
        if damping is None:
            damping = self.damping

        for module in self.modules:
            fisher = self._get_pre_inv_fisher(module)
            if fisher is None:
                continue
            elif isinstance(module, _supported_modules):
                diag_w = fisher.diag.weight
                setattr(fisher.diag, 'weight_inv', 1 / (diag_w + damping))
                if _bias_requires_grad(module):
                    diag_b = fisher.diag.bias
                    setattr(fisher.diag, 'bias_inv', 1 / (diag_b + damping))

    def precondition(self):
        for module in self.modules:
            fisher = self._get_pre_inv_fisher(module)
            if fisher is None:
                continue
            w_inv = fisher.diag.weight_inv
            module.weight.grad.mul_(w_inv)
            if _bias_requires_grad(module):
                b_inv = fisher.diag.bias_inv
                module.bias.grad.mul_(b_inv)

    def precondition_vector(self, vec):
        idx = 0
        rst = []
        for module in self.modules:
            fisher = self._get_pre_inv_fisher(module)
            if fisher is None:
                continue
            assert fisher.diag is not None, module
            rst.append(vec[idx].mul(fisher.diag.weight_inv))
            idx += 1
            if _bias_requires_grad(module):
                rst.append(vec[idx].mul(fisher.diag.bias_inv))
                idx += 1

        assert idx == len(vec)
        return rst


def _bias_requires_grad(module):
    return hasattr(module, 'bias') and module.bias.requires_grad


def _cholesky_inv(X):
    u = torch.cholesky(X)
    return torch.cholesky_inverse(u)

