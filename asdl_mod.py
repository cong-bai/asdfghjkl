import asdl
from asdl.matrices import *


_invalid_ema_decay = -1
_module_level_shapes = [SHAPE_LAYER_WISE, SHAPE_KRON, SHAPE_SWIFT_KRON, SHAPE_UNIT_WISE, SHAPE_DIAG]


class KfacGradientMakerForTest(asdl.KfacGradientMaker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_dict = {}

    def forward_and_backward(self):
        step = self.state['step']

        self._startup()

        if self.do_forward_and_backward(step):
            self.forward()
            self.backward()
            self.obs_dict["iter0_grad_1st"] = [p.grad.detach().cpu() for p in self.model.parameters()]
        if self.do_update_curvature(step):
            self.update_curvature()

        self.obs_dict["kron"] = {}
        for enum_shape, shape in enumerate(_module_level_shapes):
            for enum_module, module in enumerate(self.modules_for(shape)):
                if self.world_rank == self.partitions[enum_shape][enum_module]:
                    if not self.is_module_for_inv_and_precondition(module):
                        continue
                    matrix = self._get_module_symmatrix(module, shape)
                    self.obs_dict["kron"][enum_module] = {
                        "A_inv": matrix.A_inv.detach().cpu(), "B_inv":matrix.B_inv.detach().cpu()
                    }

        if self.do_update_preconditioner(step):
            self.update_preconditioner()

        self.precondition()

        self._teardown()

        self.state['step'] += 1

        return self._model_output, self._loss

    def get_named_fisher_from_model(self):
        """
        returns a list of all the tensors of the FIM
        """
        tensor_list = []
        for shape in _module_level_shapes:
            keys_list = self._keys_list_from_shape(shape)
            for name, module in self.named_modules_for(shape):
                for keys in keys_list:
                    tensor = self.fisher_maker.get_fisher_tensor(module, *keys)
                    if tensor is None:
                        continue
                    tensor_list.append(tensor)
        return tensor_list

    def update_curvature(self):
        config = self.config
        fisher_maker = self.fisher_maker
        scale = self.scale

        ema_decay = config.ema_decay
        if ema_decay != _invalid_ema_decay:
            scale *= ema_decay
            self._scale_fisher(1 - ema_decay)

        self.delegate_forward_and_backward(
            fisher_maker, data_size=self.config.data_size, scale=scale,
            accumulate=self.do_accumulate, calc_loss_grad=True,
            calc_inv=not self.do_accumulate, damping=self.config.damping
        )

        self.obs_dict["iter0_grad_1st"] = [p.grad.detach().cpu() for p in self.model.parameters()]

        if self.do_accumulate and self.world_size > 1:
            self.reduce_scatter_curvature()
