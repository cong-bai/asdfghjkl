import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP

import asdl
from asdl.matrices import *
from ddp import sync_params_and_buffers
from model import get_vgg_tiny


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
        if self.do_update_curvature(step):
            self.update_curvature()
        fisher_list = self.get_named_fisher_from_model() # For testing
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

        self.delegate_forward_and_backward(fisher_maker,
                                           data_size=self.config.data_size,
                                           scale=scale,
                                           accumulate=self.do_accumulate,
                                           calc_loss_grad=True,
                                           calc_inv=not self.do_accumulate,
                                           damping=self.config.damping
                                           )

        self.obs_dict["iter0_grad_1st"] = [p.grad.detach().cpu() for p in self.model.parameters()]

        if self.do_accumulate and self.world_size > 1:         
            self.reduce_scatter_curvature()


def get_attr(target_list, key):
    return [target[key] for target in target_list]


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_batched_data(data_size, img_channel, img_size, cls_num):
    images = torch.randn((data_size, img_channel, img_size, img_size), dtype=torch.float32)
    targets = torch.randint(cls_num, (data_size,))
    return images, targets


def _train(model, optimizer, grad_maker, images_list, targets_list, gpu):
    observe_dict = {}
    model.train()
    for i, (images, targets) in enumerate(zip(images_list, targets_list)):
        images, targets = images.cuda(gpu), targets.cuda(gpu)
        # TODO: This should be fixed in later asdl update
        loss_func = nn.CrossEntropyLoss()

        optimizer.zero_grad()
        dummy_y = grad_maker.setup_model_call(model, images)
        grad_maker.setup_loss_call(loss_func, dummy_y, targets)
        outputs, loss = grad_maker.forward_and_backward()
        if i == 0:
            grad_list = [param.grad.detach().cpu() for param in model.parameters()]
            observe_dict["iter0_grad_2nd"] = grad_list
            observe_dict["iter0_loss"] = loss.detach().cpu()
        nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
    return observe_dict


def _dist_kfac_worker(gpu, model_func, world_size, images, targets, data_split,
                      lr, momentum, weight_decay, damping, ema_decay,
                      curvature_upd_interval, preconditioner_upd_interval,
                      ignore_modules, fisher_type, dump_prefix):
    observe_dict = {}
    set_seed(gpu)
    dist.init_process_group(backend="nccl", init_method='tcp://localhost:36000',
                            world_size=world_size, rank=gpu)

    batch_size = len(data_split[gpu][0])
    for batch in data_split[gpu]:
        assert len(batch) == batch_size
    images_list = [images[indexes] for indexes in data_split[gpu]]
    targets_list = [targets[indexes] for indexes in data_split[gpu]]

    model = model_func().cuda(gpu)
    observe_dict["init_model_before"] = {k: v.cpu() for i, (k, v) in enumerate(model.state_dict().items())}
    if world_size > 1:
        sync_params_and_buffers(model)
        observe_dict["init_model_after"] = {k: v.cpu() for i, (k, v) in enumerate(model.state_dict().items())}

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay, nesterov=True)
    config = asdl.PreconditioningConfig(
        data_size=batch_size,
        damping=damping,
        curvature_upd_interval=curvature_upd_interval,
        preconditioner_upd_interval=preconditioner_upd_interval,
        ignore_modules=ignore_modules,
        ema_decay=ema_decay,
    )
    grad_maker = KfacGradientMakerForTest(model, config, fisher_type=fisher_type, swift=False)

    observe_dict = {
        **observe_dict,
        **_train(model, optimizer, grad_maker, images_list, targets_list, gpu),
        **grad_maker.obs_dict
    }
    torch.save(observe_dict, dump_prefix + f"_{gpu}.pt")
    dist.destroy_process_group()


class TestNGDDDP:
    def __init__(
        self, fisher_type, model_func, temp_dir,
        data_size, img_channel, img_size, cls_num, single_data_split, dist_data_split,
        ignore_modules
    ):
        assert torch.cuda.device_count() > 1

        self.fisher_type = fisher_type
        self.model_func = model_func
        self.temp_dir = temp_dir
        self.ignore_modules = ignore_modules

        self.single_data_split = single_data_split
        self.dist_data_split = dist_data_split
        self.images, self.targets = get_batched_data(data_size, img_channel, img_size, cls_num)
        self._run_single()
        self._run_dist()
        self.obs_single = torch.load(os.path.join(temp_dir, "single_0.pt"))
        self.obs_dist_0 = torch.load(os.path.join(temp_dir, "dist_0.pt"))
        self.obs_dist_1 = torch.load(os.path.join(temp_dir, "dist_1.pt"))

    def _get_data_copy(self):
        return torch.clone(self.images), torch.clone(self.targets)

    def _run_single(self):
        _dist_kfac_worker(
            0, self.model_func, 1, *self._get_data_copy(), self.single_data_split,
            0, 0.9, 1e-5, 1e-7, 1e-4, 5, 10,
            self.ignore_modules, self.fisher_type, os.path.join(self.temp_dir, "single")
        )

    def _run_dist(self):
        torch.multiprocessing.spawn(
            _dist_kfac_worker,
            args=(
                self.model_func, 2, *self._get_data_copy(), self.dist_data_split,
                0, 0.9, 1e-5, 1e-7, 1e-4, 5, 10,
                self.ignore_modules, self.fisher_type, os.path.join(self.temp_dir, "dist")
            ),
            nprocs=2
        )

    def _test_model_init(self):
        # Check that initial model is correctly synced
        # Load the saved initial model and iterate over the state dict
        model_0, model_1, model_2 = get_attr([self.obs_single, self.obs_dist_0, self.obs_dist_1], "init_model_before")
        flag = False
        for (_, v0), (_, v1), (_, v2) in zip(model_0.items(), model_1.items(), model_2.items()):
            assert torch.all(v0 == v1)
            if torch.any(v1 != v2):
                flag = True
        assert flag
        model_1, model_2 = get_attr([self.obs_dist_0, self.obs_dist_1], "init_model_after")
        for (_, v0), (_, v1), (_, v2) in zip(model_0.items(), model_1.items(), model_2.items()):
            assert torch.all(v0 == v1)
            assert torch.all(v2 == v1)
        print("_test_model_init PASS")

    def _test_loss(self):
        loss_0, loss_1, loss_2 = get_attr([self.obs_single, self.obs_dist_0, self.obs_dist_1], "iter0_loss")
        torch.testing.assert_close((loss_1 + loss_2) / 2, loss_0)
        print("_test_loss PASS")
    
    def _test_1st_grad_equal(self):
        model_0, model_1, model_2 = get_attr([self.obs_single, self.obs_dist_0, self.obs_dist_1], "iter0_grad_1st")
        try:
            for i, (grad_0, grad_1, grad_2) in enumerate(zip(model_0, model_1, model_2)):
                torch.testing.assert_close((grad_1 + grad_2) / 2, grad_0)
        except AssertionError as e:
            print(f"Grad #{i} are not close!")
            raise e
        print("_test_1st_grad_equal PASS")

    def _test_2nd_grad_equal(self):
        model_0, model_1, model_2 = get_attr([self.obs_single, self.obs_dist_0, self.obs_dist_1], "iter0_grad_2nd")
        for grad_0, grad_1, grad_2 in zip(model_0, model_1, model_2):
            torch.testing.assert_close(grad_0, grad_1)
            assert torch.all(grad_1 == grad_2)
        print("_test_2nd_grad_equal PASS")

    def test(self):
        # pass
        self._test_model_init()
        self._test_loss()
        self._test_1st_grad_equal()
        self._test_2nd_grad_equal()


if __name__ == "__main__":
    set_seed(4)
    TestNGDDDP(
        FISHER_MC, get_vgg_tiny, "temp",
        # 8, 3, 32, 10, [[[0, 1, 2, 3, 4, 5, 6, 7]]], [[[0, 1, 2, 3]], [[4, 5, 6, 7]]],
        2, 3, 32, 10, [[[0, 1]]], [[[0]], [[1]]],
        [nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm]
    ).test()
