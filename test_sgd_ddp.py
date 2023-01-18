import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP

import asdl
from model import get_vgg_tiny


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
            observe_dict["iter0_grad"] = grad_list
        nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
    return observe_dict


def _dist_sgd_worker(
    gpu, model_func, world_size, images, targets, data_split,
    lr, momentum, weight_decay, dump_prefix
):
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
    model = DDP(model, device_ids=[gpu])

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay, nesterov=True)
    grad_maker = asdl.GradientMaker(model)

    observe_dict = {
        **observe_dict,
        **_train(model, optimizer, grad_maker, images_list, targets_list, gpu),
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
        _dist_sgd_worker(
            0, self.model_func, 1, *self._get_data_copy(), self.single_data_split,
            0, 0.9, 1e-5, os.path.join(self.temp_dir, "single")
        )

    def _run_dist(self):
        torch.multiprocessing.spawn(
            _dist_sgd_worker,
            args=(
                self.model_func, 2, *self._get_data_copy(), self.dist_data_split,
                0, 0.9, 1e-5, os.path.join(self.temp_dir, "dist")
            ),
            nprocs=2
        )

    def _test_grad_equal(self):
        model_0, model_1, model_2 = get_attr([self.obs_single, self.obs_dist_0, self.obs_dist_1], "iter0_grad")
        for grad_0, grad_1, grad_2 in zip(model_0, model_1, model_2):
            torch.testing.assert_close(grad_0, grad_1)
            assert torch.all(grad_1 == grad_2)
        print("_test_grad_equal PASS")

    def test(self):
        # pass
        self._test_grad_equal()


if __name__ == "__main__":
    set_seed(0)
    TestNGDDDP(
        None, get_vgg_tiny, "temp",
        8, 3, 32, 10, [[[0, 1, 2, 3, 4, 5, 6, 7]]], [[[0, 1, 2, 3]], [[4, 5, 6, 7]]],
        [nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm]
    ).test()
