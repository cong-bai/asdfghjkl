import os

import torch
from torch import nn

import asdl
from asdl.matrices import *
from dist_worker import dist_worker
from test_utils.model import get_vgg_tiny
from test_utils.utils import get_batched_data, set_seed


def run_single_node(
    model_func, images, targets, data_split, lr, momentum, weight_decay, dump_dir,
    gradmaker_config=None, fisher_type=None
):
    dist_worker(
        0, model_func, 1, images, targets, data_split, lr, momentum, weight_decay,
        os.path.join(dump_dir, "single"), gradmaker_config, fisher_type
    )
    return torch.load(os.path.join(dump_dir, "single_0.pt"))


def run_dist(
    model_func, images, targets, data_split, lr, momentum, weight_decay, dump_dir,
    gradmaker_config=None, fisher_type=None
):
    torch.multiprocessing.spawn(
        dist_worker,
        args=(
            model_func, 2, images, targets, data_split, lr, momentum, weight_decay,
            os.path.join(dump_dir, "dist"), gradmaker_config, fisher_type
        ),
        nprocs=2
    )
    return [torch.load(os.path.join(dump_dir, f"dist_{i}.pt")) for i in [0, 1]]


class TestDistGradMaker:
    def __init__(
        self, model_func, dump_dir, data_size, img_channel, img_size, cls_num,
        single_data_split, dist_data_split, config=None, fisher_type=None
    ):
        assert torch.cuda.device_count() > 1

        self.model_func = model_func
        self.dump_dir = dump_dir
        self.single_data_split = single_data_split
        self.dist_data_split = dist_data_split
        self.images, self.targets = get_batched_data(data_size, img_channel, img_size, cls_num)
        self.config = config
        self.fisher_type = fisher_type

        self.obs_single = run_single_node(
            model_func, *self._get_data_copy(), single_data_split, 0.1, 0.9, 1e-4, "temp",
            config, fisher_type
        )
        self.obs_dist_0, self.obs_dist_1 = run_dist(
            model_func, *self._get_data_copy(), dist_data_split, 0.1, 0.9, 1e-4, "temp",
            config, fisher_type
        )

    def _get_data_copy(self):
        return torch.clone(self.images), torch.clone(self.targets)

    def _sanity_check(self, device):
        # We add this because we found sometimes the output is not accurate
        # For example this model once passed on cpu but failed on gpu:
        # torch.nn.Sequential(
        #     torch.nn.Conv2d(3, 64, kernel_size=3).to(device),
        #     torch.nn.Conv2d(64, 64, kernel_size=3).to(device)
        # )
        input_data, _ = get_batched_data(8, 3, 32, 10)
        input_data = input_data.to(device)
        model = self.model_func().to(device)
        with torch.inference_mode():
            torch.testing.assert_close(
                model(input_data[:4]), model(input_data)[:4], rtol=0, atol=0
            )
        print(f"_sanity_check on {device} PASS")

    def _test_model_init(self):
        # Check that initial model is correctly synced
        # Load the saved initial model and iterate over the state dict
        model_0, model_1, model_2 = [
            o["init_model_before"]
            for o in [self.obs_single, self.obs_dist_0, self.obs_dist_1]
        ]
        flag = False
        for (_, v0), (_, v1), (_, v2) in zip(model_0.items(), model_1.items(), model_2.items()):
            assert torch.all(v0 == v1)
            if torch.any(v1 != v2):
                flag = True
        assert flag
        model_1, model_2 = [o["init_model_after"] for o in [self.obs_dist_0, self.obs_dist_1]]
        for (_, v0), (_, v1), (_, v2) in zip(model_0.items(), model_1.items(), model_2.items()):
            assert torch.all(v0 == v1) and torch.all(v2 == v1)
        print("_test_model_init PASS")

    def _test_output(self):
        out = torch.cat([self.obs_dist_0["out_by_iter"][0], self.obs_dist_1["out_by_iter"][0]])
        torch.testing.assert_close(out, self.obs_single["out_by_iter"][0], rtol=0, atol=0)
        print("_test_output PASS")

    def _test_loss(self):
        loss_0, loss_1, loss_2 = [
            o["loss_by_iter"][0]
            for o in [self.obs_single, self.obs_dist_0, self.obs_dist_1]
        ]
        torch.testing.assert_close((loss_1 + loss_2) / 2, loss_0)
        print("_test_loss PASS")

    def _test_grad_equal(self):
        model_0, model_1, model_2 = [
            o["grad_by_iter"][0]
            for o in [self.obs_single, self.obs_dist_0, self.obs_dist_1]
        ]
        try:
            for i, (grad_0, grad_1, grad_2) in enumerate(zip(model_0, model_1, model_2)):
                torch.testing.assert_close((grad_1 + grad_2) / 2, grad_0)
        except AssertionError as e:
            print(f"#{i} are not close!")
            raise e
        print("_test_grad_equal PASS")

    def test(self):
        self._test_model_init()
        self._sanity_check(torch.device("cpu"))
        self._sanity_check(torch.device("cuda:0"))
        self._test_output()
        self._test_loss()
        self._test_grad_equal()


class TestDistKFAC(TestDistGradMaker):
    def _test_grad_1st_equal(self):
        model_0, model_1, model_2 = [
            o["iter0_grad_1st"]
            for o in [self.obs_single, self.obs_dist_0, self.obs_dist_1]
        ]
        try:
            for i, (grad_0, grad_1, grad_2) in enumerate(zip(model_0, model_1, model_2)):
                torch.testing.assert_close((grad_1 + grad_2) / 2, grad_0)
        except AssertionError as e:
            print(f"_test_grad_1st_equal FAIL, #{i} are not close!")
            raise e
        print("_test_grad_1st_equal PASS")

    def _test_A_B_inv(self):
        kron_single = self.obs_single["kron"]
        kron_dist = {**self.obs_dist_0["kron"], **self.obs_dist_1["kron"]}
        torch.testing.assert_close(kron_single, kron_dist)

    def test(self):
        super().test()
        self._test_grad_1st_equal()
        self._test_A_B_inv()


def main():
    set_seed(0)
    TestDistGradMaker(
        get_vgg_tiny, "temp",
        # 2, 3, 32, 10, [[[0, 1]]], [[[0]], [[1]]],
        8, 3, 32, 10, [[[0, 1, 2, 3, 4, 5, 6, 7]]], [[[0, 1, 2, 3]], [[4, 5, 6, 7]]],
    ).test()
    config = asdl.PreconditioningConfig(
        damping=1e-3,
        curvature_upd_interval=100,
        preconditioner_upd_interval=100,
        ignore_modules=[nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm],
        # TODO: With ema_decay?
        ema_decay=-1,
    )
    TestDistKFAC(
        get_vgg_tiny, "temp",
        # 2, 3, 32, 10, [[[0, 1]]], [[[0]], [[1]]],
        8, 3, 32, 10, [[[0, 1, 2, 3, 4, 5, 6, 7]]], [[[0, 1, 2, 3]], [[4, 5, 6, 7]]],
        config, FISHER_MC
    ).test()


if __name__ == "__main__":
    main()
