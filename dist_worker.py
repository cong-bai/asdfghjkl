import torch
import torch.distributed as dist
from torch import nn, optim

import asdl
from asdl_mod import KfacGradientMakerForTest
from test_utils.dist import sync_params_and_buffers
from test_utils.utils import set_seed


def train(model, optimizer, grad_maker, images_list, targets_list, gpu):
    observe_dict = {}
    model.train()
    grad_by_iter = []
    loss_by_iter = []
    output_by_iter = []
    for i, (images, targets) in enumerate(zip(images_list, targets_list)):
        images, targets = images.cuda(gpu), targets.cuda(gpu)
        # TODO: This should be fixed in later asdl update
        loss_func = nn.CrossEntropyLoss()

        optimizer.zero_grad()
        dummy_y = grad_maker.setup_model_call(model, images)
        grad_maker.setup_loss_call(loss_func, dummy_y, targets)
        outputs, loss = grad_maker.forward_and_backward()
        grad_by_iter.append([param.grad.detach().cpu() for param in model.parameters()])
        loss_by_iter.append(loss.detach().cpu())
        output_by_iter.append(outputs.detach().cpu())
        nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

    observe_dict["grad_by_iter"] = grad_by_iter
    observe_dict["loss_by_iter"] = loss_by_iter
    observe_dict["out_by_iter"] = output_by_iter
    return observe_dict


def dist_worker(
    gpu, model_func, world_size, images, targets, data_split, lr, momentum,
    weight_decay, dump_prefix, gradmaker_config=None, fisher_type=None,
):
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    
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
    observe_dict["init_model_before"] = {
        k: v.cpu() for i, (k, v) in enumerate(model.state_dict().items())
    }
    if world_size > 1:
        sync_params_and_buffers(model)
        observe_dict["init_model_after"] = {
            k: v.cpu() for i, (k, v) in enumerate(model.state_dict().items())
        }

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay, nesterov=True)

    if gradmaker_config is not None:
        gradmaker_config.data_size = len(data_split[gpu][0])
        for batch in data_split[gpu]:
            assert len(batch) == batch_size
        grad_maker = KfacGradientMakerForTest(
            model, gradmaker_config, fisher_type=fisher_type, swift=False
        )
    else:
        grad_maker = asdl.GradientMaker(model)

    observe_dict = {
        **observe_dict,
        **train(model, optimizer, grad_maker, images_list, targets_list, gpu),
    }
    if hasattr(grad_maker, "obs_dict"):
        observe_dict = {**observe_dict, **grad_maker.obs_dict}
    torch.save(observe_dict, dump_prefix + f"_{gpu}.pt")
    dist.destroy_process_group()
