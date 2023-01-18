import argparse
import time

import timm
import torch
import torch.distributed as dist
import torchvision
from timm.data.transforms_factory import create_transform
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import asdl
from asdl import FISHER_MC, FISHER_EMP
from ddp import sync_params_and_buffers


def get_grad_vec(model):
    grad_list = [
        torch.clone(p.grad.reshape(-1)).detach() for p in model.parameters() if p.grad is not None
    ]
    return torch.cat(grad_list)


def train_one_epoch(model, optimizer, grad_maker, data_loader, print_freq=10,
                    gpu=0, clip_grad_norm=0):
    model.train()
    end_time = time.time()

    for i, (image, target) in enumerate(data_loader):
        start_time = time.time()
        image, target = image.cuda(gpu), target.cuda(gpu)
        # TODO: This should be fixed in later asdl update
        loss_func = nn.CrossEntropyLoss()

        optimizer.zero_grad()
        dummy_y = grad_maker.setup_model_call(model, image)
        grad_maker.setup_loss_call(loss_func, dummy_y, target)
        output, loss = grad_maker.forward_and_backward()
        if clip_grad_norm:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

        with torch.no_grad():
            acc = torch.sum(torch.argmax(output, dim=1) == target) / len(target) * 100
        if i % print_freq == 0 and gpu == 0:
            print(f"[{i}/{len(data_loader)}]\t loss: {loss:.4f}\t acc: {acc:.3f}%\t"
                  f"time: {time.time() - end_time:.3f}\t data_time: {start_time - end_time:.3f}")

        end_time = time.time()


def evaluate(model, criterion, data_loader, gpu):
    model.eval()
    target_list = []
    output_list = []

    with torch.inference_mode():
        for image, target in data_loader:
            image = image.cuda(gpu)
            target_list.append(target.cuda(gpu))
            output_list.append(model(image))

        target = torch.cat(target_list)
        output = torch.cat(output_list)
        loss = criterion(output, target)
        acc = torch.sum(torch.argmax(output, dim=1) == target) / len(target) * 100
    return loss, acc.cpu()


def main(args):

    gpu = args.rank
    dist.init_process_group(backend="nccl", init_method=args.dist_url,
                            world_size=args.world_size, rank=gpu)
    torch.cuda.set_device(gpu)

    # Data
    train_transform = create_transform(224, is_training=True,
                                       interpolation="bilinear",
                                       auto_augment="rand-m9-mstd0.5-inc1")
    dataset = CIFAR10(root=args.data_path, transform=train_transform, download=True)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, sampler=sampler)

    if args.rank == 0:
        val_transform = create_transform(224, interpolation="bilinear", crop_pct=1)
        dataset_test = CIFAR10(root=args.data_path, train=False, transform=val_transform)
        data_loader_test = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=4)

    # Model
    if args.model.startswith("timm_"):
        model = timm.create_model(
            args.model.replace("timm_", ""), pretrained=args.pretrained, num_classes=10
        )
    else:
        model = torchvision.models.__dict__[args.model](pretrained=args.pretrained, num_classes=10)
    model.cuda(gpu)
    sync_params_and_buffers(model)
    # For a digest of the model:
    # print([p.mean().item() for p in model.parameters()])

    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay, nesterov=args.nesterov)
    if args.opt == "kfac_mc":
        config = asdl.PreconditioningConfig(
            data_size=args.batch_size / 2,
            damping=args.damping,
            curvature_upd_interval=args.cov_update_freq,
            preconditioner_upd_interval=args.inv_update_freq,
            ignore_modules=[nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm],
            ema_decay=args.ema_decay,
        )
        grad_maker = asdl.KfacGradientMaker(model, config, fisher_type=FISHER_MC, swift=False)
    else:
        grad_maker = asdl.GradientMaker(model)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, grad_maker, data_loader, args.print_freq,
                        gpu, args.clip_grad_norm)
        lr_scheduler.step()
        if args.rank == 0:
            loss, acc = evaluate(model, criterion, data_loader_test, gpu)
            print(f"Epoch {epoch} acc: {acc:.3f}\t loss: {loss:.4f}")


def get_args_parser():

    parser = argparse.ArgumentParser(description="PyTorch Classification Training")

    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--dist-url', default='tcp://localhost:36000', type=str)

    parser.add_argument("--data-path", default="data", type=str)
    # To use a timm model, add "timm_" before the timm model name, e.g. timm_deit_tiny_patch16_224
    parser.add_argument("--model", default="timm_vit_tiny_patch16_224", type=str)
    parser.add_argument("--pretrained", default=True, dest="pretrained", action="store_true")

    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--print-freq", default=50, type=int)

    parser.add_argument("--opt", default="kfac_mc", type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--momentum", default=0.75, type=float)
    parser.add_argument("--nesterov", default=True, action="store_true")
    parser.add_argument("--weight-decay", default=1e-5, type=float)
    parser.add_argument("--clip-grad-norm", default=10, type=float)
    # K-FAC
    parser.add_argument("--cov-update-freq", type=int, default=10)
    parser.add_argument("--inv-update-freq", type=int, default=100)
    parser.add_argument("--ema-decay", type=float, default=1e-4)
    parser.add_argument("--damping", type=float, default=1e-7)

    return parser


if __name__ == "__main__":
    main(get_args_parser().parse_args())
