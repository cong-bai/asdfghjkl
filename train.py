import argparse
import datetime
import os
import time

import timm
import torch
import torch.utils.data
import torchvision
from torch import nn

import asdfghjkl as asdl
from asdfghjkl import FISHER_MC, FISHER_EMP
from trainutils.dataset import load_data
from trainutils.visionref import utils

# TODO: KL-clip

def train_one_epoch(model, optimizer, grad_maker, data_loader, device, epoch, mixup_fn, args):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for _, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        loss_func = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        if mixup_fn:
            image = image[:image.shape[0] // 2 * 2, :, :, :] # In mixup_fn, batch size must be even
            target = target[:target.shape[0] // 2 * 2]
            image, target = mixup_fn(image, target)
            loss_func = torch.nn.CrossEntropyLoss()

        optimizer.zero_grad()
        dummy_y = grad_maker.setup_model_call(model, image)
        grad_maker.setup_loss_call(loss_func, dummy_y, target)
        if isinstance(grad_maker, asdl.precondition.natural_gradient.KfacGradientMaker):
            output, loss = grad_maker.forward_and_backward(accumulate=True)
        else:
            output, loss = grad_maker.forward_and_backward()

        if args.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        optimizer.step()

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))


def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")

    dataset, dataset_test = load_data(args.dataset, train_dir, val_dir, args.val_input_size, args.val_crop_pct, args.train_input_size, args.interpolation, args.auto_augment, args.random_erase)
    num_classes = len(dataset.classes)
    persist_dataset = args.dataset == "cifar10"
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=True,
        persistent_workers=persist_dataset,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=persist_dataset,
    )
    mixup_fn = None
    if args.mixup_alpha > 0 or args.cutmix_alpha > 0:
        mixup_fn = timm.data.Mixup(args.mixup_alpha, args.cutmix_alpha, label_smoothing=args.label_smoothing, num_classes=num_classes)

    print("Creating model")
    if args.model.startswith("timm_"):
        model = timm.create_model(
            args.model.replace("timm_", ""), pretrained=args.pretrained, num_classes=num_classes
        )
    else:
        model = torchvision.models.__dict__[args.model](
            pretrained=args.pretrained, num_classes=num_classes
        )
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    if args.norm_weight_decay is None:
        parameters = model.parameters()
    else:
        param_groups = torchvision.ops._utils.split_normalization_params(model)
        wd_groups = [args.norm_weight_decay, args.weight_decay]
        parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]

    opt_name = args.opt.lower()
    if opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    elif opt_name in ("sgd", "kfac_mc", "kfac_emp"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )

    if opt_name == "kfac_mc":
        config = asdl.NaturalGradientConfig(data_size=args.batch_size,
                                            fisher_type=FISHER_MC,
                                            damping=args.damping,
                                            curvature_upd_interval=args.cov_update_freq,
                                            preconditioner_upd_interval=args.inv_update_freq,
                                            ignore_modules=[nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm],
                                            ema_decay=args.ema_decay)
        grad_maker = asdl.KfacGradientMaker(model, config, swift=False)
    elif opt_name == "kfac_emp":
        config = asdl.NaturalGradientConfig(data_size=args.batch_size,
                                            fisher_type=FISHER_EMP,
                                            damping=args.damping,
                                            curvature_upd_interval=args.cov_update_freq,
                                            preconditioner_upd_interval=args.inv_update_freq,
                                            ignore_modules=[nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm],
                                            ema_decay=args.ema_decay)
        grad_maker = asdl.KfacGradientMaker(model, config, swift=False)
    else:
        grad_maker = asdl.GradientMaker(model)

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "multisteplr":
        main_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_decay_epoch, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, MultiStepLR, "
            "CosineAnnealingLR and ExponentialLR are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        evaluate(model, criterion, data_loader_test, device=device)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.epochs):
        train_one_epoch(model, optimizer, grad_maker, data_loader, device, epoch, mixup_fn, args)
        lr_scheduler.step()
        evaluate(model, criterion, data_loader_test, device=device)
        if args.output_dir:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            # utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            # utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser():

    parser = argparse.ArgumentParser(description="PyTorch Classification Training")

    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--data-path", default="/root/autodl-tmp/imagenet", type=str)
    # To use a timm model, add "timm_" before the timm model name, e.g. timm_deit_tiny_patch16_224
    parser.add_argument("--model", default="timm_deit_tiny_patch16_224", type=str)
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")

    parser.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", default=768, type=int)
    parser.add_argument("--test-only", dest="test_only", action="store_true")
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--workers", default=6, type=int)
    parser.add_argument("--print-freq", default=50, type=int)
    parser.add_argument("--output-dir", default=".", type=str)

    parser.add_argument("--opt", default="sgd", type=str, choices=["sgd", "rmsprop", "adamw", "kfac_mc", "kfac_emp"])
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--momentum", default=0, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--norm-weight-decay", default=None, type=float) # WD For norm layers
    parser.add_argument("--lr-scheduler", default="cosineannealinglr", type=str)
    parser.add_argument("--lr-warmup-epochs", default=5, type=int)
    parser.add_argument("--lr-warmup-method", default="linear", type=str)
    parser.add_argument("--lr-warmup-decay", default=0.1, type=float) # First epoch lr decay
    parser.add_argument("--lr-step-size", default=None, type=int)
    parser.add_argument('--lr-decay-epoch', nargs='+', type=int, default=[15, 25, 30])
    parser.add_argument("--lr-gamma", default=None, type=float)
    # K-FAC
    parser.add_argument('--cov-update-freq', type=int, default=10)
    parser.add_argument('--inv-update-freq', type=int, default=100)
    parser.add_argument('--ema-decay', type=float, default=0.05)
    parser.add_argument('--damping', type=float, default=0.001)
    parser.add_argument('--kl-clip', type=float, default=0.001)

    parser.add_argument("--interpolation", default="bilinear", type=str)
    parser.add_argument("--val-input-size", default=224, type=int)
    parser.add_argument("--val-crop-pct", default=1, type=int)
    parser.add_argument("--train-input-size", default=224, type=int)
    parser.add_argument("--auto-augment", default="rand-m9-mstd0.5-inc1", type=str)
    parser.add_argument("--random-erase", default=0.0, type=float)
    parser.add_argument("--mixup-alpha", default=0.8, type=float)
    parser.add_argument("--cutmix-alpha", default=1.0, type=float)
    # For DeiT-Ti/DeiT-S, the repeated augment doesn't really matter
    # see: https://github.com/facebookresearch/deit/issues/89

    parser.add_argument("--label-smoothing", default=0.1, type=float)
    parser.add_argument("--use-deterministic-algorithms", action="store_true")
    parser.add_argument("--clip-grad-norm", default=None, type=float)

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
