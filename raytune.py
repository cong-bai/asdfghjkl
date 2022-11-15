import os
import sys

from ray import tune
from ray.air.config import RunConfig
from ray.tune.search.hebo import HEBOSearch

from train import get_args_parser, main


class HiddenPrints:
    def __init__(self):
        self._original_stdout = None

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w', encoding="utf8")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def train(config):
    args, _ = get_args_parser().parse_known_args()
    args.ray = True
    args.pretrained = True
    args.epochs = 20
    args.ignore_warning = True
    args.data_path = "/home/cong/codes/asdl_new/asdfghjkl/data"
    for k, v in config.items():
        setattr(args, k, v)
    with HiddenPrints():
        main(args)

search_space = {
    "lr": tune.loguniform(3e-3, 1e-1),
    "momentum": tune.uniform(0, 0.98),
    "weight_decay": tune.loguniform(1e-6, 1e-3),
    "lr_warmup_epochs": tune.randint(0, 10),
    "clip_grad_norm": tune.loguniform(1e-1, 10),
    "lr_warmup_decay": tune.loguniform(1e-4, 1),
    "lr_eta_min": tune.loguniform(1e-4, 1),
    "batch_size": tune.choice([128, 256, 512])
}

tuner = tune.Tuner(
    tune.with_resources(train, {"gpu": 1}),
    tune_config=tune.TuneConfig(
        mode="max",
        metric="mean_accuracy",
        num_samples=384,
        time_budget_s=46800,
        search_alg=HEBOSearch(max_concurrent=8)
    ),
    # run_config=RunConfig(verbose=1),
    param_space=search_space,
)

results = tuner.fit()
