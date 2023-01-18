import torch

from train_dist import get_args_parser
from train_dist import main as main_worker

def main():
    torch.multiprocessing.spawn(single_launcher, nprocs=torch.cuda.device_count())


def single_launcher(gpu_index):
    args = get_args_parser().parse_args()
    args.world_size = torch.cuda.device_count()
    args.rank = gpu_index
    main_worker(args)


if __name__ == "__main__":
    main()
