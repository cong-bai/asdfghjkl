import random

import numpy as np
import torch


def get_attr(target_list, key):
    return [target[key] for target in target_list]


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_batched_data(data_size, img_channel, img_size, cls_num):
    images = torch.randn((data_size, img_channel, img_size, img_size), dtype=torch.float32)
    targets = torch.randint(cls_num, (data_size,))
    # targets = torch.tensor([7, 7, 7, 7, 0, 0, 0, 0])
    return images, targets
