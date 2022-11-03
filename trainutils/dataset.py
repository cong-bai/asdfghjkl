import torchvision
from timm.data.transforms_factory import create_transform


def load_data(dataset, traindir, valdir, val_input_size, val_crop_pct, train_input_size, interpolation,
              auto_augment_policy, random_erase_prob, dataset_root="data"):

    train_transform = create_transform(
        train_input_size,
        is_training=True,
        auto_augment=auto_augment_policy,
        interpolation=interpolation,
        re_prob=random_erase_prob,
    )
    val_transform = create_transform(
        val_input_size,
        is_training=False,
        interpolation=interpolation,
        crop_pct=val_crop_pct
    )

    if dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(root=dataset_root, transform=train_transform, download=True)
        dataset_test = torchvision.datasets.CIFAR10(root=dataset_root, train=False, transform=val_transform)
    elif dataset == "imagenet":
        dataset = torchvision.datasets.ImageFolder(traindir, transform=train_transform)
        dataset_test = torchvision.datasets.ImageFolder(valdir, transform=val_transform)

    return dataset, dataset_test
