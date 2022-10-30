import time

import torch
import torchvision
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode

from .visionref import mixup


class ClassificationPresetTrain:
    def __init__(
        self,
        crop_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        random_erase_prob=0.0,
    ):
        trans = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                trans.append(autoaugment.RandAugment(interpolation=interpolation))
            elif auto_augment_policy == "ta_wide":
                trans.append(autoaugment.TrivialAugmentWide(interpolation=interpolation))
            else:
                aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
                trans.append(autoaugment.AutoAugment(policy=aa_policy, interpolation=interpolation))
        trans.extend(
            [
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


class ClassificationPresetEval:
    def __init__(
        self,
        crop_size,
        resize_size=256,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
    ):

        self.transforms = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=interpolation),
                transforms.CenterCrop(crop_size),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)


def load_data(traindir, valdir, val_resize_size, val_crop_size, train_crop_size, interpolation,
              auto_augment_policy=None, random_erase_prob=0.0):

    interpolation = InterpolationMode(interpolation)

    print("Loading training data")
    st = time.time()
    dataset = torchvision.datasets.ImageFolder(
        traindir,
        ClassificationPresetTrain(
            crop_size=train_crop_size,
            interpolation=interpolation,
            auto_augment_policy=auto_augment_policy,
            random_erase_prob=random_erase_prob,
        ),
    )
    print(f"Took {time.time() - st:.2f}s")

    print("Loading validation data")
    dataset_test = torchvision.datasets.ImageFolder(
        valdir,
        # Resize, Center Crop, ToTensor, ImageNet
        ClassificationPresetEval(
            crop_size=val_crop_size, resize_size=val_resize_size, interpolation=interpolation
        ),
    )

    return dataset, dataset_test


def get_dataloader(dataset, dataset_test, mixup_alpha, cutmix_alpha, batch_size, workers):
    collate_fn = None
    num_classes = len(dataset.classes)
    mixup_transforms = []
    if mixup_alpha > 0.0:
        mixup_transforms.append(mixup.RandomMixup(num_classes, p=1.0, alpha=mixup_alpha))
    if cutmix_alpha > 0.0:
        mixup_transforms.append(mixup.RandomCutmix(num_classes, p=1.0, alpha=cutmix_alpha))
    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
        collate_fn = lambda batch: mixupcutmix(*default_collate(batch))  # noqa: E731
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_fn,
        shuffle=True,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
    )
    return data_loader, data_loader_test
