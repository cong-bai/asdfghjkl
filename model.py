import torch
from torch import nn


def make_layers(cfg, batch_norm):
    layers = []
    in_channels = 3
    stride = 1
    for v in cfg:
        if v == "M":
            stride = 2
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=stride)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
            stride = 1
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, features, dropout):
        super().__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 10),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_vgg_tiny(batch_norm=False, dropout=0):
    features = make_layers(["M", 16, "M", 32, 64, "M", 64], batch_norm=batch_norm)
    return VGG(features, dropout)
