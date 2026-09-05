"""
Model architecture for the AQI image classifier.

This is copied verbatim from the training notebook (AirQualityNet) so that
the state_dict keys line up exactly with best_model.pth. Do not rename layers
here unless you re-export the checkpoint to match.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AirQualityNet(nn.Module):
    def __init__(self, num_classes=3, dropout=0.5):
        super().__init__()

        # -------- Backbone --------
        self.stage1 = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(2),
        )

        self.stage2 = nn.Sequential(
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2),
        )

        self.stage3 = nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2),
        )

        self.stage4 = nn.Sequential(
            ConvBlock(128, 256),
            nn.MaxPool2d(2),
        )

        # -------- Head --------
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def load_model(checkpoint_path: str, num_classes: int = 3, device: str = "cpu") -> AirQualityNet:
    """Instantiate AirQualityNet and load trained weights from a checkpoint.

    Works whether the .pth file is a raw state_dict (what we found in your
    checkpoint) or a dict wrapping it under a "model" key (the training loop
    in the notebook saves both formats at different points, so this handles
    either).
    """
    model = AirQualityNet(num_classes=num_classes)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model" in checkpoint and not any(
        k.startswith(("stage", "classifier")) for k in checkpoint.keys()
    ):
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
