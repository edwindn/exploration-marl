import numpy as np
import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    """
    Canonical IMPALA residual block:
      y = x + Conv3x3(ReLU(Conv3x3(ReLU(x))))
    This matches the common reproduction of the IMPALA-CNN residual unit.
    """

    def __init__(self, channels: int, activation: str = "relu"):
        super().__init__()
        self.relu = nn.ReLU(inplace=True) if activation == "relu" else nn.Tanh()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        y = self.relu(x)
        y = self.conv1(y)
        y = self.relu(y)
        y = self.conv2(y)
        return x + y


class IMPALA(nn.Module):
    """
    IMPALA CNN encoder with canonical conv sequences.

    Each sequence: Conv3x3 -> MaxPool(stride) -> ResidualBlocks -> ReLU
    This follows the canonical IMPALA-CNN architecture while maintaining Ray RLlib compatibility.
    """
    config = {
            "input_dims": (50, 50, 3),
            "cnn_filters": [(16, 3, 1), (32, 3, 1), (32, 3, 1)],
            "cnn_activation": "relu",
            "num_res_blocks": 3,
            "feature_size": 64
    }

    def __init__(self):
        super().__init__()

        cfg = IMPALA.config
        cnn_layers = []
        in_channels = cfg["input_dims"][-1]

        for i, (out_channels, kernel_size, stride) in enumerate(cfg["cnn_filters"]):
            cnn_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1))

            if stride > 1:
                cnn_layers.append(nn.MaxPool2d(kernel_size=3, stride=stride, padding=1))

            for _ in range(cfg["num_res_blocks"]):
                cnn_layers.append(ResNetBlock(out_channels, cfg["cnn_activation"]))

            cnn_layers.append(nn.ReLU(inplace=True))

            in_channels = out_channels

        cnn_layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*cnn_layers)

        self._calculate_cnn_output_size()

        self.final_mlp = nn.Sequential(
            nn.Linear(self._cnn_output_size, cfg["feature_size"]),
            nn.ReLU() if cfg["cnn_activation"] == "relu" else nn.Tanh(),
        )

        self._output_dims = (cfg["feature_size"],)

    def _calculate_cnn_output_size(self):
        h, w, c = IMPALA.config["input_dims"]
        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            cnn_output = self.cnn(dummy_input)
            self._cnn_output_size = cnn_output.shape[1]

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)

        if x.shape[-1] <= 8:
            x = x.permute(0, 3, 1, 2)

        cnn_features = self.cnn(x)
        output_features = self.final_mlp(cnn_features)

        return output_features
