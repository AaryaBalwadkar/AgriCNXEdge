import torch.nn as nn
from backbone import DINOv3ConvNeXtBackbone
from heads import MultiTaskHeads


class AgriculturalModel(nn.Module):

    def __init__(self, model_config):
        super().__init__()
        self.backbone = DINOv3ConvNeXtBackbone(model_config)
        self.heads    = MultiTaskHeads(model_config)

    def forward(self, x):
        features = self.backbone(x)
        return self.heads(features)
