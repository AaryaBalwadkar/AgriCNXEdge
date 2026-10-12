import torch
import sys
import os
sys.path.append('e:/Projects/newmodel')
from config import Config

config = Config()
model = torch.hub.load(config.model.dinov3_repo_dir, config.model.backbone_name, source='local', weights=None)

x = torch.randn(2, 3, 224, 224)
features1 = model.get_intermediate_layers(x, n=4, return_class_token=False)
print("No reshape:", [f.shape for f in features1])

features2 = model.get_intermediate_layers(x, n=4, reshape=True, return_class_token=False)
print("With reshape:", [f.shape for f in features2])
