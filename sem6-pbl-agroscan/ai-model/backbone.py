import math
import torch
import torch.nn as nn

class DINOv3ConvNeXtBackbone(nn.Module):

    def __init__(self, model_config):
        super().__init__()

        if model_config.pretrained_backbone:
            print(f"Loading TRUE DINOv3 from local repo: {model_config.dinov3_repo_dir}...")
            # Using the exact syntax you found in the README
            self.backbone = torch.hub.load(
                model_config.dinov3_repo_dir, 
                model_config.backbone_name, 
                source='local', 
                weights=model_config.dinov3_weights
            )
        else:
            self.backbone = torch.hub.load(
                model_config.dinov3_repo_dir, 
                model_config.backbone_name, 
                source='local', 
                weights=None
            )

        # Freeze the backbone to protect T4 VRAM
        for param in self.parameters():
            param.requires_grad = False
            
        # EXPERT FIX: Force eval mode immediately to freeze normalization stats
        self.backbone.eval()

    def train(self, mode=True):
        """
        Override the default train() method.
        Ensures the frozen backbone stays in eval() mode so normalization 
        layers don't update their running statistics.
        """
        super().train(mode)
        self.backbone.eval()
        return self
        
    def forward(self, x):
        features = self.backbone.get_intermediate_layers(x, n=4, reshape=True, return_class_token=False)

        reshaped_features = {}
        keys = ["f1", "f2", "f3", "f4"]
        
        for key, feat in zip(keys, features):
            # If DINO returned a flattened [B, N, C] tensor, rebuild the spatial grid
            if feat.dim() == 3:
                B, N, C = feat.shape
                H = int(math.sqrt(N)) # N is H*W. Since images are square, H = sqrt(N)
                
                # Reshape to [B, H, W, C] and then permute to PyTorch standard [B, C, H, W]
                feat = feat.view(B, H, H, C).permute(0, 3, 1, 2)
            
            reshaped_features[key] = feat
        
        return reshaped_features
