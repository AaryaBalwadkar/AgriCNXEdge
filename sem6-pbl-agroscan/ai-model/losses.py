import torch
import torch.nn as nn
import torch.nn.functional as F

# ══════════════════════════════════════════════════════════════════════════════
# 1. CENTER-POINT DETECTION LOSS (CenterNet Logic)
# ══════════════════════════════════════════════════════════════════════════════

class CenterPointLoss(nn.Module):
    """
    Implementation of the CenterNet loss suite:
    - Modified Focal Loss for heatmaps
    - L1 Loss for Width/Height and Offset
    """
    def __init__(self, lambda_wh=0.1, lambda_off=1.0):
        super().__init__()
        self.lambda_wh = lambda_wh
        self.lambda_off = lambda_off

    def _neg_loss(self, pred, gt):
        """Modified focal loss for heatmap (penalty-reduced)"""
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)
        loss = 0
        
        # Log-sigmoid for numerical stability
        pos_loss = torch.log(pred.clamp(min=1e-12)) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log((1 - pred).clamp(min=1e-12)) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def _reg_loss(self, regs, gt_regs, mask):
        """L1 loss applied only at object center indices"""
        mask = mask.expand_as(gt_regs).float()
        loss = F.l1_loss(regs * mask, gt_regs * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss

    def forward(self, pred_dict, gt_dict):
        # pred_dict: {'heatmap', 'wh', 'offset'} from heads.py
        # gt_dict:   {'heatmap', 'wh', 'offset', 'reg_mask'} from dataset.py
        
        hm_loss  = self._neg_loss(torch.sigmoid(pred_dict['heatmap']), gt_dict['heatmap'])
        wh_loss  = self._reg_loss(pred_dict['wh'], gt_dict['wh'], gt_dict['reg_mask'])
        off_loss = self._reg_loss(pred_dict['offset'], gt_dict['offset'], gt_dict['reg_mask'])

        return hm_loss, self.lambda_wh * wh_loss, self.lambda_off * off_loss

class FocalLoss(nn.Module):

    def __init__(self, gamma=2.0, ignore_index=-1):
        super().__init__()
        self.gamma        = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss    = F.cross_entropy(inputs, targets, reduction="none", ignore_index=self.ignore_index)
        pt         = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        mask       = (targets != self.ignore_index).float()
        return (focal_loss * mask).sum() / mask.sum().clamp(min=1.0)

class HomoscedasticUncertaintyWeighter(nn.Module):
    def __init__(self, task_names, min_log_var=-3.0, max_log_var=1.0):
        super().__init__()
        self.task_names = task_names
        
        # --- EXPERT FIX: Bounded Variance ---
        # max_log_var = 1.0 means the lowest the weight (precision) can ever go is exp(-1) = 0.36.
        # This allows the network to adapt dynamically, but physically prevents it 
        # from "muting" the YOLO detection head.
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var
        
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1))
            for name in task_names
        })

    def forward(self, raw_losses):
        total = torch.zeros(1, device=next(iter(raw_losses.values())).device)
        weight_info = {}
        
        for name, loss in raw_losses.items():
            # Clamp the learnable parameter before calculating precision
            clamped_log_var = torch.clamp(self.log_vars[name], min=self.min_log_var, max=self.max_log_var)
            
            precision = torch.exp(-clamped_log_var)               # 1 / (sigma^2)
            weighted = precision * loss + 0.5 * clamped_log_var   # Kendall & Gal formula
            
            total = total + weighted
            weight_info[name] = precision.item()
            
        return total.squeeze(), weight_info

class MultiTaskDistillationLoss(nn.Module):
    def __init__(self, train_config):
        super().__init__()
        self.alpha = train_config.alpha
        self.beta  = train_config.beta
        
        self.weighter = HomoscedasticUncertaintyWeighter(
            task_names=["leaf", "fruit", "pest", "yield"]
        )
        self.focal_loss = FocalLoss(ignore_index=train_config.ignore_index)
        
        # Replace YOLODecoupledLoss with CenterPointLoss
        self.center_point_loss = CenterPointLoss(lambda_wh=0.1, lambda_off=1.0)

    def _cls_loss(self, s_logits, t_logits, labels):
        # [Existing Focal + KD loss logic remains unchanged]
        T, alpha = 3.0, self.alpha # Assuming T=3.0 from prompt history
        gt_loss = self.focal_loss(s_logits, labels)
        kd_loss = F.kl_div(F.log_softmax(s_logits / T, dim=-1), 
                           F.softmax(t_logits / T, dim=-1), 
                           reduction="none").sum(dim=-1) * (T ** 2)
        mask = (labels != -1).float()
        kd_loss = (kd_loss * mask).sum() / mask.sum().clamp(min=1.0)
        return alpha * gt_loss + (1.0 - alpha) * kd_loss

    def forward(self, student_preds, teacher_preds, gt_labels):
        # 1. Classification
        l_loss = self._cls_loss(student_preds["leaf"],  teacher_preds["t_leaf"],  gt_labels["gt_leaf"])
        f_loss = self._cls_loss(student_preds["fruit"], teacher_preds["t_fruit"], gt_labels["gt_fruit"])
        p_loss = self._cls_loss(student_preds["pest"],  teacher_preds["t_pest"],  gt_labels["gt_pest"])

        # 2. Center-Point Yield Detection
        # Calculate loss against Ground Truth
        gt_hm, gt_wh, gt_off = self.center_point_loss(student_preds["yield_map"], gt_labels["yield_gt"])
        
        # Calculate Distillation Loss against Teacher maps
        # We distill the heatmap directly using MSE to match the teacher's "confidence" peaks
        t_yield = teacher_preds["t_yield_maps"] # Expected from updated dataset.py later
        kd_hm_loss = F.mse_loss(torch.sigmoid(student_preds["yield_map"]["heatmap"]), t_yield["heatmap"])
        
        yield_loss = (gt_hm + gt_wh + gt_off) + self.beta * kd_hm_loss

        # 3. Dynamic Weighting
        total_loss, weight_info = self.weighter({
            "leaf": l_loss, "fruit": f_loss, "pest": p_loss, "yield": yield_loss
        })

        return {
            "total_loss": total_loss,
            "yield_loss": yield_loss,
            "weights":    weight_info
        }