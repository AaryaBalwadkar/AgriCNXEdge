import os
import torch
import numpy as np
import cv2
import csv
import json
from torch.utils.data import Dataset
from config import LEAF_CLASS_MAP, FRUIT_CLASS_MAP, PEST_CLASS_MAP


class AgriculturalDataset(Dataset):

    def __init__(self, config, split_file_path, transforms=None):
        self.config = config
        self.transforms = transforms
        self.max_boxes = config.data.max_boxes
        # Use stride 4 (f1 resolution) for the heatmap
        self.output_res = config.data.image_size // 4 

        with open(split_file_path, "r", encoding="utf-8") as f:
            self.image_ids = [line.strip() for line in f if line.strip()]

        self.gt_leaf_dict, self.gt_fruit_dict, self.gt_pest_dict = {}, {}, {}
        self.gt_yield_boxes_dict, self.gt_yield_count_dict = {}, {}
        self.t_leaf_dict, self.t_fruit_dict, self.t_pest_dict, self.t_yield_dict = {}, {}, {}, {}

        # Canonical class maps
        self.leaf_class_map, self.fruit_class_map, self.pest_class_map = LEAF_CLASS_MAP, FRUIT_CLASS_MAP, PEST_CLASS_MAP

        # --- Re-integrated Label Loading Logic ---
        raw_gt_leaf  = self._read_csv(os.path.join(config.data.labels_dir, "leaf_labels.csv"))
        raw_gt_fruit = self._read_csv(os.path.join(config.data.labels_dir, "fruit_labels.csv"))
        raw_gt_pest  = self._read_csv(os.path.join(config.data.labels_dir, "pest_labels.csv"))

        for row in raw_gt_leaf:
            self.gt_leaf_dict[row["image_id"]] = self.leaf_class_map.get(row["class_name"], -1)
        for row in raw_gt_fruit:
            self.gt_fruit_dict[row["image_id"]] = self.fruit_class_map.get(row["class_name"], -1)
        for row in raw_gt_pest:
            self.gt_pest_dict[row["image_id"]] = self.pest_class_map.get(row["class_name"], -1)

        for row in self._read_csv(os.path.join(config.data.labels_dir, "yield_boxes.csv")):
            iid = row["image_id"]
            try:
                parts = [float(x) for x in row["yolo_annotation"].split()]
                if len(parts) == 5:
                    self.gt_yield_boxes_dict.setdefault(iid, []).append(parts[1:])
            except Exception: pass

        for row in self._read_csv(os.path.join(config.data.teacher_csv_dir, "leaf_soft.csv")):
            try: self.t_leaf_dict[row["image_id"]] = json.loads(row["leaf_logits"])
            except Exception: pass

        # ... (Repeat for fruit_soft.csv and pest_soft.csv as per your original file)

        for row in self._read_csv(os.path.join(config.data.teacher_csv_dir, "yield_soft.csv")):
            try:
                boxes_data = json.loads(row["json_boxes"])
                box_list = [b["yolo"] for b in boxes_data if "yolo" in b]
                self.t_yield_dict[row["image_id"]] = {"boxes": box_list}
            except Exception: pass

    @staticmethod
    def _read_csv(path):
        if not os.path.exists(path): return []
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _draw_gaussian(self, heatmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = self._gaussian2d((diameter, diameter), sigma=diameter / 6)
        x, y = int(center[0]), int(center[1])
        height, width = heatmap.shape[0:2]
        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)
        masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    def _gaussian2d(self, shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        return torch.from_numpy(h).float()

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.image_ids)

    def _clean_boxes(self, boxes):
        """Clips YOLO boxes strictly to [0.0, 1.0] to prevent floating-point crashes."""
        cleaned = []
        for b in boxes:
            cx, cy, w, h = b[0], b[1], b[2], b[3]
            # Convert center-format to edge-format
            x1, y1 = cx - w/2, cy - h/2
            x2, y2 = cx + w/2, cy + h/2
            
            # Force coordinates strictly into the [0.0, 1.0] boundary
            x1 = np.clip(x1, 0.0, 1.0)
            y1 = np.clip(y1, 0.0, 1.0)
            x2 = np.clip(x2, 0.0, 1.0)
            y2 = np.clip(y2, 0.0, 1.0)
            
            new_w = x2 - x1
            new_h = y2 - y1
            
            # Only keep valid boxes (removes boxes that got clipped to nothing)
            if new_w > 0.001 and new_h > 0.001:
                new_cx = x1 + new_w / 2
                new_cy = y1 + new_h / 2
                cleaned.append([new_cx, new_cy, new_w, new_h])
        return cleaned
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.config.data.images_dir, image_id)
        
        bgr = cv2.imread(img_path)
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else np.zeros((self.config.data.image_size, self.config.data.image_size, 3), dtype=np.uint8)

        # EXPERT FIX: Clean boxes to prevent Albumentations -0.0000007 crashes
        raw_gt = self.gt_yield_boxes_dict.get(image_id, [])
        raw_t  = self.t_yield_dict.get(image_id, {"boxes": []})["boxes"]
        
        gt_boxes = self._clean_boxes(raw_gt)
        t_boxes  = self._clean_boxes(raw_t)

        # Safely Transform BOTH Ground Truth and Teacher Boxes together
        if self.transforms:
            all_boxes = gt_boxes + t_boxes
            # Label 0 for GT, Label 1 for Teacher (tracks them through crops/flips)
            labels = [0] * len(gt_boxes) + [1] * len(t_boxes)

            aug = self.transforms(image=image, bboxes=all_boxes, class_labels=labels)
            image = aug["image"]
            transformed_boxes = aug["bboxes"]
            transformed_labels = aug["class_labels"]

            # Separate them back out
            gt_boxes = [b for b, l in zip(transformed_boxes, transformed_labels) if l == 0]
            t_boxes  = [b for b, l in zip(transformed_boxes, transformed_labels) if l == 1]

        # Prepare Ground Truth Center-Point Maps
        hm = torch.zeros((1, self.output_res, self.output_res))
        wh = torch.zeros((2, self.output_res, self.output_res))
        off = torch.zeros((2, self.output_res, self.output_res))
        reg_mask = torch.zeros((1, self.output_res, self.output_res))

        for box in gt_boxes:
            ctx = np.clip(box[0] * self.output_res, 0, self.output_res - 1)
            cty = np.clip(box[1] * self.output_res, 0, self.output_res - 1)
            bw, bh = box[2] * self.output_res, box[3] * self.output_res
            ct_int = [int(ctx), int(cty)]
            radius = max(0, int((min(bw, bh) * 0.5)))
            self._draw_gaussian(hm[0], ct_int, radius)
            wh[:, ct_int[1], ct_int[0]] = torch.tensor([bw, bh])
            off[:, ct_int[1], ct_int[0]] = torch.tensor([ctx - ct_int[0], cty - ct_int[1]])
            reg_mask[:, ct_int[1], ct_int[0]] = 1

        # Teacher Distillation Heatmap
        t_hm = torch.zeros((1, self.output_res, self.output_res))
        for t_box in t_boxes:
            t_ctx = np.clip(t_box[0] * self.output_res, 0, self.output_res - 1)
            t_cty = np.clip(t_box[1] * self.output_res, 0, self.output_res - 1)
            self._draw_gaussian(t_hm[0], [int(t_ctx), int(t_cty)], radius=2)

        return {
            "image": image if isinstance(image, torch.Tensor) else torch.from_numpy(image).permute(2,0,1).float().div(255),
            "gt_leaf": torch.tensor(self.gt_leaf_dict.get(image_id, -1), dtype=torch.long),
            "gt_fruit": torch.tensor(self.gt_fruit_dict.get(image_id, -1), dtype=torch.long),
            "gt_pest": torch.tensor(self.gt_pest_dict.get(image_id, -1), dtype=torch.long),
            "yield_gt": {"heatmap": hm, "wh": wh, "offset": off, "reg_mask": reg_mask},
            "t_leaf": torch.tensor(self.t_leaf_dict.get(image_id, [0]*5), dtype=torch.float32),
            "t_fruit": torch.tensor(self.t_fruit_dict.get(image_id, [0]*5), dtype=torch.float32),
            "t_pest": torch.tensor(self.t_pest_dict.get(image_id, [0]*6), dtype=torch.float32),
            "t_yield_maps": {"heatmap": t_hm}
        }