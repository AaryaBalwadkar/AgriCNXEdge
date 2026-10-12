"""
phase4_export.py
================
Step 1 of the on-device plan: re-export the trained student WITH the
per-apple fruit path so all four tasks can run on the phone.

Old export (adc_student.onnx):  image -> leaf, pest, raw_det.
New export (adc_student_full.onnx):
    (image [B,3,512,512], boxes [N,4] cxcywh-normalised,
     box_idx [N] batch index)
    -> leaf_logits [B,4], pest_logits [B,4], raw_det [B,6,32,32],
       fruit_logits [M,3]   (M == N valid rows)

Phone flow becomes two calls sharing one session:
  call 1 (image)                     -> leaf, pest, raw_det
  CPU: decode raw_det (conf 0.3, NMS IoU 0.45) -> boxes
  call 2 (image, boxes, box_idx)     -> fruit_logits

USAGE (laptop CPU, no GPU needed):
    python phase4_export.py
    python phase4_export.py --checkpoint <path> --out <name>.onnx

NOTES:
  * The DINOv3 pretrain file is absent on this machine; a temp stub is
    used because every parameter is overwritten by best_model.pth
    (strict load verified below — any mismatch aborts loudly).
  * dynamo=False keeps the classic exporter path used by the original
    export_onnx.py (opset 17, dynamic batch + dynamic num_rois).
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Proposed Model"))

import numpy as np
import torch
import torch.nn as nn

from student_config import get_config
from student_model import ADCStudent, roi_align_crops


class FullStudent(nn.Module):
    """ADCStudent with an explicit boxes input for the fruit path."""

    def __init__(self, base: ADCStudent):
        super().__init__()
        self.m = base

    def forward(self, x, boxes, box_idx):
        mc = self.m.cfg.model
        feats = self.m.backbone(x)
        p3 = self.m.neck(feats["f3"], feats["f4"])
        leaf = self.m.leaf_head(feats["f1"])
        pest = self.m.pest_head(feats["f2"])
        raw = self.m.det_head(p3)
        crops = roi_align_crops(p3, boxes, box_idx,
                                output_size=mc.roi_output_size,
                                spatial_scale=mc.roi_spatial_scale)
        fruit = self.m.fruit_head(crops)
        return leaf, pest, raw, fruit


def main():
    ap = argparse.ArgumentParser(description="Full-model ONNX export (CPU)")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default="adc_student_full.onnx")
    args = ap.parse_args()

    cfg = get_config()
    candidates = ([args.checkpoint] if args.checkpoint else []) + [
        str(Path(cfg.paths.checkpoint_dir) / "best_model.pth"),
        # local layout keeps student outputs under Proposed Model/student/
        str(ROOT / "Proposed Model" / "student" / "checkpoints"
            / "best_model.pth"),
    ]
    ckpt_path = next((c for c in candidates if Path(c).exists()), None)
    if ckpt_path is None:
        sys.exit(f"ABORT: no checkpoint found. tried: {candidates}")
    print(f"[1/5] checkpoint: {ckpt_path}")

    # Stub DINOv3 file (absent locally); fully overwritten by strict load.
    with tempfile.TemporaryDirectory() as td:
        stub = str(Path(td) / "dinov3_stub.pth")
        torch.save({}, stub)
        cfg.paths.dinov3_weights = stub
        print("[2/5] building ADCStudent (stub backbone, will overwrite)...")
        base = ADCStudent(cfg)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"      ckpt epoch={ckpt.get('epoch')} "
          f"best_val={ckpt.get('best_val')}")
    missing, unexpected = base.load_state_dict(ckpt["model"], strict=False)
    n_missing = len(missing)
    print(f"      strict=False load: missing={n_missing} "
          f"unexpected={len(unexpected)}")
    if unexpected:
        print("      UNEXPECTED KEYS (first 5):", unexpected[:5])
        sys.exit("ABORT: checkpoint does not match current code.")
    # Missing keys are only acceptable if they are LoRA adapters absent
    # from a frozen checkpoint (they init to identity: B=zeros).
    non_lora_missing = [k for k in missing if "lora_" not in k]
    if non_lora_missing:
        print("      NON-LoRA MISSING KEYS (first 5):",
              non_lora_missing[:5])
        sys.exit("ABORT: checkpoint is not a full student state dict.")
    if n_missing:
        print("      (missing keys are all LoRA adapters -> identity, OK)")
    base.eval()

    model = FullStudent(base).eval()
    img = torch.randn(1, 3, cfg.model.image_size, cfg.model.image_size)
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2],
                          [0.3, 0.4, 0.1, 0.15]], dtype=torch.float32)
    bidx = torch.tensor([0, 0], dtype=torch.long)

    out_path = Path(ckpt_path).parent.parent / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[3/5] exporting opset 17 -> {out_path} ...")
    with torch.no_grad():
        torch.onnx.export(
            model, (img, boxes, bidx), str(out_path),
            opset_version=17,
            input_names=["image", "boxes", "box_idx"],
            output_names=["leaf_logits", "pest_logits", "raw_det",
                          "fruit_logits"],
            dynamic_axes={
                "image": {0: "batch"},
                "boxes": {0: "num_rois"},
                "box_idx": {0: "num_rois"},
                "leaf_logits": {0: "batch"},
                "pest_logits": {0: "batch"},
                "raw_det": {0: "batch"},
                "fruit_logits": {0: "num_rois"},
            },
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"      size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")

    # ---- verification ----
    import onnx
    import onnxruntime as ort
    print("[4/5] onnx checker...")
    onnx.checker.check_model(str(out_path))
    print("      checker PASSED")
    print("[5/5] onnxruntime parity vs old export...")
    sess_new = ort.InferenceSession(str(out_path),
                                    providers=["CPUExecutionProvider"])
    print("      new inputs :",
          [(i.name, i.shape) for i in sess_new.get_inputs()])
    print("      new outputs:",
          [(o.name, o.shape) for o in sess_new.get_outputs()])

    rng = np.random.RandomState(0)
    img_np = rng.randn(1, 3, 512, 512).astype(np.float32)
    feed_new = {"image": img_np, "boxes": np.array(
        [[0.5, 0.5, 0.2, 0.2], [0.3, 0.4, 0.1, 0.15]], dtype=np.float32),
        "box_idx": np.array([0, 0], dtype=np.int64)}
    leaf_n, pest_n, raw_n, fruit_n = sess_new.run(None, feed_new)

    old_path = Path(ckpt_path).parent.parent / "adc_student.onnx"
    report = {"checkpoint": ckpt_path,
              "checkpoint_epoch": ckpt.get("epoch"),
              "out": str(out_path),
              "size_mb": round(out_path.stat().st_size / 1024 / 1024, 2),
              "fruit_shape": list(fruit_n.shape),
              "fruit_finite": bool(np.isfinite(fruit_n).all())}
    if old_path.exists():
        sess_old = ort.InferenceSession(str(old_path),
                                        providers=["CPUExecutionProvider"])
        leaf_o, pest_o, raw_o = sess_old.run(None, {"image": img_np})
        report["parity_leaf_maxabs"] = float(np.abs(leaf_o - leaf_n).max())
        report["parity_pest_maxabs"] = float(np.abs(pest_o - pest_n).max())
        report["parity_det_maxabs"] = float(np.abs(raw_o - raw_n).max())
        print(f"      parity max|diff|: leaf {report['parity_leaf_maxabs']:.2e}, "
              f"pest {report['parity_pest_maxabs']:.2e}, "
              f"det {report['parity_det_maxabs']:.2e}")
    else:
        print("      (old adc_student.onnx not found; parity skipped)")
    print(f"      fruit_logits shape {fruit_n.shape}, "
          f"finite={report['fruit_finite']}")

    # smoke test on a real orchard image (rough cell-count sanity, no NMS)
    from PIL import Image
    yp = (ROOT / "YOLO26s apple object detection model 1" / "data"
          / "yield" / "images" / "val")
    cand = sorted(yp.glob("*.jpg"))[2] if yp.exists() else None
    if cand is not None:
        im = np.array(Image.open(cand).convert("RGB").resize((512, 512)),
                      dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        im = ((im - mean) / std).transpose(2, 0, 1)[None]
        _, _, raw_real, _ = sess_new.run(
            None, {"image": im.astype(np.float32),
                   "boxes": feed_new["boxes"], "box_idx": feed_new["box_idx"]})
        sig = 1.0 / (1.0 + np.exp(-raw_real[0, :4]))
        obj = 1.0 / (1.0 + np.exp(-raw_real[0, 4]))
        cls = 1.0 / (1.0 + np.exp(-raw_real[0, 5]))
        hot = int((((obj * cls) > 0.3)).sum())
        report["smoke_image"] = cand.name
        report["smoke_hot_cells"] = hot
        print(f"      smoke {cand.name}: {hot} cells over conf 0.3 "
              f"(raw grid, pre-NMS)")

    rp = Path(ckpt_path).parent.parent / "export_report.json"
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"      report: {rp}")
    print("EXPORT COMPLETE")


if __name__ == "__main__":
    main()
