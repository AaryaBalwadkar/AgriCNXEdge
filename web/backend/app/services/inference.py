import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from app.core.model_metadata import FRUIT_CLASSES, LEAF_CLASSES, PEST_CLASSES
from app.services.preprocess import preprocess_image

CONF_THRESHOLD = 0.3
NMS_IOU = 0.45


class InferenceService:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.session: ort.InferenceSession | None = None
        self.input_name: str | None = None
        self.input_names: list[str] = []
        self.output_names: list[str] = []
        if self.model_path.exists():
            self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
            self.input_names = [model_input.name for model_input in self.session.get_inputs()]
            self.input_name = self.input_names[0]
            self.output_names = [output.name for output in self.session.get_outputs()]

    @property
    def loaded(self) -> bool:
        return self.session is not None and self.input_name is not None

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "model_path": str(self.model_path),
            "inputs": self.input_names,
            "outputs": self.output_names,
        }

    def predict(self, content: bytes, filename: str) -> dict[str, Any]:
        if not self.loaded:
            raise RuntimeError("Model artifact not found or could not be loaded.")

        image_tensor = preprocess_image(content)
        dummy_boxes = np.array([[0.5, 0.5, 0.1, 0.1]], dtype=np.float32)
        dummy_box_idx = np.array([0], dtype=np.int64)

        started = time.perf_counter()
        first_outputs = self._run(image_tensor, dummy_boxes, dummy_box_idx)
        first_by_name = self._outputs_by_name(first_outputs)

        leaf_logits = first_by_name.get("leaf_logits", first_outputs[0] if len(first_outputs) > 0 else None)
        pest_logits = first_by_name.get("pest_logits", first_outputs[1] if len(first_outputs) > 1 else None)
        raw_det = first_by_name.get("raw_det", first_outputs[2] if len(first_outputs) > 2 else None)

        boxes = self._decode_detections(raw_det)
        fruit_result = None

        if boxes:
            box_tensor = np.array([[b["cx"], b["cy"], b["w"], b["h"]] for b in boxes], dtype=np.float32)
            box_idx = np.zeros((len(boxes),), dtype=np.int64)
            second_outputs = self._run(image_tensor, box_tensor, box_idx)
            second_by_name = self._outputs_by_name(second_outputs)
            fruit_logits = second_by_name.get("fruit_logits")
            fruit_result = self._fruit_result(fruit_logits, boxes)

        latency_ms = (time.perf_counter() - started) * 1000

        return {
            "filename": filename,
            "latency_ms": round(latency_ms, 2),
            "leaf": self._class_result(leaf_logits, LEAF_CLASSES),
            "pest": self._class_result(pest_logits, PEST_CLASSES),
            "fruit": fruit_result,
            "yield_detection": {
                "apple_count": len(boxes),
                "boxes": boxes,
                "tensor_shape": list(np.asarray(raw_det).shape) if raw_det is not None else [],
            },
            "raw_output_names": self.output_names,
        }

    def _run(self, image: np.ndarray, boxes: np.ndarray, box_idx: np.ndarray) -> list[np.ndarray]:
        feed: dict[str, np.ndarray] = {}
        for name in self.input_names:
            if name == "image":
                feed[name] = image
            elif name == "boxes":
                feed[name] = boxes
            elif name == "box_idx":
                feed[name] = box_idx
        return self.session.run(None, feed)

    def _outputs_by_name(self, outputs: list[np.ndarray]) -> dict[str, np.ndarray]:
        return {name: outputs[index] for index, name in enumerate(self.output_names) if index < len(outputs)}

    def _class_result(self, logits: np.ndarray | None, labels: list[str]) -> dict[str, Any] | None:
        if logits is None:
            return None
        values = np.asarray(logits).reshape(-1)
        class_count = min(values.size, len(labels))
        if class_count == 0:
            return None
        probs = self._softmax(values[:class_count])
        best = int(np.argmax(probs))
        return {
            "label": labels[best],
            "confidence": round(float(probs[best]), 4),
            "scores": {labels[i]: round(float(probs[i]), 4) for i in range(class_count)},
        }

    def _fruit_result(self, fruit_logits: np.ndarray | None, boxes: list[dict[str, float]]) -> dict[str, Any] | None:
        if fruit_logits is None:
            return None
        logits = np.asarray(fruit_logits)
        if logits.size == 0:
            return None
        rows = logits.reshape(-1, len(FRUIT_CLASSES))
        predictions = []
        votes: dict[str, int] = {}
        for index, row in enumerate(rows[: len(boxes)]):
            probs = self._softmax(row)
            best = int(np.argmax(probs))
            label = FRUIT_CLASSES[best]
            votes[label] = votes.get(label, 0) + 1
            predictions.append({
                "box": boxes[index],
                "label": label,
                "confidence": round(float(probs[best]), 4),
            })
        majority = max(votes, key=votes.get) if votes else None
        return {
            "label": majority,
            "apple_predictions": predictions,
        }

    def _decode_detections(self, raw_det: np.ndarray | None) -> list[dict[str, float]]:
        if raw_det is None:
            return []
        raw = np.asarray(raw_det)
        if raw.ndim != 4 or raw.shape[1] < 6:
            return []

        grid = raw[0]
        candidates = []
        height = grid.shape[1]
        width = grid.shape[2]
        for y in range(height):
            for x in range(width):
                cx = float(self._sigmoid(grid[0, y, x]))
                cy = float(self._sigmoid(grid[1, y, x]))
                w = float(self._sigmoid(grid[2, y, x]))
                h = float(self._sigmoid(grid[3, y, x]))
                score = float(self._sigmoid(grid[4, y, x]) * self._sigmoid(grid[5, y, x]))
                if score > CONF_THRESHOLD:
                    candidates.append({"cx": cx, "cy": cy, "w": w, "h": h, "confidence": score})

        candidates.sort(key=lambda box: box["confidence"], reverse=True)
        kept = []
        for candidate in candidates:
            if all(self._iou(candidate, box) <= NMS_IOU for box in kept):
                kept.append(candidate)

        return [
            {key: round(float(value), 4) for key, value in box.items()}
            for box in kept[:100]
        ]

    def _iou(self, a: dict[str, float], b: dict[str, float]) -> float:
        ax1 = a["cx"] - a["w"] / 2
        ay1 = a["cy"] - a["h"] / 2
        ax2 = a["cx"] + a["w"] / 2
        ay2 = a["cy"] + a["h"] / 2
        bx1 = b["cx"] - b["w"] / 2
        by1 = b["cy"] - b["h"] / 2
        bx2 = b["cx"] + b["w"] / 2
        by2 = b["cy"] + b["h"] / 2
        inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
        intersection = inter_w * inter_h
        union = a["w"] * a["h"] + b["w"] * b["h"] - intersection
        return 0.0 if union <= 0 else intersection / union

    def _softmax(self, values: np.ndarray) -> np.ndarray:
        shifted = values - np.max(values)
        exp = np.exp(shifted)
        return exp / np.sum(exp)

    def _sigmoid(self, values: np.ndarray | float) -> np.ndarray | float:
        return 1.0 / (1.0 + np.exp(-values))
