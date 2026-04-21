#!/usr/bin/env python3
"""Export a YOLOv9 checkpoint to an ONNX file that Frigate `yolo-generic`
can consume directly.

Why this exists
---------------
The default export pipeline used for the EdgeTPU build produces 2 raw head
outputs (`boxes_dfl_logit`, `class_scores_logit`). Frigate's ONNX detector
(`frigate/detectors/plugins/onnx.py` + `frigate/util/model.py::post_process_yolo`)
expects a **single** prediction tensor shaped `(1, 4+NC, N)` (or the transposed
`(1, N, 4+NC)`) with:
  - row 0..3 = box xywh in *pixel* space
  - row 4..4+NC-1 = per-class probabilities (post-sigmoid)

This script wraps the YOLOv9 model with a small decoder module that:
  1. Runs the backbone + DualDDetectEdgeTPU head to get DFL + class logits.
  2. Softmax + DFL bin weighting to get ltrb distances.
  3. Adds anchors/strides and converts xyxy -> xywh in pixel coords.
  4. Applies sigmoid on class logits.
  5. Concatenates to (1, 4+NC, N) as a single ONNX output `output0`.

Expected Frigate config (GPU / ONNX path):
    model:
      model_type: yolo-generic
      width: 320
      height: 320
      input_tensor: nchw
      input_dtype: float
      path: /config/model_cache/<stem>_frigate.onnx

Usage
-----
  conda activate cv-base
  python scripts/export_onnx_frigate.py \
      --weights /home/nqhia/work/BoxAI/box/result/phase2/best.pt \
      --yolov9 /home/nqhia/work/BoxAI/box/yolov9 \
      --cfg yolov9/models/detect/yolov9-s-relu6.yaml \
      --imgsz 320 --nc 3 \
      --out config/model_cache/best_320_frigate.onnx
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path


def _build_frigate_decoder(nc: int, reg_max: int, width: int, height: int):
    """Return an `nn.Module` that wraps a YOLOv9 export-mode model and
    produces a single `(1, 4+nc, N)` tensor compatible with Frigate.
    """
    import torch
    from torch import nn

    class FrigateDecoder(nn.Module):
        def __init__(self, base: nn.Module):
            super().__init__()
            self.base = base
            self.reg_max = reg_max
            self.width = width
            self.height = height

            all_anchors = []
            all_strides = []
            for stride in (8, 16, 32):
                fh, fw = height // stride, width // stride
                gy, gx = torch.meshgrid(
                    torch.arange(fh, dtype=torch.float32),
                    torch.arange(fw, dtype=torch.float32),
                    indexing="ij",
                )
                pts = torch.stack([gx.flatten(), gy.flatten()], dim=1) + 0.5
                all_anchors.append(pts)
                all_strides.append(torch.full((fh * fw, 1), stride, dtype=torch.float32))

            self.register_buffer("anchors", torch.cat(all_anchors, dim=0))
            self.register_buffer("strides_t", torch.cat(all_strides, dim=0))
            self.register_buffer("project", torch.arange(reg_max, dtype=torch.float32))

        def _split_outputs(self, y):
            """Normalize base(x) output to (boxes_dfl_logit, class_scores_logit)."""
            if isinstance(y, (list, tuple)):
                if len(y) >= 2:
                    box, cls = y[0], y[1]
                    if box.shape[-1] != self.reg_max * 4 and cls.shape[-1] != self.reg_max * 4:
                        raise RuntimeError(
                            f"Unexpected head outputs: shapes {[tuple(t.shape) for t in y]}"
                        )
                    if cls.shape[-1] == self.reg_max * 4:
                        box, cls = cls, box
                    return box, cls
            raise RuntimeError("YOLOv9 export head should return (boxes_dfl, class_scores)")

        def forward(self, x):
            y = self.base(x)
            box_dfl, class_logit = self._split_outputs(y)

            B, N, _ = box_dfl.shape
            dfl = box_dfl.reshape(B, N, 4, self.reg_max)
            dfl = torch.softmax(dfl, dim=-1)
            distances = (dfl * self.project).sum(dim=-1)

            anchors = self.anchors.unsqueeze(0)
            strides = self.strides_t.unsqueeze(0)

            x1y1 = (anchors - distances[..., :2]) * strides
            x2y2 = (anchors + distances[..., 2:4]) * strides

            cxcy = (x1y1 + x2y2) * 0.5
            wh = x2y2 - x1y1
            boxes_xywh = torch.cat([cxcy, wh], dim=-1)

            scores = torch.sigmoid(class_logit)

            predictions = torch.cat([boxes_xywh, scores], dim=-1)
            predictions = predictions.transpose(1, 2).contiguous()
            return predictions

    return FrigateDecoder


def best_onnx_opset() -> int:
    import onnx
    import torch

    v = ".".join(torch.__version__.split(".")[:2])
    mapping = {
        "2.0": 17, "2.1": 17, "2.2": 17, "2.3": 17,
        "2.4": 20, "2.5": 20, "2.6": 20, "2.7": 20, "2.8": 20,
    }
    opset = mapping.get(v, 17)
    return min(opset, onnx.defs.onnx_opset_version())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--yolov9", type=Path, required=True, help="Path to WongKinYiu/yolov9 repo root")
    ap.add_argument("--cfg", type=Path, required=True, help="e.g. yolov9/models/detect/yolov9-s-relu6.yaml")
    ap.add_argument("--out", type=Path, required=True, help="Output .onnx path")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--nc", type=int, default=3)
    ap.add_argument("--reg-max", type=int, default=16)
    ap.add_argument("--simplify", action="store_true")
    args = ap.parse_args()

    yolov9_root = args.yolov9.resolve()
    if not yolov9_root.is_dir():
        sys.exit(f"yolov9 repo not found: {yolov9_root}")
    sys.path.insert(0, str(yolov9_root))

    cfg_path = args.cfg
    if not cfg_path.is_absolute():
        c1 = (Path.cwd() / cfg_path).resolve()
        c2 = (yolov9_root / cfg_path).resolve()
        cfg_path = c1 if c1.is_file() else c2
    if not cfg_path.is_file():
        sys.exit(f"Config yaml not found: {args.cfg}")

    weights = args.weights.resolve()
    if not weights.is_file():
        sys.exit(f"Weights not found: {weights}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    import onnx
    import torch
    from models.yolo import Model

    print(f"[1/4] Build base model from {cfg_path}")
    model = Model(str(cfg_path), ch=3, nc=args.nc)

    print(f"[2/4] Load weights from {weights}")
    ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)

    def _sd(entry):
        if entry is None:
            return None
        return entry.float().state_dict() if hasattr(entry, "float") else entry

    sd = _sd(ckpt.get("ema")) or _sd(ckpt.get("model"))
    if sd is None:
        sys.exit("Checkpoint has no 'model' or 'ema' state_dict")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  loaded (missing={len(missing)}, unexpected={len(unexpected)})")

    model.fuse()
    model.eval()

    head = model.model[-1]
    if not hasattr(head, "export"):
        sys.exit("Last layer has no .export attribute; not DualDDetectEdgeTPU?")
    head.export = True

    Decoder = _build_frigate_decoder(
        nc=args.nc, reg_max=args.reg_max, width=args.imgsz, height=args.imgsz
    )
    wrapper = Decoder(model).eval()

    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz, dtype=torch.float32)
    with torch.no_grad():
        out = wrapper(dummy)
    expected_channels = 4 + args.nc
    if out.shape[1] != expected_channels:
        sys.exit(
            f"Unexpected decoder output channel dim: got {tuple(out.shape)}, "
            f"expected (1, {expected_channels}, N)"
        )
    print(f"  decoder dry-run OK; output shape {tuple(out.shape)}")

    opset = best_onnx_opset()
    export_kw = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kw["dynamo"] = False

    print(f"[3/4] Export ONNX -> {args.out} (opset={opset})")
    torch.onnx.export(
        wrapper,
        dummy,
        str(args.out),
        input_names=["images"],
        output_names=["output0"],
        opset_version=opset,
        dynamic_axes=None,
        **export_kw,
    )

    onnx_model = onnx.load(str(args.out))
    onnx.checker.check_model(onnx_model)
    print(f"  ONNX validated ({Path(args.out).stat().st_size / 1024 / 1024:.2f} MB)")

    if args.simplify:
        print("[4/4] Simplify (onnxslim)")
        try:
            import onnxslim
            onnx.save(onnxslim.slim(onnx.load(str(args.out))), str(args.out))
        except Exception as e:
            print(f"  [warn] onnxslim failed, keeping un-slimmed model: {e}")

    print("\nDone. Configure Frigate with:")
    print("  model_type: yolo-generic")
    print(f"  width: {args.imgsz}")
    print(f"  height: {args.imgsz}")
    print("  input_tensor: nchw")
    print("  input_dtype: float")
    print(f"  path: /config/model_cache/{args.out.name}")


if __name__ == "__main__":
    main()
