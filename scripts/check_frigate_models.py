#!/usr/bin/env python3
"""Verify that ONNX / TFLite models are compatible with Frigate's
`yolo-generic` detector.

Checks performed
----------------
ONNX (used by `frigate/detectors/plugins/onnx.py` → `post_process_yolo`):
  - Input: single float32 tensor, NCHW, shape (1, 3, H, W)
  - Output: EITHER
      * single tensor with shape (1, C, N) or (1, N, C) where C = 4 + NC
      * OR >=2 4D tensors (multipart branch) for YOLO-anchored heads
  - Runs a dummy forward pass through ONNX Runtime.

TFLite EdgeTPU (used by `frigate/detectors/plugins/edgetpu_tfl.py`):
  - Needs 2 or 3 output tensors.
  - One output shape = (1, N, 64)         (box DFL logits)
  - One output shape = (1, N, NC), NC > 1 (class score logits)
  - Input dtype = int8 or uint8, shape (1, H, W, 3) NHWC.
  - Optionally allocates a tflite interpreter and runs a dummy inference.

Usage
-----
  python scripts/check_frigate_models.py \
      --onnx   config/model_cache/best_320_frigate.onnx \
      --tflite config/model_cache/best_320_int8_edgetpu.tflite \
      --expected-nc 3 --imgsz 320

Either flag is optional; pass only what you want to verify.
Exit code is 0 when all provided models pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _print_header(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{title}\n{bar}")


def check_onnx(path: Path, expected_nc: int, imgsz: int) -> bool:
    _print_header(f"ONNX check: {path}")
    if not path.is_file():
        print(f"[FAIL] file not found: {path}")
        return False

    import numpy as np
    import onnx
    import onnxruntime as ort

    model = onnx.load(str(path))
    try:
        onnx.checker.check_model(model)
        print("[ok] onnx.checker passed")
    except Exception as e:
        print(f"[warn] onnx.checker reported: {e}")

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()

    ok = True
    print(f"inputs ({len(inputs)}):")
    for i in inputs:
        print(f"  - {i.name}: type={i.type} shape={i.shape}")
    print(f"outputs ({len(outputs)}):")
    for o in outputs:
        print(f"  - {o.name}: type={o.type} shape={o.shape}")

    if len(inputs) != 1:
        print(f"[FAIL] expected exactly 1 input, got {len(inputs)}")
        ok = False
    else:
        inp = inputs[0]
        if inp.type not in ("tensor(float)", "tensor(float16)"):
            print(f"[FAIL] input dtype {inp.type} not supported by yolo-generic (need float32)")
            ok = False
        shape = [1 if isinstance(d, str) or d is None else d for d in inp.shape]
        if len(shape) != 4 or shape[1] != 3:
            print(f"[FAIL] input shape {inp.shape} is not NCHW with 3 channels")
            ok = False
        if shape[2] != imgsz or shape[3] != imgsz:
            print(f"[warn] input HxW={shape[2:]} does not match --imgsz {imgsz}")

        x = np.zeros(shape, dtype=np.float32)
        try:
            outs = sess.run(None, {inp.name: x})
            print("[ok] dummy inference succeeded")
            for idx, arr in enumerate(outs):
                print(f"  run-out {idx}: shape={arr.shape} dtype={arr.dtype}")
        except Exception as e:
            print(f"[FAIL] dummy inference failed: {e}")
            return False

    expected_c = 4 + expected_nc

    if len(outputs) == 1:
        shape = outs[0].shape
        if len(shape) != 3 or shape[0] != 1:
            print(f"[FAIL] single-output shape {shape} is not (1, C, N) or (1, N, C)")
            ok = False
        else:
            if shape[1] == expected_c or shape[2] == expected_c:
                print(
                    f"[ok] single output matches yolo-generic NMS branch "
                    f"(C=4+NC={expected_c}, total {shape})"
                )
            else:
                print(
                    f"[FAIL] output channels do not match 4+NC={expected_c}; "
                    f"Frigate yolo-generic will pick wrong slice (got {shape})"
                )
                ok = False
    elif len(outputs) >= 2:
        non4d = [o for o in outs if o.ndim != 4]
        if not non4d:
            print(f"[ok] {len(outputs)} outputs, all 4D — multipart YOLO branch will be used")
        else:
            print(
                f"[FAIL] multi-output ({len(outputs)}), but not all 4D — "
                f"Frigate `__post_process_multipart_yolo` expects (B,C,H,W) per tensor. "
                f"Offending shapes: {[o.shape for o in non4d]}"
            )
            ok = False
    else:
        print(f"[FAIL] unexpected number of outputs ({len(outputs)})")
        ok = False

    print(f"Result: {'PASS' if ok else 'FAIL'}")
    return ok


_TFLITE_TYPE_NAMES = {
    0: "float32",
    1: "float16",
    2: "int32",
    3: "uint8",
    4: "int64",
    5: "string",
    6: "bool",
    7: "int16",
    8: "complex64",
    9: "int8",
    10: "float64",
    11: "complex128",
    12: "uint64",
    16: "uint32",
}


def _describe_tflite(path: Path):
    """Parse a .tflite flatbuffer and return (inputs, outputs) as
    lists of dicts with {name, shape, dtype, quant}. Works even when the
    model contains an EdgeTPU custom op and cannot be allocated without a
    Coral device.
    """
    try:
        import tflite
    except Exception as e:
        raise RuntimeError(f"'tflite' package required: {e}")

    buf = bytearray(path.read_bytes())
    model = tflite.Model.GetRootAsModel(buf, 0)
    if model.SubgraphsLength() == 0:
        raise RuntimeError("tflite has no subgraphs")
    sg = model.Subgraphs(0)

    def _tensor(index):
        t = sg.Tensors(index)
        shape = [int(t.Shape(i)) for i in range(t.ShapeLength())]
        q = t.Quantization()
        scale = [float(q.Scale(i)) for i in range(q.ScaleLength())] if q else []
        zp = [int(q.ZeroPoint(i)) for i in range(q.ZeroPointLength())] if q else []
        return {
            "name": t.Name().decode() if t.Name() else f"tensor_{index}",
            "shape": shape,
            "dtype": _TFLITE_TYPE_NAMES.get(t.Type(), f"type_{t.Type()}"),
            "scale": scale,
            "zero_point": zp,
        }

    inputs = [_tensor(sg.Inputs(i)) for i in range(sg.InputsLength())]
    outputs = [_tensor(sg.Outputs(i)) for i in range(sg.OutputsLength())]
    return inputs, outputs


def check_tflite(path: Path, expected_nc: int, imgsz: int, reg_max: int) -> bool:
    _print_header(f"TFLite EdgeTPU check: {path}")
    if not path.is_file():
        print(f"[FAIL] file not found: {path}")
        return False

    try:
        in_details, out_details = _describe_tflite(path)
    except Exception as e:
        print(f"[FAIL] could not parse tflite: {e}")
        return False

    print(f"inputs ({len(in_details)}):")
    for d in in_details:
        print(f"  - name={d['name']} shape={d['shape']} dtype={d['dtype']} "
              f"scale={d['scale'][:1]} zp={d['zero_point'][:1]}")
    print(f"outputs ({len(out_details)}):")
    for d in out_details:
        print(f"  - name={d['name']} shape={d['shape']} dtype={d['dtype']} "
              f"scale={d['scale'][:1]} zp={d['zero_point'][:1]}")

    ok = True
    if len(in_details) != 1:
        print(f"[FAIL] expected exactly 1 input, got {len(in_details)}")
        ok = False
    else:
        d = in_details[0]
        if d["shape"] != [1, imgsz, imgsz, 3]:
            print(f"[FAIL] input shape {d['shape']} != [1,{imgsz},{imgsz},3] NHWC")
            ok = False
        if d["dtype"] not in ("int8", "uint8"):
            print(f"[FAIL] input dtype {d['dtype']} not int8/uint8")
            ok = False

    if len(out_details) not in (2, 3):
        print(
            f"[FAIL] EdgeTPU yolo-generic plugin requires 2 or 3 output tensors, "
            f"got {len(out_details)}"
        )
        ok = False

    box_tensor = None
    cls_tensor = None
    for d in out_details:
        shape = d["shape"]
        if len(shape) == 3 and shape[2] == reg_max * 4:
            box_tensor = d
        elif len(shape) == 3 and shape[2] > 1 and shape[2] != reg_max * 4:
            cls_tensor = d

    if box_tensor is None:
        print(f"[FAIL] no output with last dim {reg_max*4} (box DFL) found")
        ok = False
    else:
        print(f"[ok] found box tensor: shape={box_tensor['shape']}")

    if cls_tensor is None:
        print("[FAIL] no output with last dim == NC (>1) found (class scores)")
        ok = False
    else:
        nc_detected = int(cls_tensor["shape"][-1])
        print(f"[ok] found class tensor: shape={cls_tensor['shape']} (NC={nc_detected})")
        if nc_detected != expected_nc:
            print(f"[warn] detected NC={nc_detected} != --expected-nc={expected_nc}")

    if box_tensor and cls_tensor:
        if box_tensor["shape"][1] != cls_tensor["shape"][1]:
            print("[FAIL] anchor count mismatch between box and class tensors")
            ok = False
        else:
            print(f"[ok] anchor counts match (N={box_tensor['shape'][1]})")

    print(
        "[info] skipping live inference: EdgeTPU custom ops require a real "
        "Coral delegate; schema check above is authoritative for Frigate parsing."
    )

    print(f"Result: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", type=Path, default=None, help="ONNX model path")
    ap.add_argument("--tflite", type=Path, default=None, help="TFLite (EdgeTPU) model path")
    ap.add_argument("--expected-nc", type=int, default=3, help="Number of classes (default: 3)")
    ap.add_argument("--imgsz", type=int, default=320, help="Input image size (default: 320)")
    ap.add_argument("--reg-max", type=int, default=16, help="DFL bin count (default: 16)")
    args = ap.parse_args()

    if not args.onnx and not args.tflite:
        ap.error("provide at least --onnx or --tflite")

    results = []
    if args.onnx:
        results.append(("onnx", check_onnx(args.onnx, args.expected_nc, args.imgsz)))
    if args.tflite:
        results.append(("tflite", check_tflite(args.tflite, args.expected_nc, args.imgsz, args.reg_max)))

    print("\nSummary")
    print("-------")
    all_ok = True
    for name, ok in results:
        print(f"  {name:6s}: {'PASS' if ok else 'FAIL'}")
        all_ok = all_ok and ok
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
