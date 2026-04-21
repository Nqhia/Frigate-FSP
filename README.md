# Frigate FSP - Fire, Smoke & Person Detection

YOLOv9-s fine-tuned model (320x320) running on Frigate NVR with **dual hardware support**:

| Mode | Hardware | Model format | Docker image |
|------|----------|-------------|--------------|
| Edge TPU | Google Coral USB / PCIe | INT8 TFLite | `frigate:stable` |
| GPU | NVIDIA GPU (CUDA) | ONNX | `frigate:stable-tensorrt` (amd64) |

**Classes:** `person`, `fire`, `smoke`

## Prerequisites

- Docker & Docker Compose
- **Edge TPU mode:** Google Coral USB Accelerator (or PCIe/M.2 variant)
- **GPU mode:** NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- RTSP camera stream

## Quick Start

### 1. Configure your camera

Edit the camera section in the appropriate config file and replace the RTSP URL:

- Edge TPU: `config/config.yml`
- GPU: `config/config-gpu.yml`

```yaml
cameras:
  cam_01:
    ffmpeg:
      inputs:
        - path: rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101
```

### 2. Place your model files

```
config/model_cache/
├── best_320_int8_edgetpu.tflite   # Edge TPU model
└── best_320_frigate.onnx          # ONNX model (for GPU, Frigate-compatible)
```

> The ONNX file must expose a single output tensor `(1, 4+NC, N)` for
> Frigate's `yolo-generic` detector. Default YOLOv9 exports that emit
> `boxes_dfl_logit` + `class_scores_logit` are **not** compatible; use
> `scripts/export_onnx_frigate.py` to re-export from `best.pt`.

### 3. Start Frigate

**Edge TPU (Coral):**

```bash
docker compose up -d
```

**GPU (NVIDIA + ONNX):**

```bash
docker compose -f docker-compose.gpu.yml up -d
```

If you are on ARM/Jetson, set image tag first (example for JetPack 6):

```bash
export FRIGATE_IMAGE=ghcr.io/blakeblackshear/frigate:stable-tensorrt-jp6
docker compose -f docker-compose.gpu.yml up -d
```

### 4. Open the Web UI

[http://localhost:5000](http://localhost:5000)

## Exporting the ONNX Model

Frigate's `yolo-generic` ONNX parser expects a single output tensor shaped
`(1, 4+NC, N)` with decoded `xywh` boxes (pixel space) and post-sigmoid
class scores. The default YOLOv9 export flow produces raw DFL logits, which
Frigate cannot parse. Use the bundled script to export a compatible model:

```bash
python scripts/export_onnx_frigate.py \
    --weights /path/to/best.pt \
    --yolov9  /path/to/yolov9 \
    --cfg     models/detect/yolov9-s-relu6.yaml \
    --imgsz 320 --nc 3 \
    --out config/model_cache/best_320_frigate.onnx --simplify
```

## Verifying Models

Run the check script before deploying. It validates shapes, dtypes, and
(for EdgeTPU) the output layout Frigate's EdgeTPU plugin relies on. It
parses `.tflite` via flatbuffers, so no Coral device is required:

```bash
python scripts/check_frigate_models.py \
    --onnx   config/model_cache/best_320_frigate.onnx \
    --tflite config/model_cache/best_320_int8_edgetpu.tflite \
    --expected-nc 3 --imgsz 320
```

> **Note:** The ONNX model uses FP32 precision (not INT8). GPU inference
> compensates with parallel throughput.

## PCIe Coral

If using a PCIe/M.2 Coral instead of USB, update both files:

`docker-compose.yml` — uncomment the PCIe device line and remove the USB device line.

`config/config.yml` — change detector device:

```yaml
detectors:
  coral:
    type: edgetpu
    device: pci
```

## Model Info

| Property       | Edge TPU                              | GPU (ONNX)                   |
|----------------|---------------------------------------|------------------------------|
| Architecture   | YOLOv9-s (ReLU6 + DualDDetectEdgeTPU) | YOLOv9-s                     |
| Input size     | 320 x 320                             | 320 x 320                    |
| Quantization   | INT8 (full EdgeTPU mapping)           | FP32                         |
| Input format   | NHWC (uint8)                          | NCHW (float32)               |
| Inference      | ~10 ms on USB Coral                   | ~5-15 ms (depends on GPU)    |

## Project Structure

```
frigate-FSP/
├── docker-compose.yml          # Edge TPU deployment
├── docker-compose.gpu.yml      # GPU (ONNX) deployment
├── config/
│   ├── config.yml              # Frigate config — Edge TPU
│   ├── config-gpu.yml          # Frigate config — GPU (ONNX)
│   ├── labels_fsp3.txt         # Label map (person, fire, smoke)
│   └── model_cache/
│       ├── best_320_int8_edgetpu.tflite
│       └── best_320_frigate.onnx
├── scripts/
│   ├── export_onnx_frigate.py  # Wrap YOLOv9 head -> Frigate-compatible ONNX
│   └── check_frigate_models.py # Validate ONNX + TFLite shapes for Frigate
├── storage/                    # Recordings & snapshots (auto-created)
└── README.md
```
