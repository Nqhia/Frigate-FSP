# Frigate FSP - Fire, Smoke & Person Detection with Coral EdgeTPU

YOLOv9-s fine-tuned model (320x320, INT8 quantized) running on Google Coral EdgeTPU via Frigate NVR.

**Classes:** `person`, `fire`, `smoke`

## Prerequisites

- Docker & Docker Compose
- Google Coral USB Accelerator (or PCIe/M.2 variant)
- RTSP camera stream

## Quick Start

1. **Configure your camera** - edit `config/config.yml` and replace the RTSP URL:

   ```yaml
   cameras:
     cam_01:
       ffmpeg:
         inputs:
           - path: rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101
   ```

2. **Start Frigate:**

   ```bash
   docker compose up -d
   ```

3. **Open the Web UI** at [http://localhost:5000](http://localhost:5000)

## PCIe Coral

If using a PCIe/M.2 Coral instead of USB, update both files:

`docker-compose.yml` - uncomment the PCIe device line and remove the USB device line.

`config/config.yml` - change detector device:

```yaml
detectors:
  coral:
    type: edgetpu
    device: pci
```

## Model Info

| Property       | Value                              |
|----------------|------------------------------------|
| Architecture   | YOLOv9-s (ReLU6 + DualDDetectEdgeTPU) |
| Input size     | 320 x 320                          |
| Quantization   | INT8 (full EdgeTPU mapping)        |
| Classes        | person, fire, smoke                |
| Inference      | ~10 ms on USB Coral                |

## Project Structure

```
frigate-FSP/
├── docker-compose.yml
├── config/
│   ├── config.yml              # Frigate configuration
│   ├── labels_fsp3.txt         # Label map (person, fire, smoke)
│   └── model_cache/
│       └── best_320_int8_edgetpu.tflite
├── storage/                    # Recordings & snapshots (auto-created)
└── README.md
```
