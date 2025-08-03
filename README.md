# MCP Server with FastMCP

## Overview

This project provides a Model Context Protocol (MCP) server written in Python using the FastMCP framework. It exposes two main tools:

1. **generate\_image**: Generates images from a text prompt using StableDiffusionXL, streams progress steps, applies an aesthetic scoring loop, and returns the best-scoring image as a Base64‑encoded PNG.
2. **aesthetic\_score**: Scores any Base64‑encoded PNG image using a pretrained vision model and returns a float score.

The server automatically detects available hardware at startup (CUDA GPU, Apple ML Compute, or CPU) and configures the StableDiffusionXL and aesthetic-scoring pipelines accordingly.

---

## Directory Structure

```
mcp_server/
├── app.py                    # FastMCP application entrypoint
├── requirements.txt          # Python dependencies
├── models/
│   ├── sdxl.py               # StableDiffusionXL wrapper
│   └── aesthetic.py          # AestheticScorer wrapper
└── utils/
    ├── device.py             # Hardware detection logic
    └── logger.py             # Structured logging setup
```

---

## Prerequisites

* Python 3.10
* Git (to clone the repo)
* Optional GPU hardware with CUDA or Apple Silicon (ML Compute)

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/mcp_server.git
   cd mcp_server
   ```

2. **Set up a virtual environment**

   ```bash
   python3.10 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Configuration

* By default, the server will:

  * Download StableDiffusionXL weights on startup.
  * Detect hardware: CUDA → MPS → CPU.
  * Listen on port `8000` and all interfaces (`0.0.0.0`).

* To customize host/port/log level, you can modify the `app.run(...)` call in `app.py` or pass environment variables if extended.

---

## Usage

### Running the Server

```bash
python app.py
```

You should see logs such as:

```
2025-07-16T18:00:00 [INFO] [mcp_server] Loaded SDXL on cuda
2025-07-16T18:00:05 [INFO] [mcp_server] Server running on http://0.0.0.0:8000
```

### Tool Endpoints

The MCP server exposes the following endpoints for use by MCP clients (e.g., LM Studio):

#### 1. generate\_image

* **Signature**: `generate_image(prompt: str, steps: int = 50, retry_threshold: float = 10.0, max_retries: int = 3) -> Stream`
* **Behavior**:

  1. Streams progress updates in the form: `step X/50 done`.
  2. Generates an image via StableDiffusionXL.
  3. Scores the image with `aesthetic_score` internally.
  4. Retries up to `max_retries` if score < `retry_threshold`.
  5. Emits final Base64‑encoded PNG of the best-scoring image.

#### 2. aesthetic\_score

* **Signature**: `aesthetic_score(image_b64: str) -> float`
* **Behavior**: Decodes the Base64 image, scores it, and returns a float score (0–10 scale).

### Example (cURL)

```bash
curl -N -X POST http://localhost:8000/tools/generate_image \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "A serene landscape at sunrise",
           "steps": 30,
           "retry_threshold": 8.5,
           "max_retries": 5
         }'
```

You will receive a streaming response:

```
step 1/30 done
step 2/30 done
...
step 30/30 done
<BASE64_PNG_DATA>
```

To score an existing image:

```bash
curl -X POST http://localhost:8000/tools/aesthetic_score \
     -H "Content-Type: application/json" \
     -d '{"image_b64":"<YOUR_BASE64_PNG>"}'
```

---

## Logging

* Uses Python’s built‑in `logging` module.
* Outputs to `stdout` with ISO8601 timestamps, log levels, and module names.
* Default level: `INFO` (configurable in `utils/logger.py`).

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
