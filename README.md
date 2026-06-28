# voice-grpc-service

GPU-accelerated text-to-speech gRPC service running [Piper TTS](https://github.com/OHF-voice/piper1-gpl) on a Jetson Orin Nano Super. Receives text over the network and plays it through the local ALSA audio device.

## Hardware / software

| Item | Value |
|---|---|
| Board | Jetson Orin Nano Super |
| JetPack | 7.2 |
| CUDA | 13.2 |
| Python | 3.12 |
| piper-tts | 1.4.2 |
| onnxruntime-gpu | 1.24.0 |
| grpcio | 1.81.1 |

## How it works

```
caller  ──gRPC──►  main.py  ──queue──►  worker thread
                                              │
                              PiperVoice (ONNX / CUDA)
                                              │
                                           aplay → speaker
```

`main.py` exposes a single gRPC method `SpeakLocally`. Incoming requests are placed onto a bounded `queue.Queue`. A single background worker thread drains the queue one job at a time — it is the sole owner of the ONNX inference session and the CUDA context. This is intentional: on Jetson's unified-memory GPU (nvgpu), concurrent `session.run()` calls from multiple threads corrupt the CUDA state. The worker thread prevents that entirely. If the queue is full (default: 3 waiting) the server returns an error immediately instead of piling up unbounded work.

---

## Environment setup — the JetPack 7.2 "long journey"

Setting up a modern AI environment on Ubuntu 24.04 (Python 3.12) with JetPack 7.2 (CUDA 13.2) requires navigating several significant architectural shifts. JetPack 7 transitioned the Orin family to the ARM SBSA (Server Base System Architecture) standard, aligning edge devices with NVIDIA's data center chips. This breaks legacy installation habits. Below is the exact, battle-tested sequence.

### 1. Bypass the Ubuntu 24.04 venv bug

Standard environment creation (`python3 -m venv .venv`) hangs indefinitely on Jetson due to a bug in the bundled `ensurepip` extraction process. Bypass it:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv curl

# Create the environment without pip, then fetch pip directly from PyPA
python3 -m venv .venv --without-pip
source .venv/bin/activate
curl -sS https://bootstrap.pypa.io/get-pip.py | python3
```

### 2. Install SBSA-compatible AI wheels

Standard PyPI wheels fall back to CPU. Pull pre-compiled, hardware-accelerated ARM64 wheels from the Jetson AI Lab SBSA index:

```bash
pip install torch torchvision torchaudio onnxruntime-gpu \
    --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130
pip install grpcio grpcio-tools piper-tts
```

See `requirements.txt` for the exact pinned versions of all installed packages.

### 3. Resolve the missing NVPL link

JetPack 7 replaced standard OpenBLAS with NVIDIA Performance Libraries (NVPL). The PyTorch wheel expects these at the OS level but they are not installed by default:

```bash
wget https://developer.download.nvidia.com/compute/nvpl/25.5/local_installers/nvpl-local-repo-ubuntu2404-25.5_1.0-1_arm64.deb
sudo dpkg -i nvpl-local-repo-ubuntu2404-25.5_1.0-1_arm64.deb
sudo cp /var/nvpl-local-repo-ubuntu2404-25.5/nvpl-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install -y nvpl
```

### 4. Resolve the "two CUDAs" conflict

PyTorch 2.9+ for ARM requires cuDSS (CUDA Direct Sparse Solver). Installing it via pip pulls down generic server dependencies (`cublas`, `cuda-runtime`) that hijack the linker path and crash against Jetson's native L4T drivers.

**Step 1** — install cuDSS, then immediately remove the conflicting generic dependencies:

```bash
pip install nvidia-cudss-cu13
pip uninstall -y nvidia-cublas-cu13 nvidia-cuda-runtime-cu13 \
    nvidia-cusparse-cu13 nvidia-nvjitlink-cu13
```

**Step 2** — tell the linker where `libcudss.so.0` lives. Add this line to `.venv/bin/activate` so it applies on every activation:

```bash
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH"
```

### 5. Compile the gRPC stubs

Run once after cloning, or whenever `protos/voice.proto` changes:

```bash
./compile-protos.sh
```

This regenerates `voice_pb2.py` and `voice_pb2_grpc.py`.

### 6. Piper espeak-ng data

The `piper/` directory must contain the espeak-ng data that ships with the Piper binary release. Download the aarch64 release from the [Piper releases page](https://github.com/rhasspy/piper/releases) and extract it — only the `espeak-ng-data/` subdirectory is needed:

```
piper/
└── espeak-ng-data/    ← required
```

---

## Voice models

Two models are included:

| File | Language | Voice | Quality |
|---|---|---|---|
| `en_US-lessac-high.onnx` | English (US) | Lessac | High |
| `de_DE-thorsten-high.onnx` | German | Thorsten | High |

Additional voices can be downloaded from the [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) repository on Hugging Face. Each voice needs two files: the `.onnx` model and the `.onnx.json` config. Example for a German female voice:

```bash
wget "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/kerstin/low/de_DE-kerstin-low.onnx"
wget "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/kerstin/low/de_DE-kerstin-low.onnx.json"
```

---

## Running the server

```bash
source .venv/bin/activate

# English, GPU (default)
python3 main.py

# German, GPU
python3 main.py --model de_DE-thorsten-high.onnx

# Explicit options
python3 main.py \
  --model de_DE-thorsten-high.onnx \
  --device plughw:0,0 \
  --port 50051

# CPU only
python3 main.py --no-cuda
```

### Server options

| Flag | Default | Description |
|---|---|---|
| `-m` / `--model` | `en_US-lessac-high.onnx` | Path to Piper ONNX voice model |
| `--no-cuda` | off | Disable GPU, run on CPU |
| `-d` / `--device` | `plughw:0,0` | ALSA device passed to `aplay` |
| `-p` / `--port` | `50051` | gRPC listen port |

On startup the server logs which provider is active:

```
INFO Voice loaded on GPU (CUDAExecutionProvider).
INFO Voice gRPC server listening on port 50051  model=en_US-lessac-high.onnx  cuda=True
```

### CUDA note for Jetson

onnxruntime prints this warning on every startup — it is harmless:

```
[W] GPU device discovery failed: Failed to open file: "/sys/class/drm/card1/device/vendor"
```

Jetson's integrated nvgpu does not appear under `/sys/class/drm/card1`. CUDA still initialises correctly and `CUDAExecutionProvider` is active. You can confirm GPU load with `jtop`.

---

## Using the client

```bash
source .venv/bin/activate

# Speak on the local machine
python3 client.py "Hello, this is a test."

# Speak on a remote Jetson
python3 client.py "Hallo Welt." --host 192.168.1.42

# Different port
python3 client.py "Test." --host 192.168.1.42 --port 50052
```

### Client options

| Flag | Default | Description |
|---|---|---|
| `text` | (required) | Text to synthesise and play |
| `-H` / `--host` | `localhost` | Server hostname or IP |
| `-p` / `--port` | `50051` | Server port |

---

## Project structure

```
voice-grpc-service/
├── main.py                       # gRPC server + synthesis worker
├── client.py                     # CLI client
├── protos/
│   └── voice.proto               # Service definition
├── voice_pb2.py                  # Generated — do not edit
├── voice_pb2_grpc.py             # Generated — do not edit
├── compile-protos.sh             # Regenerates the two files above
├── en_US-lessac-high.onnx        # English voice model
├── en_US-lessac-high.onnx.json   # English voice config
├── de_DE-thorsten-high.onnx      # German voice model
├── de_DE-thorsten-high.onnx.json # German voice config
└── piper/
    └── espeak-ng-data/           # Phoneme data for espeak-ng
```
