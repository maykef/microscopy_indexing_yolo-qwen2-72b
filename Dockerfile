# Base image
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=interactive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    nano \
    git \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

# PyTorch nightly (as in your original)
RUN pip3 install --no-cache-dir --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu126

# Build helpers
RUN pip3 install --no-cache-dir packaging setuptools wheel

# Core deps
RUN pip3 install --no-cache-dir \
    colpali-engine \
    transformers \
    accelerate \
    einops \
    bitsandbytes \
    pdf2image \
    qwen-vl-utils \
    sentencepiece \
    pillow \
    tqdm \
    protobuf

# Flash Attention (as in your original)
RUN pip3 install --no-cache-dir flash-attn --no-build-isolation

# ---------------------------
# Ultralytics (explicit)
# ---------------------------
RUN pip3 install --no-cache-dir ultralytics

# ---------------------------
# DocLayout-YOLO (DocLayNet)
# ---------------------------
# Install doclayout-yolo at build time (avoid runtime autoinstall + restart)
RUN pip3 install --no-cache-dir doclayout-yolo==0.0.4

# Patch doclayout-yolo bug that can throw:
# AttributeError: 'Conv' object has no attribute 'bn'
RUN python3 - <<'PY'
import site, pathlib

targets = []
for sp in site.getsitepackages():
    p = pathlib.Path(sp) / "doclayout_yolo" / "nn" / "modules" / "g2l_crm.py"
    if p.exists():
        targets.append(p)

if not targets:
    raise SystemExit("ERROR: doclayout_yolo g2l_crm.py not found in site-packages")

p = targets[0]
s = p.read_text()
orig = s

# Guard bn/act access (covers common variants)
s = s.replace("bn = self.dcv.bn\n        x = bn(x)\n",
              "bn = getattr(self.dcv, 'bn', None)\n        if bn is not None:\n            x = bn(x)\n")

s = s.replace("act = self.dcv.act\n        x = act(x)\n",
              "act = getattr(self.dcv, 'act', None)\n        if act is not None:\n            x = act(x)\n")

s = s.replace("x = self.dcv.bn(x)\n",
              "bn = getattr(self.dcv, 'bn', None)\n        if bn is not None:\n            x = bn(x)\n")

s = s.replace("x = self.dcv.act(x)\n",
              "act = getattr(self.dcv, 'act', None)\n        if act is not None:\n            x = act(x)\n")

p.write_text(s)

print(f"✅ Patched: {p}")
print(f"changed={s != orig}")
PY

# Sanity imports
RUN python3 -c "from doclayout_yolo import YOLOv10; print('✅ doclayout_yolo YOLOv10 import OK')"
RUN python3 -c "import ultralytics; print('✅ ultralytics import OK')"

# Pre-download ColPali (as in your original)
RUN python3 -c "from colpali_engine.models import ColPali, ColPaliProcessor; \
    print('Downloading ColPali...'); \
    ColPali.from_pretrained('vidore/colpali-v1.2', torch_dtype='float32', device_map='cpu'); \
    ColPaliProcessor.from_pretrained('vidore/colpali-v1.2'); \
    print('✅ ColPali cached')"

# Hardware check script
RUN echo 'import torch; \
import flash_attn; \
import qwen_vl_utils; \
import ultralytics; \
from doclayout_yolo import YOLOv10; \
name = torch.cuda.get_device_name(0); \
vram = torch.cuda.get_device_properties(0).total_memory / 1e9; \
print("-" * 40); \
print(f"✅ DEVICE: {name}"); \
print(f"✅ VRAM: {vram:.1f} GB"); \
print(f"✅ FLASH ATTENTION: LOADED"); \
print(f"✅ QWEN UTILS: LOADED"); \
print(f"✅ ULTRALYTICS: LOADED"); \
print(f"✅ DOCLAYOUT_YOLO: LOADED"); \
print("-" * 40)' > check_hw.py

CMD ["/bin/bash"]
