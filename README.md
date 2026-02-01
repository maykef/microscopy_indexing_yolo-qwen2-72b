# Microscopy Document Indexing & Extraction Pipeline

Advanced document processing pipeline for scientific microscopy literature using YOLO-based layout detection and Qwen2-VL-72B for high-accuracy text extraction without OCR.

## Overview

This repository provides a production-ready pipeline for processing scientific PDF documents (particularly microscopy textbooks and papers) into structured, searchable text. Unlike traditional OCR approaches, this pipeline uses:

- **DocLayout-YOLO** (YOLOv10) for precise document layout detection
- **Qwen2-VL-72B** for vision-language-based text extraction
- **Spatial-aware reading order** reconstruction for multi-column layouts
- **LaTeX formula preservation** with rendering support

The pipeline achieves superior accuracy on complex scientific documents containing equations, multi-column layouts, figures, and technical notation.

## Key Features

- ✅ **OCR-free text extraction** using vision-language models
- ✅ **Layout-aware processing** (headers, columns, formulas, captions)
- ✅ **Checkpoint/resume** for long-running extractions
- ✅ **Batched inference** with automatic OOM recovery
- ✅ **LaTeX formula detection** and rendering
- ✅ **Multi-column document** handling
- ✅ **Token counting** and performance metrics
- ✅ **Markdown export** with proper reading order

## Architecture

```
PDF → Images → YOLO Detection → Text Extraction → Structured JSON → Markdown
         ↓           ↓                  ↓                ↓              ↓
    pdf2image  DocLayout-YOLO    Qwen2-VL-72B      Block sorting   json_to_markup.py
                (layout)         (transcription)   (reading order)
```

### Pipeline Stages

1. **PDF Rasterization** (`pdf_to_images.py`)
   - Convert PDF pages to high-resolution images (200 DPI default)
   - Organized output structure for processing

2. **Layout Detection** (`text_extraction_sequential.py`)
   - YOLO-based detection of text blocks, formulas, headers, captions
   - Bounding box extraction with confidence scores
   - Support for 6 content classes: Text, Title, Section-header, List-item, Caption, Formula

3. **Text Extraction** (`text_extraction_sequential.py`)
   - Vision-language model transcription of detected blocks
   - Batched processing with OOM protection
   - Checkpoint-based resume capability

4. **Post-processing** (`json_to_markup.py`)
   - Spatial sorting for natural reading order
   - Multi-column layout detection
   - LaTeX formula rendering
   - Markdown/plaintext export

## Installation

### Prerequisites

- NVIDIA GPU with 80+ GB VRAM (for Qwen2-VL-72B in 8-bit)
- Docker with NVIDIA Container Toolkit
- CUDA 12.6+

### Quick Start with Docker

```bash
# Clone the repository
git clone https://github.com/yourusername/microscopy_indexing_yolo-qwen2-72b.git
cd microscopy_indexing_yolo-qwen2-72b

# Set up directories (customize as needed)
export DATA_DIR=/mnt/nvme8tb/microscopy_index
export HF_HUB_DIR=/mnt/nvme8tb/huggingface_cache/hub
export MODELS_DIR=/mnt/nvme8tb/microscopy_index_models

# Build and launch container
./manage_env.sh
```

The container will:
- Build the Docker image with all dependencies
- Download and cache required models
- Mount your data directories
- Configure CUDA and shared memory

### Manual Installation

If you prefer not to use Docker:

```bash
# Install system dependencies
apt-get update && apt-get install -y \
    python3-pip python3-dev git poppler-utils \
    libgl1-mesa-glx libglib2.0-0 build-essential

# Install PyTorch with CUDA support
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu126

# Install core dependencies
pip install colpali-engine transformers accelerate einops \
    bitsandbytes pdf2image qwen-vl-utils sentencepiece \
    pillow tqdm protobuf flash-attn ultralytics

# Install DocLayout-YOLO
pip install doclayout-yolo==0.0.4

# Download YOLO weights
# Place doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt in HF cache
```

## Usage

### 1. Convert PDF to Images

```bash
python pdf_to_images.py your_document.pdf --dpi 200 --format jpg
```

**Options:**
- `--dpi`: Resolution for rendering (default: 200)
- `--format`: Output format - jpg or png (default: jpg)
- `--threads`: Parallel rendering threads (default: 16)

**Output:** `./index_all_cache/visual/images/<pdf_name>/page_0000.jpg`

### 2. Extract Text from Images

```bash
python text_extraction_sequential.py
```

**Configuration** (edit in script):
```python
IMAGE_DIR = Path("index_all_cache/visual/images/Your_Document")
OUTPUT_DIR = Path("index_all_cache/scribe")
NUM_PAGES = 0  # 0 = process all pages
BATCH_SIZE = 16  # Adjust based on VRAM
```

**Features:**
- Automatic checkpoint/resume on interruption
- OOM-safe batching with dynamic sizing
- Per-page VRAM monitoring and cleanup
- Detailed performance metrics

**Output:** JSON file with detected blocks and extracted text

### 3. Convert to Markdown

```bash
python json_to_markup.py extraction.json -o output.md
```

**Options:**
- `-f, --format`: Output format (markdown or plain)
- `--no-page-numbers`: Omit page markers
- `--no-render-formulas`: Keep LaTeX as-is
- `--preserve-layout`: Experimental spatial layout preservation

**Output:** Markdown file with:
- Proper reading order (multi-column aware)
- Rendered LaTeX formulas as display math blocks
- Preserved document structure (headers, lists, captions)

### 4. Count Tokens (Analysis)

```bash
python count_tokens.py /path/to/extraction/directory -r --csv stats.csv
```

**Options:**
- `-r, --recursive`: Scan subdirectories
- `--csv`: Export statistics to CSV

**Output:** Token counts, word counts, and extraction statistics

## Performance

### Benchmarks (Qwen2-VL-72B, 8-bit, A100 80GB)

| Metric | Value |
|--------|-------|
| Throughput | ~15-25 tokens/sec |
| VRAM Usage | 70-90 GB |
| Pages/Hour | ~30-50 (depends on density) |
| Accuracy | 95%+ on scientific text |

### Optimization Tips

1. **Batch Size**: Start with 16, reduce if OOM occurs
2. **VRAM Threshold**: Set cleanup threshold to 85-90 GB
3. **Max Chunk Height**: 600px works well for most layouts
4. **DPI**: 200 DPI balances quality and processing time

## Document Classes Supported

The pipeline detects and processes:

- **Text**: Regular paragraph text
- **Title**: Document/section titles
- **Section-header**: Subsection headers
- **List-item**: Bulleted/numbered lists
- **Caption**: Figure and table captions
- **Formula**: Mathematical equations (LaTeX)

Tables are currently not extracted (YOLO model limitation).

## Advanced Features

### Checkpoint Recovery

If processing is interrupted:
```bash
# Simply re-run the script - it will resume automatically
python text_extraction_sequential.py
```

The checkpoint tracks:
- Last processed page
- Total blocks extracted
- Accumulated statistics

### Formula Deduplication

The pipeline automatically detects and removes duplicate formulas that may appear in headers/footers:

```python
# In json_to_markup.py
deduplicate_formulas(blocks, threshold=0.85)
```

### Multi-Column Detection

Spatial analysis automatically detects column breaks:
- X-coordinate clustering for column identification
- Proper reading order: top-to-bottom, left-to-right within columns
- Headers and footers handled separately

## File Structure

```
microscopy_indexing_yolo-qwen2-72b/
├── Dockerfile                          # Container definition
├── manage_env.sh                       # Build and launch script
├── pdf_to_images.py                    # PDF → images conversion
├── text_extraction_sequential.py      # Main extraction pipeline
├── json_to_markup.py                   # JSON → Markdown converter
├── count_tokens.py                     # Analysis tool
├── README.md                           # This file
└── index_all_cache/                    # Output directory (created)
    ├── visual/images/                  # Rendered PDF pages
    └── scribe/                         # Extraction results
```
