#!/usr/bin/env python3
"""
PRODUCTION VERSION - Sequential processing with checkpoint/resume
IMPROVED THROUGHPUT (SAFE: deterministic decode unchanged)

Key improvements vs your script:
- NO torch.cuda.empty_cache() per block (major speed win)
- Batched generation per page (amortizes overhead across many paragraph crops)
- Correct generated token counting (uses generated token ids, not re-tokenizing decoded text)
- Uses torch.inference_mode() (slightly faster than no_grad)
- Safer device handling when model uses device_map="auto" (don't blindly .to("cuda:0"))
- OOM-safe dynamic batch sizing (auto-reduces batch size and retries)
- Prompt string defined once (minor CPU savings)
- Optional per-page cache cleanup (instead of per-block)

Design constraints preserved:
- YOLO boxes are used as-is (no paragraph aggregation)
- Tables not extracted because YOLO classes exclude them (unchanged)
- Decoding is deterministic: do_sample=False, repetition_penalty=1.0 (unchanged)
"""

import os
import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

try:
    from transformers import AutoModelForImageTextToText as QwenModelClass
except Exception:
    from transformers import AutoModelForVision2Seq as QwenModelClass

try:
    from doclayout_yolo import YOLOv10
    DOCLAYOUT_YOLO_AVAILABLE = True
except Exception:
    DOCLAYOUT_YOLO_AVAILABLE = False


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
os.environ.pop("TRANSFORMERS_CACHE", None)
HF_HOME = os.environ.get("HF_HOME", "/root/.cache/huggingface")
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HF_HUB_CACHE", str(Path(HF_HOME) / "hub"))

IMAGE_DIR = Path("index_all_cache/visual/images/Gonzales_&_Woods_2018")
OUTPUT_DIR = Path("index_all_cache/scribe")

PDF_NAME = IMAGE_DIR.name
OUTPUT_FILE = OUTPUT_DIR / f"{PDF_NAME}_extraction_V4.json"
CHECKPOINT_FILE = OUTPUT_DIR / f"{PDF_NAME}_checkpoint_V4.json"

QWEN_MODEL_ID = "Qwen/Qwen2-VL-72B-Instruct"
YOLO_MODEL_FILENAME = "doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt"

# Processing control
NUM_PAGES = 0  # 0 = all remaining
MAX_NEW_TOKENS = 1024
YOLO_IMGSZ = 1120
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.3
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

USE_QUANTIZATION = True
QUANTIZATION_CONFIG = {
    "load_in_8bit": True,
    "llm_int8_threshold": 6.0,
}

TEXT_CLASSES = [
    "Text",
    "Title",
    "Section-header",
    "List-item",
    "Caption",
    "Formula"
]

# Batching / chunking
BATCH_SIZE = 16           # try 4 / 8 / 16; auto-reduced on OOM
MIN_BATCH_SIZE = 1
MAX_CHUNK_HEIGHT = 600   # same logic you used
PER_PAGE_EMPTY_CACHE = False  # optional: only after each page

# Prompt (defined once)
SCRIBE_PROMPT = """You are a precise OCR system. Your ONLY job is to READ and TRANSCRIBE text.

ABSOLUTE RULES - NO EXCEPTIONS:
1. Output ONLY the text visible in this image
2. NEVER add explanations, definitions, elaborations, or context
3. NEVER complete cut-off sentences or paragraphs
4. NEVER invent chemical formulas, reactions, or technical details
5. If text is cut off at an edge, STOP IMMEDIATELY at that exact point
6. If something is unclear, write [unclear] and move on
7. DO NOT try to be helpful - just transcribe

YOUR TASK: Copy visible text character-by-character. NOTHING ELSE."""


# -----------------------------------------------------------------------------
# VRAM monitoring / cleanup (SAFE, page-boundary only)
# -----------------------------------------------------------------------------
VRAM_CLEANUP_THRESHOLD_GB = 90.0

def maybe_cleanup_vram():
    if not torch.cuda.is_available():
        return

    try:
        import pynvml
        # Initialize NVML if not already done
        try:
            pynvml.nvmlInit()
        except:
            pass  # Already initialized
        
        # Get actual GPU memory usage (what system monitor shows)
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        used_gb = info.used / 1024**3
        total_gb = info.total / 1024**3
        
        # Also get PyTorch's view for comparison
        allocated = torch.cuda.memory_allocated(device=0) / 1024**3
        reserved = torch.cuda.memory_reserved(device=0) / 1024**3
        
        # Always print for monitoring
        print(f"      💾 GPU Memory: {used_gb:.1f} / {total_gb:.1f} GB ({used_gb/total_gb*100:.0f}%)")
        print(f"         PyTorch: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved")

        # Check ACTUAL GPU memory (what you see in system monitor)
        if used_gb >= VRAM_CLEANUP_THRESHOLD_GB:
            print(f"      🧹 VRAM cleanup triggered ({used_gb:.1f} GB >= {VRAM_CLEANUP_THRESHOLD_GB} GB)")
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
            # Check after cleanup
            info_after = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_after_gb = info_after.used / 1024**3
            print(f"      ✨ After cleanup: {used_after_gb:.1f} GB ({used_after_gb/total_gb*100:.0f}%)")
            
    except ImportError:
        # Fallback to PyTorch if pynvml not available
        print("      ⚠️  pynvml not available, using PyTorch metrics (may be inaccurate)")
        device = torch.device("cuda:0")
        allocated = torch.cuda.memory_allocated(device=device) / 1024**3
        reserved = torch.cuda.memory_reserved(device=device) / 1024**3
        
        print(f"      💾 VRAM: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved")

        if reserved >= VRAM_CLEANUP_THRESHOLD_GB:
            print(f"      🧹 VRAM cleanup triggered (reserved {reserved:.1f} GB >= {VRAM_CLEANUP_THRESHOLD_GB} GB)")
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


# -----------------------------------------------------------------------------
# Checkpoint management
# -----------------------------------------------------------------------------
@dataclass
class Checkpoint:
    last_page_processed: Optional[str] = None
    pages_processed: int = 0
    total_blocks_extracted: int = 0
    total_words_extracted: int = 0
    results_file: Optional[str] = None

    def save(self):
        CHECKPOINT_FILE.parent.mkdir(exist_ok=True)
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls):
        if CHECKPOINT_FILE.exists():
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(**data)
        return cls()


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
@dataclass
class PerformanceMetrics:
    yolo_load_time: float = 0.0
    qwen_load_time: float = 0.0
    yolo_detection_time: float = 0.0
    total_inference_time: float = 0.0  # wall time spent in model.generate (plus prep for those calls)
    total_execution_time: float = 0.0
    pages_processed: int = 0
    text_blocks_detected: int = 0
    text_blocks_extracted: int = 0
    total_tokens_generated: int = 0
    total_words_extracted: int = 0
    tokens_per_second: float = 0.0
    peak_vram_gb: float = 0.0

    def compute_derived(self):
        if self.total_inference_time > 0:
            self.tokens_per_second = self.total_tokens_generated / self.total_inference_time


METRICS = PerformanceMetrics()


# -----------------------------------------------------------------------------
# JSON serialization
# -----------------------------------------------------------------------------
def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
def resolve_hf_snapshot_dir(model_id: str) -> Path:
    org, name = model_id.split("/", 1)
    model_cache_dir = Path(os.environ["HF_HUB_CACHE"]) / f"models--{org}--{name}"
    snapshots_dir = model_cache_dir / "snapshots"

    if not snapshots_dir.exists():
        raise FileNotFoundError(f"Missing snapshots dir: {snapshots_dir}")

    snaps = sorted(
        [p for p in snapshots_dir.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not snaps:
        raise FileNotFoundError(f"No snapshots under: {snapshots_dir}")

    return snaps[0]


def patch_doclayout_yolo_bn_bug():
    try:
        from doclayout_yolo.nn.modules import g2l_crm
        import torch.nn.functional as F
    except Exception:
        return

    def _patched_dilated_conv(self, x, dilation):
        weight = self.dcv.conv.weight
        padding = dilation * (self.k // 2)
        x = F.conv2d(x, weight, stride=1, padding=padding, dilation=dilation)
        bn = getattr(self.dcv, "bn", None)
        if bn is not None:
            x = bn(x)
        act = getattr(self.dcv, "act", None)
        if act is not None:
            x = act(x)
        return x

    for _, obj in vars(g2l_crm).items():
        if isinstance(obj, type) and hasattr(obj, "dilated_conv"):
            setattr(obj, "dilated_conv", _patched_dilated_conv)


def load_yolo_model():
    print("\n" + "=" * 80)
    print("LOADING DOCLAYOUT-YOLO MODEL (YOLOv10)")
    print("=" * 80)

    patch_doclayout_yolo_bn_bug()

    weights_path = Path(os.environ["HF_HUB_CACHE"]) / YOLO_MODEL_FILENAME
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")

    print(f"✅ Found weights: {weights_path}")

    t0 = time.time()
    model = YOLOv10(str(weights_path))
    if hasattr(model, "to"):
        model.to(DEVICE)
    elapsed = time.time() - t0

    METRICS.yolo_load_time = elapsed
    print(f"✅ YOLOv10 loaded in {elapsed:.2f}s (device={DEVICE})")
    return model


def _first_param_device(model: torch.nn.Module) -> torch.device:
    # Safe device for inputs when device_map="auto" is used.
    try:
        return next(model.parameters()).device
    except StopIteration:
        # Fallback
        return torch.device(DEVICE)


def load_qwen_model():
    print("\n" + "=" * 80)
    print("LOADING QWEN2-VL-72B (LOCAL SNAPSHOT)")
    print("=" * 80)
    print(f"📥 Model: {QWEN_MODEL_ID}")
    print(f"   Quantization: {'8-bit' if USE_QUANTIZATION else 'Full precision'}")

    t0 = time.time()
    snapshot_path = resolve_hf_snapshot_dir(QWEN_MODEL_ID)
    print(f"✅ Using snapshot: {snapshot_path}")

    processor = AutoProcessor.from_pretrained(
        str(snapshot_path),
        trust_remote_code=True,
        local_files_only=True,
    )

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": True,
    }

    if USE_QUANTIZATION:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(**QUANTIZATION_CONFIG)
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs["device_map"] = DEVICE

    model = QwenModelClass.from_pretrained(
        str(snapshot_path),
        **model_kwargs,
    ).eval()

    elapsed = time.time() - t0
    METRICS.qwen_load_time = elapsed

    if torch.cuda.is_available():
        vram = torch.cuda.max_memory_allocated() / 1024**3
        METRICS.peak_vram_gb = vram
        print(f"✅ Qwen-72B loaded in {elapsed:.2f}s | VRAM: {vram:.2f} GB")
    else:
        print(f"✅ Qwen-72B loaded in {elapsed:.2f}s (CPU)")

    return model, processor


# -----------------------------------------------------------------------------
# Detection and extraction
# -----------------------------------------------------------------------------
@dataclass
class TextBlock:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_name: str

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def detect_text_blocks(yolo_model, image_path: Path) -> List[TextBlock]:
    t0 = time.time()

    results = yolo_model.predict(
        source=str(image_path),
        imgsz=YOLO_IMGSZ,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False,
    )

    elapsed = time.time() - t0
    METRICS.yolo_detection_time += elapsed

    blocks: List[TextBlock] = []
    if results and len(results) > 0:
        result = results[0]
        if hasattr(result, "boxes") and result.boxes is not None:
            boxes = result.boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                class_name = yolo_model.names[cls_id] if hasattr(yolo_model, "names") else str(cls_id)

                if class_name not in TEXT_CLASSES:
                    continue

                blocks.append(
                    TextBlock(
                        x1=int(xyxy[0]),
                        y1=int(xyxy[1]),
                        x2=int(xyxy[2]),
                        y2=int(xyxy[3]),
                        confidence=conf,
                        class_name=class_name,
                    )
                )
    return blocks


def sort_text_blocks(blocks: List[TextBlock], page_width: int, page_height: int) -> List[TextBlock]:
    if not blocks:
        return []

    full_width_threshold = 0.65 * page_width
    mid_x = page_width / 2.0
    top_threshold = page_height * 0.15

    headers: List[TextBlock] = []
    left_blocks: List[TextBlock] = []
    right_blocks: List[TextBlock] = []
    footnotes: List[TextBlock] = []

    for block in blocks:
        if block.width > full_width_threshold:
            if block.y1 < top_threshold:
                headers.append(block)
            else:
                footnotes.append(block)
        elif block.center_x < mid_x:
            left_blocks.append(block)
        else:
            right_blocks.append(block)

    headers.sort(key=lambda b: b.y1)
    left_blocks.sort(key=lambda b: b.y1)
    right_blocks.sort(key=lambda b: b.y1)
    footnotes.sort(key=lambda b: b.y1)

    return headers + left_blocks + right_blocks + footnotes


def _make_text_prompts(processor, imgs: List[Image.Image]) -> List[str]:
    # One prompt per image (no paragraph aggregation; purely batching plumbing)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": SCRIBE_PROMPT},
            ],
        }
        for img in imgs
    ]
    # apply_chat_template expects a list of messages per sample
    return [
        processor.apply_chat_template([m], tokenize=False, add_generation_prompt=True)
        for m in messages
    ]


def _generate_batched(
    model,
    processor,
    imgs: List[Image.Image],
    max_new_tokens: int,
) -> Tuple[List[str], List[int], float]:
    """
    Returns: (decoded_texts, generated_token_counts, elapsed_seconds)
    """
    # Build batch inputs
    text_prompts = _make_text_prompts(processor, imgs)
    inputs = processor(
        text=text_prompts,
        images=imgs,
        padding=True,
        return_tensors="pt",
    )

    # If device_map="auto", inputs should live on a "primary" device,
    # not blindly .to("cuda:0").
    dev = _first_param_device(model)
    inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}

    t0 = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    elapsed = time.time() - t0

    # Decode per sample
    in_len = inputs["input_ids"].shape[1]
    decoded: List[str] = []
    tok_counts: List[int] = []

    for i in range(output_ids.shape[0]):
        gen_ids = output_ids[i][in_len:]
        tok_counts.append(int(gen_ids.numel()))
        decoded.append(processor.decode(gen_ids, skip_special_tokens=True).strip())

    # Cleanup
    del inputs, output_ids
    return decoded, tok_counts, elapsed


def _is_oom(e: BaseException) -> bool:
    msg = str(e).lower()
    return "out of memory" in msg or "cuda oom" in msg or "cublas" in msg


def generate_with_oom_retry(
    model,
    processor,
    imgs: List[Image.Image],
    max_new_tokens: int,
    batch_size: int,
    min_batch_size: int = 1,
) -> Tuple[List[str], List[int], float]:
    """
    Runs batched generation; if OOM occurs, reduces batch size and retries.
    Returns texts, token_counts, total_elapsed.
    """
    all_texts: List[str] = []
    all_tok_counts: List[int] = []
    total_elapsed = 0.0

    i = 0
    bs = batch_size

    while i < len(imgs):
        cur_bs = min(bs, len(imgs) - i)
        batch_imgs = imgs[i:i + cur_bs]

        try:
            texts, tok_counts, elapsed = _generate_batched(model, processor, batch_imgs, max_new_tokens)
            total_elapsed += elapsed
            all_texts.extend(texts)
            all_tok_counts.extend(tok_counts)
            i += cur_bs

            # Optional: if we've recovered from earlier OOM, cautiously ramp back up
            # (keeps it stable and reduces thrash)
            if bs < batch_size:
                bs = min(batch_size, bs * 2)

        except RuntimeError as e:
            if torch.cuda.is_available() and _is_oom(e) and cur_bs > min_batch_size:
                print(f"      ⚠️ OOM at batch_size={cur_bs}; reducing batch size...")
                # Free cached memory once on OOM
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                bs = max(min_batch_size, cur_bs // 2)
                continue
            raise

    return all_texts, all_tok_counts, total_elapsed


def crop_blocks_to_images(
    page_image: Image.Image,
    blocks: List[TextBlock],
    max_chunk_height: int,
) -> Tuple[List[Image.Image], List[Tuple[int, bool]]]:
    """
    Returns:
      crops: list of PIL images (each is either a full block crop or a chunk)
      meta:  list of (block_index, is_chunk) aligned with crops
    """
    crops: List[Image.Image] = []
    meta: List[Tuple[int, bool]] = []

    for bi, block in enumerate(blocks):
        region = page_image.crop((block.x1, block.y1, block.x2, block.y2))

        if region.height > max_chunk_height:
            num_chunks = (region.height + max_chunk_height - 1) // max_chunk_height
            chunk_height = region.height // num_chunks

            for ci in range(num_chunks):
                y_start = ci * chunk_height
                y_end = region.height if ci == num_chunks - 1 else (ci + 1) * chunk_height
                chunk = region.crop((0, y_start, region.width, y_end))
                crops.append(chunk)
                meta.append((bi, True))

            region.close()
        else:
            crops.append(region)
            meta.append((bi, False))

    return crops, meta


def blocks_from_batched_outputs(
    blocks: List[TextBlock],
    crops_meta: List[Tuple[int, bool]],
    out_texts: List[str],
    out_tok_counts: List[int],
    total_elapsed: float,
) -> List[Dict]:
    """
    Reassemble chunked outputs back into per-block text and per-block stats.
    We distribute batch elapsed time proportionally by generated token count.
    """
    # Initialize per-block accumulators
    per_block_text_chunks: Dict[int, List[str]] = {i: [] for i in range(len(blocks))}
    per_block_tokens: Dict[int, int] = {i: 0 for i in range(len(blocks))}

    # Collect outputs
    for (block_idx, _is_chunk), text, tok in zip(crops_meta, out_texts, out_tok_counts):
        per_block_text_chunks[block_idx].append(text)
        per_block_tokens[block_idx] += tok

    # Allocate elapsed time proportionally by tokens (better than equal split)
    total_tokens = sum(out_tok_counts) if out_tok_counts else 0
    per_block_elapsed: Dict[int, float] = {i: 0.0 for i in range(len(blocks))}
    if total_tokens > 0 and total_elapsed > 0:
        for i in range(len(blocks)):
            per_block_elapsed[i] = total_elapsed * (per_block_tokens[i] / total_tokens)
    else:
        # fallback: equal split
        n = max(1, len(blocks))
        for i in range(len(blocks)):
            per_block_elapsed[i] = total_elapsed / n

    # Build final block results
    block_results: List[Dict] = []
    for i, block in enumerate(blocks):
        text = "\n\n".join(per_block_text_chunks[i]).strip()
        tokens = per_block_tokens[i]
        elapsed = per_block_elapsed[i]
        words = len(text.split()) if text else 0

        METRICS.total_tokens_generated += tokens
        METRICS.total_words_extracted += words
        METRICS.total_inference_time += elapsed

        block_results.append(
            {
                "bbox": [block.x1, block.y1, block.x2, block.y2],
                "confidence": float(block.confidence),
                "class_name": block.class_name,
                "text": text,
                "stats": {
                    "tokens_generated": tokens,
                    "words_extracted": words,
                    "inference_time": elapsed,
                    "tokens_per_second": (tokens / elapsed) if elapsed > 0 else 0.0,
                },
            }
        )

    return block_results


# -----------------------------------------------------------------------------
# Main with checkpoint/resume
# -----------------------------------------------------------------------------
def main() -> int:
    t_start = time.time()

    print("\n" + "=" * 80)
    print("PRODUCTION EXTRACTION - SEQUENTIAL WITH CHECKPOINT/RESUME (BATCHED SCRIBE)")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Strategy: YOLO ({', '.join(TEXT_CLASSES)}) + Qwen-72B (8-bit) batched per page")
    print(f"Batch size: {BATCH_SIZE} (auto-reduce on OOM)")
    print("=" * 80)

    if not DOCLAYOUT_YOLO_AVAILABLE:
        print("❌ doclayout_yolo not available")
        return 1

    if not IMAGE_DIR.exists():
        print(f"❌ Image directory not found: {IMAGE_DIR}")
        return 1

    checkpoint = Checkpoint.load()

    all_pages = sorted(IMAGE_DIR.glob("page_*.jpg"))
    if not all_pages:
        print(f"❌ No images found in {IMAGE_DIR}")
        return 1

    print(f"📚 Total pages available: {len(all_pages)}")

    # Determine which pages to process
    if checkpoint.last_page_processed:
        last_idx = None
        for i, p in enumerate(all_pages):
            if p.name == checkpoint.last_page_processed:
                last_idx = i
                break
        if last_idx is not None:
            pages_to_process = all_pages[last_idx + 1:]
            print(f"📌 Resuming from: {checkpoint.last_page_processed}")
            print(f"   Already processed: {checkpoint.pages_processed} pages")
            print(f"   Remaining: {len(pages_to_process)} pages")
        else:
            print(f"⚠️  Could not find last page {checkpoint.last_page_processed}, starting over")
            pages_to_process = all_pages
    else:
        print("🆕 Starting fresh extraction")
        pages_to_process = all_pages

    if NUM_PAGES > 0:
        pages_to_process = pages_to_process[:NUM_PAGES]
        print(f"🎯 Processing next {len(pages_to_process)} pages")
    else:
        print(f"🎯 Processing all remaining {len(pages_to_process)} pages")

    if not pages_to_process:
        print("✅ All pages already processed!")
        return 0

    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"📁 Output: {OUTPUT_DIR}")

    # Load models
    yolo = load_yolo_model()
    qwen_model, qwen_proc = load_qwen_model()

    print("\n" + "=" * 80)
    print(f"PROCESSING {len(pages_to_process)} PAGES")
    print("=" * 80)

    results = []

    # Load existing results if continuing
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            results = existing_data.get("results", [])
        print(f"📥 Loaded {len(results)} existing page results from {OUTPUT_FILE}")
    elif checkpoint.results_file and Path(checkpoint.results_file).exists():
        with open(checkpoint.results_file, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            results = existing_data.get("results", [])
        print(f"📥 Loaded {len(results)} existing page results from {checkpoint.results_file}")
        print(f"   (Will migrate to {OUTPUT_FILE})")

    for img_path in pages_to_process:
        print(f"\n📄 {img_path.name} (Total progress: {checkpoint.pages_processed + 1})")

        try:
            image = Image.open(img_path).convert("RGB")

            print("   🎯 Detecting text blocks...")
            blocks = detect_text_blocks(yolo, img_path)

            if not blocks:
                print("      ⚠️  No text blocks detected, skipping page")
                checkpoint.pages_processed += 1
                checkpoint.last_page_processed = img_path.name
                checkpoint.save()
                image.close()
                continue

            METRICS.text_blocks_detected += len(blocks)
            print(f"      ✅ Detected {len(blocks)} blocks")

            type_counts: Dict[str, int] = {}
            for b in blocks:
                type_counts[b.class_name] = type_counts.get(b.class_name, 0) + 1
            for class_name, count in sorted(type_counts.items()):
                print(f"         • {class_name}: {count}")

            sorted_blocks = sort_text_blocks(blocks, image.width, image.height)

            # Crop all blocks (and chunks) first
            crops, meta = crop_blocks_to_images(image, sorted_blocks, MAX_CHUNK_HEIGHT)
            print(f"   🧩 Crops prepared: {len(crops)} (includes chunks for tall regions)")

            # Batched generation (OOM-safe)
            print(f"   🚀 Batched transcription (batch_size={BATCH_SIZE})...")
            out_texts, out_tok_counts, elapsed = generate_with_oom_retry(
                qwen_model,
                qwen_proc,
                crops,
                MAX_NEW_TOKENS,
                batch_size=BATCH_SIZE,
                min_batch_size=MIN_BATCH_SIZE,
            )

            # Close crop images as soon as done
            for c in crops:
                try:
                    c.close()
                except Exception:
                    pass

            # Update VRAM peak
            if torch.cuda.is_available():
                vram = torch.cuda.max_memory_allocated() / 1024**3
                METRICS.peak_vram_gb = max(METRICS.peak_vram_gb, vram)

            # Reassemble outputs into per-block results, update metrics
            block_results = blocks_from_batched_outputs(
                sorted_blocks, meta, out_texts, out_tok_counts, elapsed
            )

            METRICS.text_blocks_extracted += len(sorted_blocks)
            METRICS.pages_processed += 1

            # Save page results
            results.append(
                {
                    "page": img_path.name,
                    "text_blocks_detected": len(blocks),
                    "text_blocks_extracted": len(sorted_blocks),
                    "blocks": block_results,
                }
            )

            # Update checkpoint
            checkpoint.pages_processed += 1
            checkpoint.last_page_processed = img_path.name
            checkpoint.total_blocks_extracted += len(sorted_blocks)
            checkpoint.total_words_extracted += sum(b["stats"]["words_extracted"] for b in block_results)

            payload = {
                "config": {
                    "strategy": "sequential_with_checkpoint_batched",
                    "yolo_weights": YOLO_MODEL_FILENAME,
                    "yolo_classes_used": TEXT_CLASSES,
                    "qwen_model": QWEN_MODEL_ID,
                    "imgsz": YOLO_IMGSZ,
                    "conf": CONFIDENCE_THRESHOLD,
                    "iou": IOU_THRESHOLD,
                    "max_tokens": MAX_NEW_TOKENS,
                    "device": DEVICE,
                    "batch_size": BATCH_SIZE,
                    "max_chunk_height": MAX_CHUNK_HEIGHT,
                },
                "results": results,
                "performance_metrics": asdict(METRICS),
            }

            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False, default=json_default)

            checkpoint.results_file = str(OUTPUT_FILE)
            checkpoint.save()

            print(f"   ✅ Page complete | {len(sorted_blocks)} blocks | {sum(out_tok_counts)} tok | {elapsed:.1f}s")
            print(f"      -> ~{(sum(out_tok_counts)/elapsed if elapsed>0 else 0):.1f} tok/s (page batch)")

            image.close()

            maybe_cleanup_vram()

            if PER_PAGE_EMPTY_CACHE and torch.cuda.is_available():
                # Optional: do this per page only (not per block)
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            # Save checkpoint even on error so you can keep moving
            checkpoint.pages_processed += 1
            checkpoint.last_page_processed = img_path.name
            checkpoint.save()

    METRICS.total_execution_time = time.time() - t_start
    METRICS.compute_derived()

    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"Pages processed (this run):  {METRICS.pages_processed}")
    print(f"Total pages processed:       {checkpoint.pages_processed}")
    print(f"Text blocks extracted:       {METRICS.text_blocks_extracted}")
    print(f"Words extracted:             {METRICS.total_words_extracted:,}")
    print(f"Tokens generated:            {METRICS.total_tokens_generated:,}")
    print(f"Generation rate (overall):   {METRICS.tokens_per_second:.1f} tok/s")
    print(f"Execution time:              {METRICS.total_execution_time:.1f}s ({METRICS.total_execution_time/60:.1f} min)")
    print(f"Peak VRAM:                   {METRICS.peak_vram_gb:.2f} GB")

    if OUTPUT_FILE.exists():
        print(f"\n📄 Results saved: {OUTPUT_FILE}")
        print(f"📌 Checkpoint: {CHECKPOINT_FILE}")

    remaining = len(all_pages) - checkpoint.pages_processed
    if remaining > 0:
        print(f"\n⏭️  {remaining} pages remaining")
        print("   Run script again to continue")
    else:
        print("\n🎉 ALL PAGES COMPLETE!")

    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
