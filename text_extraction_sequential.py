#!/usr/bin/env python3
"""
PRODUCTION VERSION - Sequential processing with checkpoint/resume

Changes from V2:
- Processes pages in sorted order (not random)
- Saves checkpoint after each page
- Resumes from last checkpoint on restart
- Processes NUM_PAGES at a time (or continues until done)
"""

import os
import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

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

IMAGE_DIR = Path("index_all_cache/visual/images/Inoue_&_Pawley_2006")
OUTPUT_DIR = Path("index_all_cache/scribe")

# Extract PDF name from image directory path
PDF_NAME = IMAGE_DIR.name  # e.g., "Inoue_&_Pawley_2006"
OUTPUT_FILE = OUTPUT_DIR / f"{PDF_NAME}_extraction.json"
CHECKPOINT_FILE = OUTPUT_DIR / f"{PDF_NAME}_checkpoint.json"

QWEN_MODEL_ID = "Qwen/Qwen2-VL-72B-Instruct"
YOLO_MODEL_FILENAME = "doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt"

# Processing control
NUM_PAGES = 20  # Number of pages to process in this run (0 = process all remaining)
MAX_NEW_TOKENS = 8192
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
    "Caption"
]

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
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls):
        if CHECKPOINT_FILE.exists():
            with open(CHECKPOINT_FILE, 'r') as f:
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
    total_inference_time: float = 0.0
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
# Model loading (same as V2)
# -----------------------------------------------------------------------------
def resolve_hf_snapshot_dir(model_id: str) -> Path:
    org, name = model_id.split("/", 1)
    model_cache_dir = Path(os.environ["HF_HUB_CACHE"]) / f"models--{org}--{name}"
    snapshots_dir = model_cache_dir / "snapshots"
    
    if not snapshots_dir.exists():
        raise FileNotFoundError(f"Missing snapshots dir: {snapshots_dir}")
    
    snaps = sorted([p for p in snapshots_dir.iterdir() if p.is_dir()],
                   key=lambda p: p.stat().st_mtime, reverse=True)
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
    if hasattr(model, 'to'):
        model.to(DEVICE)
    elapsed = time.time() - t0
    
    METRICS.yolo_load_time = elapsed
    print(f"✅ YOLOv10 loaded in {elapsed:.2f}s (device={DEVICE})")
    
    return model

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
    
    model_kwargs = {
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
        **model_kwargs
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
# Detection and extraction (same as V2)
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
    
    blocks = []
    
    if results and len(results) > 0:
        result = results[0]
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                
                class_name = yolo_model.names[cls_id] if hasattr(yolo_model, 'names') else str(cls_id)
                
                if class_name not in TEXT_CLASSES:
                    continue
                
                blocks.append(TextBlock(
                    x1=int(xyxy[0]),
                    y1=int(xyxy[1]),
                    x2=int(xyxy[2]),
                    y2=int(xyxy[3]),
                    confidence=conf,
                    class_name=class_name
                ))
    
    return blocks

def sort_text_blocks(blocks: List[TextBlock], page_width: int, page_height: int) -> List[TextBlock]:
    if not blocks:
        return []
    
    full_width_threshold = 0.65 * page_width
    mid_x = page_width / 2.0
    top_threshold = page_height * 0.15
    bottom_threshold = page_height * 0.85
    
    headers = []
    left_blocks = []
    right_blocks = []
    footnotes = []
    
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

def extract_text_from_block(model, processor, image: Image.Image, block: TextBlock, page_name: str) -> Dict:
    t0 = time.time()
    
    region_img = image.crop((block.x1, block.y1, block.x2, block.y2))
    
    MAX_CHUNK_HEIGHT = 600
    if region_img.height > MAX_CHUNK_HEIGHT:
        chunks = []
        num_chunks = (region_img.height + MAX_CHUNK_HEIGHT - 1) // MAX_CHUNK_HEIGHT
        chunk_height = region_img.height // num_chunks
        
        for i in range(num_chunks):
            y_start = i * chunk_height
            y_end = region_img.height if i == num_chunks - 1 else (i + 1) * chunk_height
            chunk_img = region_img.crop((0, y_start, region_img.width, y_end))
            chunk_text = _extract_from_image(model, processor, chunk_img)
            chunks.append(chunk_text)
        
        text = "\n\n".join(chunks)
        tokens = sum(len(processor.tokenizer.encode(c)) for c in chunks)
        words = len(text.split())
    else:
        text = _extract_from_image(model, processor, region_img)
        tokens = len(processor.tokenizer.encode(text))
        words = len(text.split())
    
    elapsed = time.time() - t0
    
    METRICS.total_inference_time += elapsed
    METRICS.total_tokens_generated += tokens
    METRICS.total_words_extracted += words
    
    if torch.cuda.is_available():
        vram = torch.cuda.max_memory_allocated() / 1024**3
        METRICS.peak_vram_gb = max(METRICS.peak_vram_gb, vram)
    
    region_img.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        "bbox": [block.x1, block.y1, block.x2, block.y2],
        "confidence": float(block.confidence),
        "class_name": block.class_name,
        "text": text,
        "stats": {
            "tokens_generated": tokens,
            "words_extracted": words,
            "inference_time": elapsed,
            "tokens_per_second": tokens / elapsed if elapsed > 0 else 0.0
        }
    }

def _extract_from_image(model, processor, img: Image.Image) -> str:
    prompt = """You are a precise OCR system. Your ONLY job is to READ and TRANSCRIBE text.

ABSOLUTE RULES - NO EXCEPTIONS:
1. Output ONLY the text visible in this image
2. NEVER add explanations, definitions, elaborations, or context
3. NEVER complete cut-off sentences or paragraphs
4. NEVER invent chemical formulas, reactions, or technical details
5. If text is cut off at an edge, STOP IMMEDIATELY at that exact point
6. If something is unclear, write [unclear] and move on
7. DO NOT try to be helpful - just transcribe

YOUR TASK: Copy visible text character-by-character. NOTHING ELSE."""
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text_prompt],
        images=[img],
        padding=True,
        return_tensors="pt"
    ).to(DEVICE)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.0,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    
    generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
    text = processor.decode(generated_ids, skip_special_tokens=True).strip()
    
    del inputs, output_ids, generated_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return text

# -----------------------------------------------------------------------------
# Main with checkpoint/resume
# -----------------------------------------------------------------------------
def main() -> int:
    t_start = time.time()
    
    print("\n" + "=" * 80)
    print("PRODUCTION EXTRACTION - SEQUENTIAL WITH CHECKPOINT/RESUME")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Strategy: YOLO ({', '.join(TEXT_CLASSES)}) + Qwen-72B (8-bit)")
    print("=" * 80)
    
    if not DOCLAYOUT_YOLO_AVAILABLE:
        print("❌ doclayout_yolo not available")
        return 1
    
    if not IMAGE_DIR.exists():
        print(f"❌ Image directory not found: {IMAGE_DIR}")
        return 1
    
    # Load checkpoint
    checkpoint = Checkpoint.load()
    
    # Get all pages in sorted order
    all_pages = sorted(IMAGE_DIR.glob("page_*.jpg"))
    if not all_pages:
        print(f"❌ No images found in {IMAGE_DIR}")
        return 1
    
    print(f"📚 Total pages available: {len(all_pages)}")
    
    # Determine which pages to process
    if checkpoint.last_page_processed:
        # Find index of last processed page
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
        print(f"🆕 Starting fresh extraction")
        pages_to_process = all_pages
    
    # Limit to NUM_PAGES if specified
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
        with open(OUTPUT_FILE, 'r') as f:
            existing_data = json.load(f)
            results = existing_data.get('results', [])
        print(f"📥 Loaded {len(results)} existing page results from {OUTPUT_FILE}")
    elif checkpoint.results_file and Path(checkpoint.results_file).exists():
        # Legacy: migrate from old checkpoint location
        with open(checkpoint.results_file, 'r') as f:
            existing_data = json.load(f)
            results = existing_data.get('results', [])
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
                continue
            
            METRICS.text_blocks_detected += len(blocks)
            print(f"      ✅ Detected {len(blocks)} blocks")
            
            type_counts = {}
            for b in blocks:
                type_counts[b.class_name] = type_counts.get(b.class_name, 0) + 1
            for class_name, count in sorted(type_counts.items()):
                print(f"         • {class_name}: {count}")
            
            sorted_blocks = sort_text_blocks(blocks, image.width, image.height)
            
            block_results = []
            
            for idx, block in enumerate(sorted_blocks, 1):
                print(f"   [{idx}/{len(sorted_blocks)}] 🔍 Extracting...", end=" ", flush=True)
                
                result = extract_text_from_block(qwen_model, qwen_proc, image, block, img_path.name)
                block_results.append(result)
                
                METRICS.text_blocks_extracted += 1
                
                stats = result["stats"]
                print(f"{stats['tokens_generated']} tok, {stats['inference_time']:.1f}s")
            
            results.append({
                "page": img_path.name,
                "text_blocks_detected": len(blocks),
                "text_blocks_extracted": len(sorted_blocks),
                "blocks": block_results
            })
            
            METRICS.pages_processed += 1
            checkpoint.pages_processed += 1
            checkpoint.last_page_processed = img_path.name
            checkpoint.total_blocks_extracted += len(sorted_blocks)
            checkpoint.total_words_extracted += sum(b['stats']['words_extracted'] for b in block_results)
            
            # Save intermediate results and checkpoint after each page
            payload = {
                "config": {
                    "strategy": "sequential_with_checkpoint",
                    "yolo_weights": YOLO_MODEL_FILENAME,
                    "yolo_classes_used": TEXT_CLASSES,
                    "qwen_model": QWEN_MODEL_ID,
                    "imgsz": YOLO_IMGSZ,
                    "conf": CONFIDENCE_THRESHOLD,
                    "iou": IOU_THRESHOLD,
                    "max_tokens": MAX_NEW_TOKENS,
                    "device": DEVICE
                },
                "results": results,
                "performance_metrics": asdict(METRICS)
            }
            
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False, default=json_default)
            
            checkpoint.results_file = str(OUTPUT_FILE)
            checkpoint.save()
            
            print(f"   ✅ Page complete | Checkpoint saved")
            
            image.close()
            
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            # Save checkpoint even on error
            checkpoint.pages_processed += 1
            checkpoint.last_page_processed = img_path.name
            checkpoint.save()
    
    METRICS.total_execution_time = time.time() - t_start
    METRICS.compute_derived()
    
    # Final summary
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"Pages processed (this run):  {METRICS.pages_processed}")
    print(f"Total pages processed:       {checkpoint.pages_processed}")
    print(f"Text blocks extracted:       {METRICS.text_blocks_extracted}")
    print(f"Words extracted:             {METRICS.total_words_extracted:,}")
    print(f"Tokens generated:            {METRICS.total_tokens_generated:,}")
    print(f"Generation rate:             {METRICS.tokens_per_second:.1f} tok/s")
    print(f"Execution time:              {METRICS.total_execution_time:.1f}s ({METRICS.total_execution_time/60:.1f} min)")
    print(f"Peak VRAM:                   {METRICS.peak_vram_gb:.2f} GB")
    
    if OUTPUT_FILE.exists():
        print(f"\n📄 Results saved: {OUTPUT_FILE}")
        print(f"📌 Checkpoint: {CHECKPOINT_FILE}")
    
    remaining = len(all_pages) - checkpoint.pages_processed
    if remaining > 0:
        print(f"\n⏭️  {remaining} pages remaining")
        print(f"   Run script again to continue")
    else:
        print(f"\n🎉 ALL PAGES COMPLETE!")
    
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
