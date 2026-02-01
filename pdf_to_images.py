#!/usr/bin/env python3
"""
pdf_to_images.py - Extract PDF pages as individual images

Usage:
    python pdf_to_images.py input.pdf [--dpi 200] [--format jpg]
"""

import argparse
from pathlib import Path
from pdf2image import convert_from_path
from tqdm import tqdm


# Base directory for output (relative to where script is run)
BASE_OUTPUT_DIR = Path("./index_all_cache/visual/images")


def extract_pdf_pages(
    pdf_path: Path,
    dpi: int = 200,
    image_format: str = "jpg",
    threads: int = 16
) -> list[Path]:
    """
    Extract all pages from a PDF as individual images.
    
    Output directory structure:
        ./index_all_cache/visual/images/<pdf_stem>/page_0000.jpg
    
    Args:
        pdf_path: Path to input PDF
        dpi: Resolution for rendering (default: 200)
        image_format: Output format - 'jpg' or 'png' (default: 'jpg')
        threads: Number of threads for PDF rendering (default: 16)
    
    Returns:
        List of paths to saved images
    """
    # Validate inputs
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    if image_format.lower() not in ['jpg', 'jpeg', 'png']:
        raise ValueError(f"Unsupported format: {image_format}. Use 'jpg' or 'png'")
    
    # Create output directory based on PDF stem
    pdf_stem = pdf_path.stem  # e.g., "Birk_2017" from "Birk_2017.pdf"
    output_dir = BASE_OUTPUT_DIR / pdf_stem
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Normalize format
    fmt = 'JPEG' if image_format.lower() in ['jpg', 'jpeg'] else 'PNG'
    ext = 'jpg' if fmt == 'JPEG' else 'png'
    
    print(f"📄 Processing: {pdf_path.name}")
    print(f"📁 Output: {output_dir.absolute()}")
    print(f"🎨 Format: {fmt} @ {dpi} DPI")
    
    # Convert PDF to images
    print("🔄 Rendering pages...")
    pages = convert_from_path(
        str(pdf_path),
        dpi=dpi,
        thread_count=threads
    )
    
    print(f"💾 Saving {len(pages)} pages...")
    saved_paths = []
    
    for i, page in enumerate(tqdm(pages, desc="Saving", unit="page")):
        output_path = output_dir / f"page_{i:04d}.{ext}"
        
        if fmt == 'JPEG':
            page.save(output_path, fmt, quality=95, optimize=True)
        else:
            page.save(output_path, fmt, optimize=True)
        
        saved_paths.append(output_path)
    
    print(f"✅ Complete! {len(saved_paths)} pages saved to {output_dir.absolute()}")
    
    return saved_paths


def main():
    parser = argparse.ArgumentParser(
        description="Extract PDF pages as individual images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to input PDF file"
    )
    
    parser.add_argument(
        "-d", "--dpi",
        type=int,
        default=200,
        help="Resolution for rendering"
    )
    
    parser.add_argument(
        "-f", "--format",
        type=str,
        choices=['jpg', 'jpeg', 'png'],
        default='jpg',
        help="Output image format"
    )
    
    parser.add_argument(
        "-t", "--threads",
        type=int,
        default=16,
        help="Number of threads for PDF rendering"
    )
    
    args = parser.parse_args()
    
    # Extract pages
    extract_pdf_pages(
        pdf_path=args.pdf_path,
        dpi=args.dpi,
        image_format=args.format,
        threads=args.threads
    )


if __name__ == "__main__":
    main()
