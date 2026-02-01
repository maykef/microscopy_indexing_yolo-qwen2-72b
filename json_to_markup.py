#!/usr/bin/env python3
"""
json_to_markup.py - Reconstruct text with proper reading order from YOLO bboxes

Converts extraction JSON to markdown with spatially-aware block ordering.
Handles multi-column layouts, preserves paragraph breaks, and maintains headers.
"""

import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple

# Try to import latex rendering
try:
    from latex2mathml.converter import convert as latex_to_mathml
    LATEX_AVAILABLE = True
except ImportError:
    LATEX_AVAILABLE = False


class Block:
    """Represents a text block with spatial coordinates."""
    
    def __init__(self, data: Dict):
        self.bbox = data['bbox']  # [x1, y1, x2, y2]
        self.class_name = data['class_name']
        self.text = data['text']
        self.confidence = data['confidence']
        
        # Calculate derived properties
        self.x1, self.y1, self.x2, self.y2 = self.bbox
        self.center_x = (self.x1 + self.x2) / 2
        self.center_y = (self.y1 + self.y2) / 2
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1
        
    def __repr__(self):
        return f"Block({self.class_name}, y={self.y1:.0f}, text='{self.text[:30]}...')"


def detect_columns(blocks: List[Block], x_threshold: float = 100) -> List[List[Block]]:
    """
    Detect column structure by analyzing X-coordinate clustering.
    
    Args:
        blocks: List of blocks to analyze
        x_threshold: Minimum X-gap to consider a column break
        
    Returns:
        List of column groups (each group is a list of blocks)
    """
    if not blocks:
        return []
    
    # Sort by center_x to find columns
    sorted_blocks = sorted(blocks, key=lambda b: b.center_x)
    
    columns = []
    current_column = [sorted_blocks[0]]
    
    for block in sorted_blocks[1:]:
        # Check X-gap from last block in current column
        x_gap = block.center_x - current_column[-1].center_x
        
        if x_gap > x_threshold:
            # Start new column
            columns.append(current_column)
            current_column = [block]
        else:
            current_column.append(block)
    
    columns.append(current_column)
    return columns


def sort_reading_order(blocks: List[Block], y_tolerance: float = 20) -> List[Block]:
    """
    Sort blocks in natural reading order (top-to-bottom, left-to-right).
    
    Handles multi-column layouts by detecting column breaks.
    
    Args:
        blocks: List of blocks to sort
        y_tolerance: Y-distance within which blocks are considered on same line
        
    Returns:
        Sorted list of blocks in reading order
    """
    if not blocks:
        return []
    
    # First, group into horizontal bands (rows)
    sorted_by_y = sorted(blocks, key=lambda b: b.y1)
    
    rows = []
    current_row = [sorted_by_y[0]]
    
    for block in sorted_by_y[1:]:
        # Check if block is on same horizontal line
        y_diff = abs(block.y1 - current_row[0].y1)
        
        if y_diff <= y_tolerance:
            current_row.append(block)
        else:
            # Sort current row left-to-right, then start new row
            rows.append(sorted(current_row, key=lambda b: b.x1))
            current_row = [block]
    
    # Add last row
    rows.append(sorted(current_row, key=lambda b: b.x1))
    
    # Flatten rows back into single list
    return [block for row in rows for block in row]


def calculate_spacing(prev_block: Block, curr_block: Block) -> Tuple[int, str]:
    """
    Calculate spacing between blocks.
    
    Returns:
        (newlines, spacing_type) where spacing_type is one of:
        'inline', 'paragraph', 'section'
    """
    y_gap = curr_block.y1 - prev_block.y2
    
    if y_gap < 10:
        return 0, 'inline'
    elif y_gap < 30:
        return 1, 'paragraph'
    else:
        return 2, 'section'


def render_latex(latex_str: str, format: str = 'markdown') -> str:
    """
    Render LaTeX string to appropriate output format.
    
    Args:
        latex_str: LaTeX formula string
        format: Output format ('markdown', 'html', 'mathml')
        
    Returns:
        Rendered formula string
    """
    # Clean up common LaTeX wrappers
    latex_str = latex_str.strip()
    
    # Remove outer \( \) or \[ \] if present (more aggressive)
    latex_str = re.sub(r'^\\\((.+?)\\\)\s*\.?$', r'\1', latex_str)
    latex_str = re.sub(r'^\\\[(.+?)\\\]\s*\.?$', r'\1', latex_str, flags=re.DOTALL)
    
    # Remove any remaining inline \(...\) wrappers
    latex_str = re.sub(r'\\\(([^)]+)\\\)', r'\1', latex_str)
    latex_str = re.sub(r'\\\[([^\]]+)\\\]', r'\1', latex_str)
    
    # Remove trailing punctuation (periods, commas inside display math)
    latex_str = re.sub(r'\s*[.,;:!?]+\s*$', '', latex_str)
    
    # Clean up extra whitespace
    latex_str = re.sub(r'\s+', ' ', latex_str).strip()
    
    if format == 'mathml' and LATEX_AVAILABLE:
        try:
            return latex_to_mathml(latex_str)
        except Exception as e:
            # Fallback to original LaTeX
            return f"$$\n{latex_str}\n$$"
    elif format == 'html':
        # Wrap for MathJax/KaTeX
        return f'<div class="math">$$\n{latex_str}\n$$</div>'
    else:  # markdown (default)
        # Display math block (works in Obsidian, Typora, Jupyter, GitHub)
        return f"$$\n{latex_str}\n$$"


def format_block_text(block: Block, prev_block: Block = None, render_formulas: bool = True) -> str:
    """
    Format block text with appropriate markdown styling.
    
    Args:
        block: Current block to format
        prev_block: Previous block (for spacing calculation)
        render_formulas: Whether to render LaTeX formulas
        
    Returns:
        Formatted markdown text
    """
    text = block.text.strip()
    
    # Add spacing before block
    if prev_block:
        newlines, spacing_type = calculate_spacing(prev_block, block)
        prefix = '\n' * newlines
    else:
        prefix = ''
    
    # Apply formatting based on class
    if block.class_name == 'Title':
        return f"{prefix}# {text}\n"
    elif block.class_name == 'Section-header':
        return f"{prefix}## {text}\n"
    elif block.class_name == 'Caption':
        return f"{prefix}*{text}*\n"
    elif block.class_name == 'List-item':
        return f"{prefix}- {text}\n"
    elif block.class_name == 'Formula':
        # Render or preserve LaTeX
        if render_formulas:
            rendered = render_latex(text, format='markdown')
            return f"\n{rendered}\n"
        else:
            return f"\n\n{text}\n\n"
    else:  # Text
        return f"{prefix}{text}\n"


def normalize_latex(latex_str: str) -> str:
    """
    Normalize LaTeX for comparison (remove wrappers, whitespace, punctuation).
    
    Args:
        latex_str: LaTeX string
        
    Returns:
        Normalized LaTeX
    """
    # Remove common wrappers (need to escape backslashes properly)
    latex_str = re.sub(r'^\\\\\((.+?)\\\\\)$', r'\1', latex_str.strip())
    latex_str = re.sub(r'^\\\\\[(.+?)\\\\\]$', r'\1', latex_str.strip())
    latex_str = re.sub(r'^\$\$(.+?)\$\$$', r'\1', latex_str.strip())
    latex_str = re.sub(r'^\$(.+?)\$$', r'\1', latex_str.strip())
    
    # Remove trailing punctuation
    latex_str = re.sub(r'[.,;:!?]+$', '', latex_str)
    
    # Remove extra whitespace
    latex_str = re.sub(r'\s+', ' ', latex_str)
    
    return latex_str.strip()


def deduplicate_formulas(blocks: List[Block], threshold: float = 0.85) -> List[Block]:
    """
    Remove duplicate formula blocks based on text similarity.
    
    Args:
        blocks: List of blocks
        threshold: Similarity threshold (0-1) for considering formulas duplicates
        
    Returns:
        Deduplicated list of blocks
    """
    seen_formulas = {}
    deduplicated = []
    
    for block in blocks:
        if block.class_name == 'Formula':
            normalized = normalize_latex(block.text)
            
            # Check if similar formula already seen
            is_duplicate = False
            for seen_norm, seen_block in seen_formulas.items():
                # Simple similarity: character-level comparison
                if normalized == seen_norm or normalized in seen_norm or seen_norm in normalized:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_formulas[normalized] = block
                deduplicated.append(block)
        else:
            deduplicated.append(block)
    
    return deduplicated


def process_page(page_data: Dict, preserve_layout: bool = False, 
                render_formulas: bool = True, deduplicate: bool = True) -> str:
    """
    Process single page and return markdown text.
    
    Args:
        page_data: Page data from JSON
        preserve_layout: If True, preserve spatial layout (experimental)
        render_formulas: If True, render LaTeX formulas to display math blocks
        deduplicate: If True, remove duplicate formulas
        
    Returns:
        Markdown formatted text
    """
    blocks = [Block(b) for b in page_data['blocks']]
    
    if not blocks:
        return ""
    
    # Deduplicate formulas if requested
    if deduplicate:
        blocks = deduplicate_formulas(blocks)
    
    # Sort in reading order
    sorted_blocks = sort_reading_order(blocks)
    
    # Format each block
    output = []
    prev_block = None
    
    for block in sorted_blocks:
        formatted = format_block_text(block, prev_block, render_formulas)
        output.append(formatted)
        prev_block = block
    
    return ''.join(output)


def json_to_markdown(json_path: str, output_path: str = None, 
                     page_numbers: bool = True,
                     preserve_layout: bool = False,
                     render_formulas: bool = True,
                     deduplicate: bool = True) -> str:
    """
    Convert extraction JSON to markdown.
    
    Args:
        json_path: Path to extraction JSON
        output_path: Path to output markdown file (optional)
        page_numbers: Include page number markers
        preserve_layout: Preserve spatial layout (experimental)
        render_formulas: Render LaTeX formulas to display math blocks
        deduplicate: Remove duplicate formula blocks
        
    Returns:
        Markdown text
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    output = []
    
    for i, page_data in enumerate(data['results']):
        page_num = i + 1
        
        if page_numbers:
            output.append(f"\n---\n**Page {page_num}**\n\n")
        
        page_text = process_page(page_data, preserve_layout, render_formulas, deduplicate)
        output.append(page_text)
    
    markdown_text = ''.join(output)
    
    if output_path:
        Path(output_path).write_text(markdown_text, encoding='utf-8')
        print(f"✓ Wrote {len(data['results'])} pages to {output_path}")
    
    return markdown_text


def json_to_plaintext(json_path: str, output_path: str = None) -> str:
    """
    Convert extraction JSON to plain text (no markdown formatting).
    
    Args:
        json_path: Path to extraction JSON
        output_path: Path to output text file (optional)
        
    Returns:
        Plain text
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    output = []
    
    for page_data in data['results']:
        blocks = [Block(b) for b in page_data['blocks']]
        sorted_blocks = sort_reading_order(blocks)
        
        for block in sorted_blocks:
            output.append(block.text.strip())
            output.append('\n\n')
    
    plain_text = ''.join(output)
    
    if output_path:
        Path(output_path).write_text(plain_text, encoding='utf-8')
        print(f"✓ Wrote plain text to {output_path}")
    
    return plain_text


def main():
    parser = argparse.ArgumentParser(
        description='Convert extraction JSON to markdown with proper reading order'
    )
    parser.add_argument('json_path', help='Path to extraction JSON file')
    parser.add_argument('-o', '--output', help='Output file path (default: input.md)')
    parser.add_argument('-f', '--format', choices=['markdown', 'plain'], 
                       default='markdown', help='Output format')
    parser.add_argument('--no-page-numbers', action='store_true',
                       help='Omit page number markers')
    parser.add_argument('--no-render-formulas', action='store_true',
                       help='Keep LaTeX as-is, do not render to display math')
    parser.add_argument('--preserve-layout', action='store_true',
                       help='Preserve spatial layout (experimental)')
    
    args = parser.parse_args()
    
    # Default output path
    if not args.output:
        json_path = Path(args.json_path)
        ext = '.txt' if args.format == 'plain' else '.md'
        args.output = json_path.with_suffix(ext)
    
    # Convert
    if args.format == 'plain':
        json_to_plaintext(args.json_path, args.output)
    else:
        json_to_markdown(
            args.json_path, 
            args.output,
            page_numbers=not args.no_page_numbers,
            preserve_layout=args.preserve_layout,
            render_formulas=not args.no_render_formulas
        )


if __name__ == '__main__':
    main()
