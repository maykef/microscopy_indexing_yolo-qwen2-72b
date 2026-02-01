#!/usr/bin/env python3
"""
count_tokens.py - Count tokens in extraction JSON files

Analyzes all JSON files in a directory and reports:
- Tokens per file
- Total tokens across all files
- Words extracted
- Pages processed
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def count_tokens_in_json(json_path: Path) -> Dict[str, int]:
    """
    Count tokens, words, and pages in a single extraction JSON.
    
    Args:
        json_path: Path to extraction JSON file
        
    Returns:
        Dictionary with counts
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_tokens = 0
        total_words = 0
        total_blocks = 0
        
        # Check if this is an extraction JSON with results
        if 'results' not in data:
            return {
                'pages': 0,
                'blocks': 0,
                'tokens': 0,
                'words': 0,
                'error': 'Not an extraction JSON'
            }
        
        pages = len(data['results'])
        
        for page in data['results']:
            for block in page.get('blocks', []):
                total_blocks += 1
                
                # Count tokens from stats if available
                if 'stats' in block and 'tokens_generated' in block['stats']:
                    total_tokens += block['stats']['tokens_generated']
                
                # Count words from text
                if 'text' in block:
                    words = len(block['text'].split())
                    total_words += words
        
        return {
            'pages': pages,
            'blocks': total_blocks,
            'tokens': total_tokens,
            'words': total_words,
            'error': None
        }
        
    except Exception as e:
        return {
            'pages': 0,
            'blocks': 0,
            'tokens': 0,
            'words': 0,
            'error': str(e)
        }


def analyze_directory(directory: Path, recursive: bool = False) -> List[Tuple[Path, Dict]]:
    """
    Analyze all JSON files in directory.
    
    Args:
        directory: Directory to scan
        recursive: If True, scan subdirectories
        
    Returns:
        List of (filepath, stats) tuples
    """
    pattern = '**/*.json' if recursive else '*.json'
    json_files = sorted(directory.glob(pattern))
    
    results = []
    for json_file in json_files:
        stats = count_tokens_in_json(json_file)
        results.append((json_file, stats))
    
    return results


def format_number(n: int) -> str:
    """Format number with thousand separators."""
    return f"{n:,}"


def print_report(results: List[Tuple[Path, Dict]], base_dir: Path):
    """
    Print formatted report.
    
    Args:
        results: List of (filepath, stats) tuples
        base_dir: Base directory for relative paths
    """
    print("=" * 80)
    print("TOKEN COUNT REPORT")
    print("=" * 80)
    print()
    
    # Per-file stats
    print("PER-FILE STATISTICS:")
    print("-" * 80)
    print(f"{'File':<40} {'Pages':>8} {'Blocks':>8} {'Tokens':>12} {'Words':>12}")
    print("-" * 80)
    
    total_pages = 0
    total_blocks = 0
    total_tokens = 0
    total_words = 0
    error_count = 0
    
    for filepath, stats in results:
        rel_path = filepath.relative_to(base_dir)
        
        if stats['error']:
            print(f"{str(rel_path):<40} {'ERROR':>8} - {stats['error']}")
            error_count += 1
            continue
        
        print(f"{str(rel_path):<40} "
              f"{stats['pages']:>8} "
              f"{stats['blocks']:>8} "
              f"{format_number(stats['tokens']):>12} "
              f"{format_number(stats['words']):>12}")
        
        total_pages += stats['pages']
        total_blocks += stats['blocks']
        total_tokens += stats['tokens']
        total_words += stats['words']
    
    print("-" * 80)
    print(f"{'TOTAL':<40} "
          f"{total_pages:>8} "
          f"{total_blocks:>8} "
          f"{format_number(total_tokens):>12} "
          f"{format_number(total_words):>12}")
    print()
    
    # Summary
    print("SUMMARY:")
    print("-" * 80)
    print(f"Files processed:     {len(results) - error_count}")
    print(f"Files with errors:   {error_count}")
    print(f"Total pages:         {format_number(total_pages)}")
    print(f"Total text blocks:   {format_number(total_blocks)}")
    print(f"Total tokens:        {format_number(total_tokens)}")
    print(f"Total words:         {format_number(total_words)}")
    
    if total_pages > 0:
        print(f"\nAverage tokens/page: {total_tokens / total_pages:.1f}")
        print(f"Average words/page:  {total_words / total_pages:.1f}")
    
    if total_blocks > 0:
        print(f"Average tokens/block: {total_tokens / total_blocks:.1f}")
        print(f"Average words/block:  {total_words / total_blocks:.1f}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Count tokens in extraction JSON files'
    )
    parser.add_argument('directory', nargs='?', 
                       default='/mnt/tank/RAG_Data_Microscopy/',
                       help='Directory to scan (default: /mnt/tank/RAG_Data_Microscopy/)')
    parser.add_argument('-r', '--recursive', action='store_true',
                       help='Scan subdirectories recursively')
    parser.add_argument('--csv', help='Export results to CSV file')
    
    args = parser.parse_args()
    
    directory = Path(args.directory)
    
    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        return 1
    
    if not directory.is_dir():
        print(f"Error: Not a directory: {directory}")
        return 1
    
    # Analyze
    print(f"Scanning: {directory}")
    if args.recursive:
        print("Mode: Recursive")
    print()
    
    results = analyze_directory(directory, args.recursive)
    
    if not results:
        print("No JSON files found.")
        return 0
    
    # Print report
    print_report(results, directory)
    
    # Export CSV if requested
    if args.csv:
        import csv
        with open(args.csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['File', 'Pages', 'Blocks', 'Tokens', 'Words', 'Error'])
            
            for filepath, stats in results:
                rel_path = filepath.relative_to(directory)
                writer.writerow([
                    str(rel_path),
                    stats['pages'],
                    stats['blocks'],
                    stats['tokens'],
                    stats['words'],
                    stats['error'] or ''
                ])
        
        print(f"\n✓ Exported to: {args.csv}")
    
    return 0


if __name__ == '__main__':
    exit(main())
