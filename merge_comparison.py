#!/usr/bin/env python3
"""Compare and merge triton-generator-merge into this repository."""

import shutil
from pathlib import Path

source = Path("/home/shravan/Documents/Projects/tmp/segfault/triton-generator-merge")
target = Path("/home/shravan/Documents/Projects/segfault-triton-toyisa")

def get_all_files(base, exclude_dirs=None):
    """Get all files relative to base, excluding certain directories."""
    if exclude_dirs is None:
        exclude_dirs = {'.git', '__pycache__', '.pytest_cache', 'build', 'dist', '.eggs'}
    
    files = []
    for item in base.rglob('*'):
        if item.is_file():
            # Check if any parent is in exclude_dirs
            if any(part in exclude_dirs for part in item.parts):
                continue
            rel_path = item.relative_to(base)
            files.append(rel_path)
    return sorted(files)

print("Comparing directories...")
print(f"Source: {source}")
print(f"Target: {target}")
print()

source_files = get_all_files(source)
target_files = get_all_files(target)

source_set = set(source_files)
target_set = set(target_files)

new_files = source_set - target_set
missing_files = target_set - source_set
common_files = source_set & target_set

print(f"=== NEW FILES in source (will be copied): {len(new_files)} ===")
for f in sorted(new_files):
    print(f"  + {f}")

print(f"\n=== FILES only in target (not in source): {len(missing_files)} ===")
for f in sorted(list(missing_files)[:20]):  # Show first 20
    print(f"  - {f}")
if len(missing_files) > 20:
    print(f"  ... and {len(missing_files) - 20} more")

print(f"\n=== Common files: {len(common_files)} ===")

# Check for files that differ
different_files = []
for rel_path in common_files:
    source_file = source / rel_path
    target_file = target / rel_path
    
    try:
        source_content = source_file.read_bytes()
        target_content = target_file.read_bytes()
        
        if source_content != target_content:
            different_files.append(rel_path)
    except Exception as e:
        print(f"Error comparing {rel_path}: {e}")

print(f"\n=== DIFFERENT FILES (content differs): {len(different_files)} ===")
for f in sorted(different_files):
    print(f"  ~ {f}")

# Copy new files
if new_files:
    print(f"\n=== COPYING {len(new_files)} new files ===")
    for rel_path in sorted(new_files):
        source_file = source / rel_path
        target_file = target / rel_path
        
        # Create parent directory if needed
        target_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            shutil.copy2(source_file, target_file)
            print(f"  Copied: {rel_path}")
        except Exception as e:
            print(f"  ERROR copying {rel_path}: {e}")

print("\n=== MERGE COMPLETE ===")
print(f"New files copied: {len(new_files)}")
print(f"Different files: {len(different_files)} (NOT automatically updated)")
print("\nTo update different files, review them manually:")
for f in sorted(different_files):
    print(f"  diff {target/f} {source/f}")
