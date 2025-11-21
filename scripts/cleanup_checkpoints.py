#!/usr/bin/env python3
"""
Cleanup old training checkpoints while preserving important ones.

Usage:
    python scripts/cleanup_checkpoints.py --keep 3
    python scripts/cleanup_checkpoints.py --dry-run
    python scripts/cleanup_checkpoints.py --archive best_run
"""
import argparse
import shutil
from pathlib import Path
from datetime import datetime
import sys


def get_checkpoint_info(checkpoint_dir):
    """Extract timestamp and metadata from checkpoint directory."""
    name = checkpoint_dir.name

    # Get modification time
    mtime = checkpoint_dir.stat().st_mtime
    mod_time = datetime.fromtimestamp(mtime)

    # Get size
    size_mb = sum(f.stat().st_size for f in checkpoint_dir.rglob('*') if f.is_file()) / (1024 * 1024)

    # Determine type
    if 'final' in name:
        checkpoint_type = 'final'
    elif 'epoch' in name:
        checkpoint_type = 'epoch'
    elif 'step' in name:
        checkpoint_type = 'step'
    else:
        checkpoint_type = 'unknown'

    return {
        'path': checkpoint_dir,
        'name': name,
        'mod_time': mod_time,
        'size_mb': size_mb,
        'type': checkpoint_type
    }


def list_checkpoints(output_dir):
    """List all checkpoints in output directory."""
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"Output directory not found: {output_dir}")
        return []

    checkpoints = []
    for item in output_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            checkpoints.append(get_checkpoint_info(item))

    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda x: x['mod_time'], reverse=True)

    return checkpoints


def cleanup_checkpoints(output_dir, keep_count=3, dry_run=False, verbose=True):
    """
    Remove old checkpoints, keeping only the most recent ones.

    Args:
        output_dir: Path to outputs directory
        keep_count: Number of most recent checkpoints to keep
        dry_run: If True, only show what would be deleted
        verbose: Print detailed information
    """
    checkpoints = list_checkpoints(output_dir)

    if not checkpoints:
        print("No checkpoints found.")
        return

    # Separate final checkpoints (always keep at least 1)
    final_checkpoints = [c for c in checkpoints if c['type'] == 'final']
    other_checkpoints = [c for c in checkpoints if c['type'] != 'final']

    # Keep most recent final + keep_count others
    keep_final = final_checkpoints[:1]  # Keep most recent final
    keep_others = other_checkpoints[:keep_count]

    to_keep = keep_final + keep_others
    to_delete = [c for c in checkpoints if c not in to_keep]

    if verbose:
        print("\n" + "="*70)
        print("CHECKPOINT CLEANUP")
        print("="*70)
        print(f"\nFound {len(checkpoints)} total checkpoints")
        print(f"Keeping {len(to_keep)} (1 final + {keep_count} others)")
        print(f"Removing {len(to_delete)}")

        total_size = sum(c['size_mb'] for c in checkpoints)
        keep_size = sum(c['size_mb'] for c in to_keep)
        delete_size = sum(c['size_mb'] for c in to_delete)

        print(f"\nTotal size: {total_size:.1f} MB")
        print(f"Keeping: {keep_size:.1f} MB")
        print(f"Freeing: {delete_size:.1f} MB")

        print("\n" + "="*70)
        print("KEEPING:")
        print("="*70)
        for c in to_keep:
            print(f"  ✓ {c['name']:<50} {c['size_mb']:>6.1f} MB  ({c['mod_time'].strftime('%Y-%m-%d %H:%M')})")

        if to_delete:
            print("\n" + "="*70)
            print("DELETING:")
            print("="*70)
            for c in to_delete:
                print(f"  ✗ {c['name']:<50} {c['size_mb']:>6.1f} MB  ({c['mod_time'].strftime('%Y-%m-%d %H:%M')})")

    if not to_delete:
        print("\nNothing to delete!")
        return

    if dry_run:
        print("\n[DRY RUN] No files were actually deleted.")
        return

    # Confirm deletion
    response = input(f"\nDelete {len(to_delete)} checkpoints? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    # Delete checkpoints
    for c in to_delete:
        print(f"Deleting {c['name']}...")
        shutil.rmtree(c['path'])

    print(f"\n✓ Deleted {len(to_delete)} checkpoints, freed {delete_size:.1f} MB")


def archive_checkpoint(output_dir, checkpoint_name, archive_name):
    """
    Archive a specific checkpoint to a named directory.

    Useful for preserving best runs before cleanup.
    """
    output_path = Path(output_dir)
    checkpoint_path = output_path / checkpoint_name
    archive_dir = output_path / "archive"
    archive_path = archive_dir / archive_name

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_name}")
        return False

    archive_dir.mkdir(exist_ok=True)

    if archive_path.exists():
        print(f"Archive already exists: {archive_name}")
        response = input("Overwrite? [y/N]: ")
        if response.lower() != 'y':
            return False
        shutil.rmtree(archive_path)

    print(f"Archiving {checkpoint_name} → archive/{archive_name}")
    shutil.copytree(checkpoint_path, archive_path)

    size_mb = sum(f.stat().st_size for f in archive_path.rglob('*') if f.is_file()) / (1024 * 1024)
    print(f"✓ Archived ({size_mb:.1f} MB)")

    return True


def main():
    parser = argparse.ArgumentParser(description="Manage training checkpoints")
    parser.add_argument('--output-dir', default='outputs/qwen3_refcoco',
                       help='Output directory containing checkpoints')
    parser.add_argument('--keep', type=int, default=3,
                       help='Number of recent checkpoints to keep (default: 3)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be deleted without actually deleting')
    parser.add_argument('--archive', type=str,
                       help='Archive a checkpoint before cleanup (format: checkpoint_name:archive_name)')
    parser.add_argument('--list', action='store_true',
                       help='List all checkpoints and exit')

    args = parser.parse_args()

    # List checkpoints
    if args.list:
        checkpoints = list_checkpoints(args.output_dir)
        if not checkpoints:
            return

        print("\n" + "="*70)
        print("CHECKPOINTS")
        print("="*70)
        total_size = 0
        for c in checkpoints:
            print(f"{c['name']:<50} {c['size_mb']:>6.1f} MB  ({c['mod_time'].strftime('%Y-%m-%d %H:%M')})")
            total_size += c['size_mb']
        print("="*70)
        print(f"Total: {len(checkpoints)} checkpoints, {total_size:.1f} MB")
        return

    # Archive checkpoint
    if args.archive:
        if ':' not in args.archive:
            print("Archive format should be: checkpoint_name:archive_name")
            print("Example: --archive checkpoint_20231121_143022_final:best_v1")
            return

        checkpoint_name, archive_name = args.archive.split(':', 1)
        archive_checkpoint(args.output_dir, checkpoint_name, archive_name)
        print("\nNow run cleanup if desired: python scripts/cleanup_checkpoints.py")
        return

    # Cleanup
    cleanup_checkpoints(args.output_dir, keep_count=args.keep, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
