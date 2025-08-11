"""Temporary file cleanup utilities."""
from __future__ import annotations

import os
import shutil
import time
from collections.abc import Generator
from contextlib import contextmanager

from app.core.logging import get_logger

__all__ = ["cleanup_old_files", "temp_file_manager", "cleanup_outputs_directory"]

logger = get_logger(__name__)


def cleanup_old_files(directory: str, max_age_hours: int = 24) -> int:
    """Clean up files older than max_age_hours in the given directory.

    Args:
        directory: Directory path to clean
        max_age_hours: Maximum age in hours before files are deleted

    Returns:
        Number of files deleted
    """
    if not os.path.exists(directory):
        return 0

    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    deleted_count = 0

    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > max_age_seconds:
                        os.remove(file_path)
                        deleted_count += 1
                        logger.debug(f"Deleted old file: {file_path}")
                except OSError as e:
                    logger.warning(f"Failed to delete file {file_path}: {e}")

            # Clean up empty directories
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    if not os.listdir(dir_path):  # Directory is empty
                        os.rmdir(dir_path)
                        logger.debug(f"Deleted empty directory: {dir_path}")
                except OSError:
                    pass  # Directory not empty or other issue

    except OSError as e:
        logger.error(f"Error during cleanup of {directory}: {e}")

    if deleted_count > 0:
        logger.info(f"Cleaned up {deleted_count} old files from {directory}")

    return deleted_count


def cleanup_outputs_directory(tmp_dir: str, max_files: int = 100) -> int:
    """Clean up the outputs directory, keeping only the most recent files.

    Args:
        tmp_dir: Base temporary directory
        max_files: Maximum number of files to keep

    Returns:
        Number of files deleted
    """
    outputs_dir = os.path.join(tmp_dir, "outputs")
    if not os.path.exists(outputs_dir):
        return 0

    try:
        # Get all files with their modification times
        files_with_times = []
        for root, _, files in os.walk(outputs_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    mtime = os.path.getmtime(file_path)
                    files_with_times.append((file_path, mtime))
                except OSError:
                    continue

        # Sort by modification time (newest first)
        files_with_times.sort(key=lambda x: x[1], reverse=True)

        # Delete files beyond the limit
        deleted_count = 0
        for file_path, _ in files_with_times[max_files:]:
            try:
                os.remove(file_path)
                deleted_count += 1
                logger.debug(f"Deleted old output file: {file_path}")
            except OSError as e:
                logger.warning(f"Failed to delete file {file_path}: {e}")

        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old output files")

        return deleted_count

    except OSError as e:
        logger.error(f"Error during outputs cleanup: {e}")
        return 0


@contextmanager
def temp_file_manager(base_dir: str) -> Generator[str, None, None]:
    """Context manager for creating and automatically cleaning up temporary directories.

    Args:
        base_dir: Base directory for temporary files

    Yields:
        Path to temporary directory
    """
    import tempfile

    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp(dir=base_dir)
        logger.debug(f"Created temporary directory: {temp_dir}")
        yield temp_dir
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.debug(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temporary directory {temp_dir}: {e}")
